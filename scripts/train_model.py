from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from lz_embed.transformer_based import TokenizedLZPlusEmbedding, TokenizedLZPlusEmbeddingConfig, WeightType
from lz_embed.classical import AlphabetInfo, NGramSpectrumEmbedding, NGramSpectrumEmbeddingConfig
from datatrove.pipeline.readers import ParquetReader
from tqdm import tqdm
import os
from glob import glob
import json
from relu_embed.classification import MultiproblemReLUClassifier
from relu_embed.data import DatasetInfo, load_and_process_dataset
from relu_embed.embedding import NNEmbeddingConfig
import torch
import json


@dataclass
class Arguments:
    model_type: str = field(metadata=dict(
        choices=["lz", "relu", "ngram"],
        help="What type of embedding model to use"
    ))
    model_save_dir: str = field(metadata=dict(
        help="Directory where output files will be stored"))
    base_model: str = field(default="BAAI/bge-base-en-v1.5", metadata=dict(
        help="For lz and relu, the huggingface ID of the base embedding model to use"
    ))
    tokenizer_name: str = field(default=None, metadata=dict(
        help="Tokenizer huggingface ID, if different than the base_model name."
    ))
    device: str = field(default="cpu", metadata=dict(help="cpu or cuda device"))
    embedding_dim: int = field(default=256, metadata=dict(help="Target embedding dimension"))
    overwrite: bool = field(default=False, metadata=dict(
        help="If the same model has already been trained, overwrite that previous model. Otherwise, we continue training"
    ))

    # LZ arguments
    lz_num_train_docs: int = field(default=1_000_000, metadata=dict(
        help="[LZ only] Number of Fineweb documents to use to train the LZ tree"
    ))
    lz_max_spa_size: int = field(default=int(15e9), metadata=dict(
        help="[LZ only] Maximum size (bytes) of the LZ tree"
    ))
    lz_save_interval: int = field(default=10_000, metadata=dict(
        help="[LZ only] Configures how often (in documets) to save the SPA to disk"
    ))
    lz_backshift_and_ensemble: bool = field(default=True, metadata=dict(
        help="[LZ only] Whether to enable backshift parsing and ensemble inference for the LZ SPA"
    ))
    lz_backshift_parsing_len: int = field(default=10, metadata=dict(
        help="[LZ only] Backshift parsing length for LZ SPA"
    ))
    lz_ensemble_size: int = field(default=6, metadata=dict(
        help="[LZ only] Maximum number of LZ tree nodes that the LZ SPA ensemble uses"
    ))
    spa_lower_bound: float = field(default=1e-3, metadata=(dict(
        help="[LZ only] Lowest value the SPA is allowed to take"
    )))
    weight_type: str = field(default="log_loss", metadata=dict(
        choices=["uniform", "log_loss", "zipf"],
        help="[LZ only] Weighting of tokens for computing embedding"
    ))
    lz_pca: bool = field(default=True, metadata=dict(
        help="[LZ only] Whether to perform PCA on base model embeddings to reduce dimension"
    ))
    
    # Ngram arguments
    ngram_pca: bool = field(default=True, metadata=dict(
        help="[NGRAM only] Whether to perform PCA on ngram embeddings of a sample set of documents from Fineweb to reduce the ngram embedding dimension"
    ))
    ngram_num_train_docs: int = field(default=400_000, metadata=dict(
        help="[NGRAM only] Number of documents used for PCA"
    ))

    # ReLU arguments
    classification_problem_json: str = field(default="mteb_info/classification_problems.json", metadata=dict(
        help="[RELU only] Path of JSON file with classification problem information; see mteb_info/classification_problems.json for an example"
    ))
    classifier_batch_size: int = field(default=256, metadata=dict(
        help="[RELU only] Batch size for training multiheaded classifier"
    ))
    classifier_epochs: int = field(default=5, metadata=dict(
        help="[RELU only] Number of epochs to train multiheaded classifier. **If the model has already been trained, this is the number of additional epochs**"
    ))
    classifier_lr: float = field(default=1e-3, metadata=dict(
        help="[RELU only] learning rate for multiheaded classifier"
    ))
    classifier_lr_decay: float = field(default=None)
    eval_iterval: int = field(default=500_000)
    

def train_lz(args: Arguments):
    if args.overwrite and os.path.exists(args.model_save_dir):
        for file in glob(f"{args.model_save_dir}/*"):
            os.remove(file)

    if args.weight_type == "uniform":
        weight_type = WeightType.UNIFORM
    elif args.weight_type == "zipf":
        weight_type = WeightType.ZIPF
    elif args.weight_type == "uniform":
        weight_type = WeightType.UNIFORM
    else:
        raise NotImplementedError("Unknown weight_type")
    
    model = TokenizedLZPlusEmbedding( 
        TokenizedLZPlusEmbeddingConfig(
            inner_model_name=args.base_model,
            tokenizer_name=args.tokenizer_name,
            backshift_and_ensemble=args.lz_backshift_and_ensemble,
            backshift_parsing_len=args.lz_backshift_parsing_len,
            ensemble_n=args.lz_ensemble_size,
            spa_lower_bound=args.spa_lower_bound,
            pca=args.lz_pca,
            pca_dim=args.embedding_dim,
            weight_type=weight_type,
            device=args.device
        ),
        args.model_save_dir
    )

    if model.spa_num_docs_used > 0:
        print(f"Continuing training from {model.spa_num_docs_used} documents")
        if os.path.getsize(model.spa_file) > args.lz_max_spa_size:
            print("SPA is already at max size! Canceling training .")
            return

    data_reader = ParquetReader(
        "hf://datasets/HuggingFaceFW/fineweb/sample/10BT",
        limit=args.lz_num_train_docs-model.spa_num_docs_used,
        skip=model.spa_num_docs_used
    )
    for (i, document) in enumerate(tqdm(data_reader())):
        model.spa.reset_state()
        save = (i % args.lz_save_interval) == args.lz_save_interval - 1
        model.train_spa(document.text, save=save)

        if save and os.path.getsize(model.spa_file) > args.lz_max_spa_size:
            print("SPA reached max size! Exiting train loop.")
            break
    model.save_pretrained()


def train_ngram(args: Arguments):
    if args.overwrite and os.path.exists(args.model_save_dir):
        for file in glob(f"{args.model_save_dir}/*"):
            os.remove(file)

    model = NGramSpectrumEmbedding(
        NGramSpectrumEmbeddingConfig(
            alphabet_info=AlphabetInfo(valid_character_string="abcdefghijklmnopqrstuvwxyz"),
            n=2,
            pca_dim=args.embedding_dim,
            use_pca=args.ngram_pca
        ),
        args.model_save_dir
    )

    if args.ngram_pca:
        data_reader = ParquetReader(
            "hf://datasets/HuggingFaceFW/fineweb/sample/10BT",
            limit=args.ngram_num_train_docs-model.pca_num_docs_used,
            skip=model.pca_num_docs_used
        )
        def get_dataloader(data_reader):
            for doc in data_reader():
                yield doc.text
        model.train_subspace(dataloader=get_dataloader(data_reader), device=args.device)
    model.save_pretrained()


def train_relu(args: Arguments):
    classifier_dir = f"{args.model_save_dir}/classifier"
    classifier_fname = f"{classifier_dir}/model.bin"
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(classifier_dir, exist_ok=True)
    if args.overwrite:
        for file in glob(f"{classifier_dir}/*"):
            os.remove(file)
        os.removedirs(classifier_dir)
        for file in glob(f"{args.model_save_dir}/*"):
            os.remove(file)
        os.makedirs(classifier_dir, exist_ok=True)

    if args.classifier_epochs > 0:
        with open(args.classification_problem_json) as f:
            problems = json.load(f)
        problems = [
            DatasetInfo(**prob) for prob in problems
        ]
        problem_names = [x.dataset for x in problems]

        n_classes = []
        Xs, ys, Xtests, ytests, ids = [], [], [], [], []
        for (i, prob) in enumerate(problems):
            X, y, Xtest, ytest = load_and_process_dataset(prob)
            n_classes.append(int(torch.max(y).item()) + 1)
            Xs.append(X)
            ys.append(y)
            Xtests.append(Xtest)
            ytests.append(ytest)
            ids.append(i)

        if os.path.exists(classifier_fname):
            print("Trained classifier found. Resuming training.")
            classifier = torch.load(classifier_fname, weights_only=False)
        else:
            classifier = MultiproblemReLUClassifier(
                n_classes=n_classes, problem_names=problem_names,
                input_size=Xs[0].shape[1],
                embedding_size=args.embedding_dim,
                device=args.device
            )
        
        train_dataloader = classifier.get_dataloader_batch(
            Xs, ys, ids, batch_size=args.classifier_batch_size, normalize_rows=True
        )
        test_dataloaders = [
            classifier.get_dataloader_single(
                Xtest, ytest, id_test, batch_size=args.classifier_batch_size, normalize_rows=True
            ) for (Xtest, ytest, id_test) in zip(Xtests, ytests, ids)
        ]

        classifier.train(
            train_dataloader, test_dataloaders,
            epochs=args.classifier_epochs,
            lr=args.classifier_lr,
            lr_decay=args.classifier_lr_decay,
            eval_interval=args.eval_iterval
        )

        torch.save(classifier, classifier_fname)
    else:
        classifier = torch.load(classifier_fname, weights_only=False)

    embedding_model = classifier.get_embedding_model(
        NNEmbeddingConfig(
            tokenizer_name=args.tokenizer_name if args.tokenizer_name is not None \
                else args.base_model,
            embedding_dimension=args.embedding_dim,
            normalize_token_counts=True, # these should always be True
            normalize_embeds=True, # and this one too
            device="cpu"
        ), args.model_save_dir
    )
    embedding_model.save_pretrained()
        

if __name__ == "__main__":
    parser = HfArgumentParser((Arguments))
    (args, ) = parser.parse_args_into_dataclasses()
    
    if args.model_type == "lz":
        train_lz(args)
    elif args.model_type == "relu":
        train_relu(args)
    elif args.model_type == "ngram":
        train_ngram(args)
    else:
        raise NotImplementedError("Unknown model_type")
