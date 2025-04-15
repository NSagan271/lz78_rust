import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import Union
from lz78 import LZ78SPA, Sequence, CharacterMap, spa_from_file
import gc
from enum import IntEnum
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card import SentenceTransformerModelCardData
import numpy as np
from openai import OpenAI
import re
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
from model2vec.distill.inference import create_output_embeddings_from_model_and_tokens
from model2vec.distill.tokenizer import remove_tokens


class WeightType(IntEnum):
    UNIFORM = 0
    INV_PROB = 1
    PROB = 2
    LOG_LOSS = 3
    ZIPF = 4


class EmbeddingType(IntEnum):
    TRANSFORMERS = 0
    OPENAI = 1


def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    """
    From documentation of https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class DeepEmbeddingModel(torch.nn.Module):
    def __init__(self, model_str: str, model_type: int, tokenizer_str: str = None, max_length=8192, device="cpu"):
        super().__init__()

        if tokenizer_str is None:
            tokenizer_str = model_str

        self.model_type = model_type
        if model_type == EmbeddingType.TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str, trust_remote_code=True, device=device)
            self.model = AutoModel.from_pretrained(model_str, trust_remote_code=True, device_map=device)
        elif model_type == EmbeddingType.OPENAI:
            self.model = model_str
            self.client = OpenAI()
        else:
            raise NotImplementedError()
        
        self.max_length = max_length
        self.to(device)

    def embed(self, text: Union[str, list[str]], normalize=True):
        if isinstance(text, str):
            text = [text]
        if self.model_type == EmbeddingType.TRANSFORMERS:
            with torch.no_grad():
                batch_dict = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
                outputs = self.model(**batch_dict)

                embeds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                if normalize:
                    return F.normalize(embeds, p=2, dim=1)
                return embeds
        elif self.model_type == EmbeddingType.OPENAI:
            return torch.Tensor([
                data.embedding for data in self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            ).data])
        

class LZPlusEmbeddingModel(SentenceTransformer):
    def __init__(
        self, inner_model_name: str,
        valid_character_string: str,
        inner_model_type = EmbeddingType.TRANSFORMERS,
        max_depth: int = None,
        tokenizer_name: str = None,
        backshift_and_ensemble: bool = True,
        backshift_parsing_len: int = 6,
        ensemble_n: int = 6,
        max_batch_size: int=256,
        weight_type=WeightType.UNIFORM,
        make_lowercase=True,
        pca = False,
        device = "cpu"
    ):
        super().__init__()
        self.make_lowercase = make_lowercase
        self.charmap = CharacterMap(valid_character_string)
        self.alphabet_size = self.charmap.alphabet_size()
        self.model = DeepEmbeddingModel(
            inner_model_name, inner_model_type, tokenizer_name, device=device)
        self.to(device)

        self.lz_trained = False
        self.spa = LZ78SPA(alphabet_size=self.alphabet_size,
                           compute_training_loss=False,
                           max_depth=max_depth)
        if backshift_and_ensemble:
            self.spa.set_inference_config(
                backshift_parsing=True,
                backshift_ctx_len=backshift_parsing_len,
                ensemble_n=ensemble_n,
                ensemble_type="entropy",
            )

        # compute fixed length
        self.length = self.model.embed("hello").shape[1]
        self.inner_model_length = self.length
        
        self.max_batch_size = max_batch_size
        self.weight_type = weight_type

        self.model_card_data = SentenceTransformerModelCardData(
            language="eng-Latn",
            model_name=f"lz/lz78_plus_{inner_model_name.replace('/', '_')}"
        )
        self.debug_ = False
        self.subspace = None
        self.pca = pca

    def set_config(
        self,
        backshift_parsing_len=None,
        ensemble_n = None,
        ensemble_type = None,
        pca = None,
        max_batch_size = None,
        weight_type = None
    ):
        self.spa.set_inference_config(
            backshift_ctx_len=backshift_parsing_len,
            ensemble_type=ensemble_type,
            ensemble_n=ensemble_n
        )
        
        if pca is not None:
            self.pca = pca
            if pca and self.subspace is not None:
                self.length = self.subspace.shape[1]
            else:
                self.length = self.inner_model_length
        
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size

        if weight_type is not None:
            self.weight_type = weight_type

    def debug(self, debug: bool):
        self.debug_ = debug

    def get_max_seq_length(self) -> int | None:
        return None

    def tokenize_one(self, text: str):
        return torch.tensor(self.charmap.encode(
            re.sub('\s{2,}', ' ', self.charmap.filter_string(text))
        ), dtype=torch.uint16, device=self.device)

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]]):
        if type(texts) == list and type(texts[0]) == str:
            if self.make_lowercase:
                texts = [text.lower() for text in texts]
            tokenized = [self.tokenize_one(text) for text in texts]
            lengths = [len(t) for t in tokenized]
            res = torch.zeros((len(lengths), max(lengths)), dtype=torch.uint16, device=self.device)
            mask = torch.zeros_like(res)
            for (i, l) in enumerate(lengths):
                mask[0:l] = 1
                res[i, 0:l] = tokenized[i]

            return {
                "input_ids": res,
                "lengths": lengths,
                "attention_mask": mask
            }
        else:
            raise NotImplementedError()
        
    def train_spa(self, sequence: str):
        tokenized = self.tokenize([sequence])["input_ids"][0].tolist()
        
        self.spa.reset_state()
        self.spa.train_on_block(Sequence(tokenized, alphabet_size=self.alphabet_size))
        self.lz_trained = True

    def compute_subspace(
        self, dimension: int,
        num_gen_seqs: int = 1000,
        gen_seq_len: int = 100,
        backshift_len: int = 3,
        ensemble_n: int = 3,
        gen_temperature: float = 0.3,
        gen_top_k: int = 10,
        standardize: bool = False,
        enable_low_rank_projection: bool = True
    ):
        pca_enabled_before = self.pca
        self.pca = False
        self.length = self.inner_model_length

        self.spa.set_generation_config(
            ensemble_n=ensemble_n, ensemble_type="depth",
            backshift_ctx_len=backshift_len
        )
        sequences = [self.spa.generate_data(
            gen_seq_len,
            temperature=gen_temperature,
            top_k=gen_top_k
        )[0] for _ in range(num_gen_seqs)]

        old_backshift_len = self.spa.get_inference_config()["backshift_ctx_len"]
        self.spa.set_inference_config(backshift_ctx_len=0)

        spa_out = self.spa.compute_test_loss_parallel(sequences, output_patch_info=True, output_per_symbol_losses=True)
        texts = []
        for (sequence, res) in zip(sequences, spa_out):
            patches = res["patch_info"]
            texts.extend([self.charmap.decode(sequence.get_data())[start:end+1] for (start, end) in patches])
        embeds = torch.zeros((len(texts), self.length), device=self.device)
        for i in tqdm(range(0, len(texts), self.max_batch_size)):
            embeds[i:i+self.max_batch_size, :] = self.model.embed(
                texts[i:i+self.max_batch_size], normalize=True)
        
        if standardize:
            mean = torch.mean(embeds, dim=0)
            std = torch.std(embeds, dim=0)
            embeds = (embeds - mean) / std
        _, _, self.subspace = torch.svd_lowrank(embeds, q=dimension*2)
        self.subspace = self.subspace[:, :dimension]

        self.pca = pca_enabled_before
        if enable_low_rank_projection:
            self.pca = True
            self.length = dimension

        self.spa.set_inference_config(backshift_ctx_len=old_backshift_len)
    
    def compute_embeddings_from_sequences(self, inputs: list[Sequence]) -> Tensor:
        spa_out = self.spa.compute_test_loss_parallel(inputs, output_patch_info=True, output_per_symbol_losses=True)

        projection = torch.eye(self.length, device=self.device)
        if self.pca and self.subspace is not None:
            projection = self.subspace

        embeds = torch.zeros((len(spa_out), self.length), device=self.device)
        for (seq_idx, res) in enumerate(spa_out):
            patches = res["patch_info"]
            log_losses = torch.Tensor(res["log_losses"])

            if self.weight_type == WeightType.UNIFORM:
                weights = torch.ones(len(patches)).to(self.device)
            elif self.weight_type == WeightType.INV_PROB:
                weights = torch.Tensor([(2**log_losses[start:end+1]).mean() for (start, end) in patches]).to(self.device)
            elif self.weight_type == WeightType.PROB:
                weights = torch.Tensor([(2**(-log_losses[start:end+1])).mean() for (start, end) in patches]).to(self.device)
            else:
                raise NotImplementedError()
            weights /= weights.sum()

            sequence = self.charmap.decode(inputs[seq_idx].get_data())
            texts = [sequence[start:end+1] for (start, end) in patches]
            if self.debug_:
                print("-"*20, "DEBUG", "-"*20)
                print("Phrases recorded: ", texts)
                print("Weights: ", weights)
                print("Per sym log losses: ", log_losses)
                print("-"*46)
                

            for i in range(0, len(texts), self.max_batch_size):
                embeds[seq_idx, :] += (weights[i:i+self.max_batch_size].unsqueeze(1) * self.model.embed(
                    texts[i:i+self.max_batch_size], normalize=True) @ projection).sum(axis=0)
            gc.collect()
            torch.cuda.empty_cache()

        
        return embeds

    def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        assert self.lz_trained, "Must train LZPlusEmbeddingModel with model.train_spa(sequence) first!"

        inputs = [
            Sequence(
                tokenized[:length].tolist(), alphabet_size=self.alphabet_size
            ) for (tokenized, length) in zip(input["input_ids"], input["lengths"]) 
        ]
        
        return {
            "sentence_embedding": self.compute_embeddings_from_sequences(inputs)
        }
    

class TokenizedLZPlusEmbedding(SentenceTransformer):
    def __init__(
        self, inner_model_name: str,
        output_dir: str,
        tokenizer_name: str = None,
        backshift_and_ensemble: bool = True,
        backshift_parsing_len: int = 6,
        ensemble_n: int = 6,
        max_batch_size: int=256,
        weight_type=WeightType.UNIFORM,
        pca = False,
        pca_dim = 512,
        compute_device = "cpu",
        overwrite_objects=False
    ):
        super().__init__()
        model_name = f"lz2/lz78_plus_{inner_model_name.replace('/', '_')}"
        
        self.output_dir = f"{output_dir}/{model_name}"

        if tokenizer_name is None:
            tokenizer_name = inner_model_name

        self.tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, device=compute_device)
        self.valid_chars = "abcdefghijklmnopqrstuvwxyz1234567890 "
        self.charmap = CharacterMap(self.valid_chars)
        self.make_lowercase = True

        self.weight_type = weight_type
        
        full_vocab = [pair[0] for pair in sorted(self.tokenizer_.get_vocab().items(), key=lambda x: x[1])]
        vocab = [x for x in full_vocab if not re.match("\[unused\d+\]", x)]
        
        _, self.embeds = create_output_embeddings_from_model_and_tokens(
            model=AutoModel.from_pretrained(inner_model_name),
            tokenizer=self.tokenizer_,
            tokens=vocab,
            device=compute_device,
        )

        self.embeds = torch.Tensor(self.embeds)

        self.tokenizer_ = remove_tokens(self.tokenizer_.backend_tokenizer, set(full_vocab) - set(vocab))
        self.num_tokens = len(vocab)

        self.pca = pca
        self.pca_dim = pca_dim
        
        p = PCA(n_components=pca_dim, svd_solver="full")
        self.pca_embeds = torch.Tensor(p.fit_transform(self.embeds))

        # Zipf weights from Model2Vec repository
        sif_coefficient=1e-4
        inv_rank = 1 / (np.arange(2, self.embeds.shape[0] + 2))
        proba = inv_rank / np.sum(inv_rank)
        self.zipf_weights = torch.Tensor(sif_coefficient / (sif_coefficient + proba))

        self.spa_file = f"{self.output_dir}/spa.bin"
        os.makedirs(os.path.dirname(self.spa_file), exist_ok=True)
        if  os.path.isfile(self.spa_file) and not overwrite_objects:
            self.spa = spa_from_file(self.spa_file)
            self.lz_trained = True
        else: 
            self.lz_trained = False
            self.spa = LZ78SPA(alphabet_size=self.charmap.alphabet_size(),
                            compute_training_loss=False)

        if backshift_and_ensemble:
            self.spa.set_inference_config(
                backshift_parsing=True,
                backshift_ctx_len=backshift_parsing_len,
                ensemble_n=ensemble_n,
                ensemble_type="entropy",
            )

        self.model_card_data = SentenceTransformerModelCardData(
            language="eng-Latn",
            model_name=model_name
        )

    def _embed_with_model(self, model: AutoModel, text: str, normalize=True):
        with torch.no_grad():
            batch_dict = self.tokenizer_(text, padding=True, truncation=True, return_tensors='pt').to(self.compute_device)
            outputs = model(**batch_dict)

            embeds = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            if normalize:
                return F.normalize(embeds, p=2, dim=1)
            return embeds

    def tokenize(self, texts):
        self.tokenizer_.no_padding()
        encoded = self.tokenizer_.encode_batch([x.lower() for x in texts], add_special_tokens=False)
        return {
            "input_ids": [x.ids for x in encoded],
            "offsets": [x.offsets for x in encoded],
            "text": texts
        }
    
    def decode_filter(self, ids, offsets, text):
        result = ""
        lengths = []
        last_nonzero_length = -1
        for i in range(len(ids)):
            (start, end) = offsets[i]
            filt = "".join(x for x in text[start:end].lower() if x in self.valid_chars)
            if len(filt) == 0:
                lengths.append(0)
                continue
            last_nonzero_length = i
            if i != len(ids) - 1 and offsets[i+1][0] > offsets[i][1]:
                result += filt + " "
                lengths.append(len(filt) + 1)
            else:
                result += filt
                lengths.append(len(filt))
        if len(result) > 0 and result[-1] == " ":
            result = result[:-1]
            lengths[last_nonzero_length] -= 1

        assert len(result) == sum(lengths)
        return result, lengths
    

    def train_spa(self, sequence: str | list[str], save=False):
        if type(sequence) == str:
            sequence = [sequence]
        filtered = ["".join(x for x in elem.lower() if x in self.valid_chars) for elem in sequence]

        for row in filtered:
            self.spa.reset_state()
            self.spa.train_on_block(Sequence(row, charmap=self.charmap))

        self.lz_trained = True
        if save:
            print(f"Saving SPA to {self.spa_file}")
            os.makedirs(os.path.dirname(self.spa_file), exist_ok=True)
            self.spa.to_file(self.spa_file)
            print("Done saving")

    def forward(self, input: dict[str, list],**kwargs) -> dict[str, Tensor]:
        assert self.lz_trained, "Must train LZPlusEmbeddingModel with model.train_spa(sequences) first!"
        
        lz_inputs = []
        char_boundaries = []
        for (input_ids, offset, text) in zip(input["input_ids"], input["offsets"], input["text"]):
            detokenized, lengths = self.decode_filter(input_ids, offset, text)
            char_boundaries.append([0] + np.cumsum(lengths).tolist())
            lz_inputs.append(Sequence(detokenized, charmap=self.charmap))
   
        spa_out = self.spa.compute_test_loss_parallel(lz_inputs, output_per_symbol_losses=True)

        if self.pca:
            token_embeds = self.pca_embeds
        else:
            token_embeds = self.embeds

        embeds = torch.zeros((len(lz_inputs), token_embeds.shape[1]))
        
        for i in range(len(lz_inputs)):
            tokens = input["input_ids"][i]

            weights = torch.zeros(len(tokens))
            for j in range(len(tokens)):
                if self.weight_type == WeightType.UNIFORM:
                    weights[j] = 1
                elif self.weight_type == WeightType.ZIPF:
                    weights[j] = self.zipf_weights[tokens[j]]
                elif self.weight_type == WeightType.LOG_LOSS or WeightType.INV_PROB:
                    if char_boundaries[i][j+1] == char_boundaries[i][j]:
                        continue
                    weights[j] = max(spa_out[i]["log_losses"][char_boundaries[i][j]:char_boundaries[i][j+1]])

                    if weights[j] != weights[j]: # no nans allowed
                        weights[j] = 1
                    # print(
                    #     weights[j],
                    #     lz_inputs[i].get_data()[char_boundaries[i][j]:char_boundaries[i][j+1]]
                    # )
                else:
                    raise NotImplementedError()
            if self.weight_type == WeightType.INV_PROB:
                weights = 2**weights
        
            embeds[i, :] = (token_embeds[tokens, :] * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        embeds /= (embeds.norm(dim=1, keepdim=True) + 1e-32)

        embeds = torch.nan_to_num(embeds)

        return {"sentence_embedding": embeds}
