import torch
from torch import Tensor, nn
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card import SentenceTransformerModelCardData
import torch
from relu_embed.utils import TokenizerWrapper, info_nce_loss
from transformers.configuration_utils import PretrainedConfig
from torch.utils.data import DataLoader, TensorDataset


class NNEmbeddingConfig(PretrainedConfig):
    def __init__(
        self,
        tokenizer_name: str = "BAAI/bge-base-en-v1.5",
        embedding_dimension: int = 256,
        normalize_token_counts: bool = True,
        normalize_embeds: bool = True,
        internal_batch_size: int = 16,
        device="cpu",
        model_name="lz/relu_embed",
        **kwargs
    ):
        self.tokenizer_name = tokenizer_name
        self.embedding_dimension = embedding_dimension
        self.normalize_token_counts = normalize_token_counts
        self.normalize_embeds = normalize_embeds
        self.internal_batch_size = internal_batch_size
        self.model_name = model_name
        self.device = device

        super().__init__(**kwargs)


class NNEmbedding(SentenceTransformer):
    config_class = NNEmbeddingConfig
    base_model_prefix = "relu"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(
        self, config: NNEmbeddingConfig,
        embedding_model: nn.Module,
        model_save_path: str
    ):
        super().__init__()

        self.config = config
        self.model_save_path = model_save_path

        self.tokenizer_ = TokenizerWrapper(config.tokenizer_name)
        self.normalize_embeds = config.normalize_embeds
        self.normalize_token_counts = config.normalize_token_counts
        self.embedding_model = embedding_model.to(config.device).eval()
        self.internal_batch_size = config.internal_batch_size
        self.dim = config.embedding_dimension

        self.to(config.device)

        self.model_card_data = SentenceTransformerModelCardData(
            language="eng-Latn",
            model_name=config.model_name
        )

    def save_pretrained(self):
        self.config.save_pretrained(self.model_save_path)
        torch.save(self.embedding_model, f"{self.model_save_path}/embedding_model.pkl")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = NNEmbeddingConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        embedding_model = torch.load(
            f"{pretrained_model_name_or_path}/embedding_model.pkl",
            weights_only=False
        )
        # Initialize the model
        model = cls(config, embedding_model, pretrained_model_name_or_path)
        return model

    def tokenize(self, texts: str | list[str], progress=False):
        return {
            "ids": self.tokenizer_.tokenize(texts)
        }
    
    def forward(self, input: dict, **kwargs) -> dict[str, Tensor]:
        tokens = input["ids"]
        embeds = torch.zeros((len(tokens), self.dim), device=self.device)
        
        X = torch.zeros(min(self.internal_batch_size, len(tokens)), self.tokenizer_.vocab_size)
        for i in range(0, len(tokens), self.internal_batch_size):
            X[:,:] = 0
            n = min(self.internal_batch_size, len(tokens) - i)
            for (row, enc) in enumerate(tokens[i:i+n]):
                X[row, enc] += 1
            
            if self.normalize_token_counts:
                X = X / (torch.linalg.norm(X, axis=1, keepdims=True) + 1e-32)
            embeds[i:i+n, :] = self.embedding_model(X[:n, :])

        if self.normalize_embeds:
            embeds /= (embeds.norm(dim=1, keepdim=True) + 1e-32)
        return {
            "sentence_embedding": embeds
        }


class ContrastiveReLUEmbedding(SentenceTransformer):
    def __init__(
        self, tokenizer_name: str,
        input_size: int,
        embedding_size: int = 256,
        hidden_size: int = None,
        device = "cpu",
        model_name = "lz/contrastive-relu",
        internal_batch_size = 16,
    ):
        super().__init__()
        self.tokenizer_ = TokenizerWrapper(tokenizer_name)
        if hidden_size is None:
            hidden_size = embedding_size
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        ).to(device)

        self.model_card_data = SentenceTransformerModelCardData(
            language="eng-Latn",
            model_name=model_name
        )

        self.internal_batch_size = internal_batch_size
        self.normalize_token_counts = True

    def tokenize(self, texts: str | list[str], progress=False):
        return {
            "ids": self.tokenizer_.tokenize(texts)
        }
    
    def forward(self, input: dict, **kwargs) -> dict[str, Tensor]:
        tokens = input["ids"]
        embeds = torch.zeros((len(tokens), self.dim), device=self.device)
        
        X = torch.zeros(min(self.internal_batch_size, len(tokens)), self.tokenizer_.vocab_size)
        for i in range(0, len(tokens), self.internal_batch_size):
            X[:,:] = 0
            n = min(self.internal_batch_size, len(tokens) - i)
            for (row, enc) in enumerate(tokens[i:i+n]):
                X[row, enc] += 1
            
            if self.normalize_token_counts:
                X = X / (torch.linalg.norm(X, axis=1, keepdims=True) + 1e-32)
            embeds[i:i+n, :] = self.model(X[:n, :])

        return {
            "sentence_embedding": embeds
        }

    def train(
        self, train_dataloader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        eval_fn = None
    ):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        
        with torch.no_grad():
            print("Test accuracy: ", eval_fn(self.model))

        for _ in tqdm(range(epochs)):
            for (X, pos, neg) in train_dataloader:
                X_emb = self.model(X.to(self.device))
                pos_emb = self.model(pos.to(self.device))
                neg_emb = self.model(neg.to(self.device))

                loss = info_nce_loss(X_emb, pos_emb, neg_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                print("Test accuracy: ", eval_fn(self.model))