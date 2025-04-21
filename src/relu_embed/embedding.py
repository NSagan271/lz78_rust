from lz78 import LZ78SPA, Sequence, NGramSPA, CharacterMap
from typing import Union
import torch
from torch import Tensor, nn
from lz_embed.utils import AlphabetInfo, LZEmbedding
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card import SentenceTransformerModelCardData
import torch
from relu_embed.utils import TokenizerWrapper


class NNEmbedding(SentenceTransformer):
    def __init__(
        self, tokenizer: TokenizerWrapper,
        embedding_model: nn.Module,
        embedding_dimesion: int,
        device="cpu",
        normalize_token_counts = True,
        normalize_embeds = False,
        internal_batch_size=16
    ):
        super().__init__()

        self.tokenizer_ = tokenizer
        self.normalize_embeds = normalize_embeds
        self.normalize_token_counts = normalize_token_counts
        self.embedding_model = embedding_model.to(device).eval()
        self.internal_batch_size = internal_batch_size
        self.dim = embedding_dimesion

        self.to(device)

        self.model_card_data = SentenceTransformerModelCardData(
            language="eng-Latn",
            model_name="lz/ngram_spectrum"
        )

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
