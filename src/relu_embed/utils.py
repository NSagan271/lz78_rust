from transformers import AutoTokenizer
import regex as re
from model2vec.distill.tokenizer import remove_tokens
import numpy as np
from tqdm import tqdm
from model2vec import StaticModel
import torch


class TokenizerWrapper:
    def __init__(self, tokenizer_name: str, potion_model_name: str=None, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, device=device)
        full_vocab = [pair[0] for pair in sorted(self.tokenizer.get_vocab().items(), key=lambda x: x[1])]
        vocab = [x for x in full_vocab if not re.match("\[unused\d+\]", x)]
        self.vocab_size = len(vocab)
        self.tokenizer = remove_tokens(self.tokenizer.backend_tokenizer, set(full_vocab) - set(vocab))
        self.tokenizer.no_padding()

        if potion_model_name is not None:
            model = StaticModel.from_pretrained(potion_model_name)
            self.projection = torch.Tensor(model.embedding).to(self.device)
        else:
            self.projection = None

        self.device = device
        self.tokenizer

    def to(self, device):
        self.device = device
        if self.projection is not None:
            self.projection.to(device)

    def tokenize(self, texts: list[str]):
        if type(texts) == str:
            texts = [texts]
        encoded = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        return [enc.ids for enc in encoded]

    def get_token_counts(self, texts: list[str] | str, progress=False):
        encoded = self.tokenize(texts)
        token_counts = torch.zeros((len(texts), self.vocab_size), device=self.device)

        iterator = enumerate(encoded)
        if progress:
            iterator = enumerate(tqdm(encoded))
        for (row, enc) in iterator:
            token_counts[row, enc] += 1
        return token_counts
    
    def get_token_count_projection(self, texts: list[str] | str, progress=False):
        assert self.projection is not None
        counts = self.get_token_counts(texts, progress=progress)
        return counts @ self.projection
