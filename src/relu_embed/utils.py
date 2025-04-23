from transformers import AutoTokenizer
import regex as re
from model2vec.distill.tokenizer import remove_tokens
import numpy as np
from tqdm import tqdm
from model2vec import StaticModel
import torch
from torch.utils.data import TensorDataset


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


def info_nce_loss(x: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, temp=1):
    pos_cosine = torch.nn.CosineSimilarity(x, positive)
    if len(negatives.shape) == 2:
        negatives = negatives.unsqueeze(1)

    neg_cosines = torch.nn.CosineSimilarity(x, negatives, dim=2)

    numer = torch.exp(-pos_cosine/temp)
    return -torch.log(
        numer / (torch.exp(-neg_cosines/temp).sum(dim=1) + numer)
    ).mean()


def randint(low: torch.Tensor, high: torch.Tensor, size: tuple):
    return torch.randint(2**63 - 1, size=size) % (high - low) + low


def classification_data_to_contrastive_dataset(
    Xs: list[torch.Tensor], ys: list[torch.Tensor],
    rows_per_sample: int = 1,
    negatives_per_row: int = 8,
):
    X_list = []
    pos_list = []
    neg_list = []
    for (X, y) in zip(Xs, ys):
        X = X.repeat(rows_per_sample, 1)
        y = y.repeat(rows_per_sample, 1)
        c = int(y.max()) + 1

        idxs_per_class = [torch.where(y == i)[0].tolist() for i in range(c+1)]
        all_idxs = set(range(len(y)))
        neg_idxs_per_class = [list(all_idxs - set(s)) for s in idxs_per_class]

        pos = torch.zeros_like(X)
        neg = torch.zeros(X.shape[0], negatives_per_row, )
        
        for cls in range(c+1):
            idxs = idxs_per_class[cls]
            pos_examples = torch.randint(0, len(idxs_per_class[cls]), size=len(idxs))
            neg_examples = torch.randint(0, len(neg_idxs_per_class[cls]),
                                         size=(negatives_per_row, len(idxs)))
            pos[idxs, :] = X[pos_examples, :]
            neg[idxs, :] = X[neg_examples, :]

        pos_list.append(pos)
        neg_list.append(neg)
        X_list.append(X)
    
    return TensorDataset(
        torch.concat(X_list), torch.concat(pos_list),
        torch.concat(neg_list)
    )


