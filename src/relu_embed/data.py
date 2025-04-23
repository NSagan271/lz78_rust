from datasets import load_dataset
import numpy as np
from dataclasses import dataclass
import torch
from relu_embed.utils import TokenizerWrapper
from sklearn.model_selection import StratifiedKFold


@dataclass
class DatasetInfo:
    dataset: str
    name: str = None
    train_split: str = "train"
    test_splits: tuple[str] = ("test",)
    has_splits: bool = True
    text_column: str = "text"
    label_column: str = "label"
    train_limit: int = None


def load_and_process_dataset(
    info: DatasetInfo,
    base_model: str = "BAAI/bge-base-en-v1.5",
    seed: int = 0
):
    data = load_dataset(info.dataset, info.name)
    tokenizer = TokenizerWrapper(base_model)
    
    X = tokenizer.get_token_counts(
        data[info.train_split][info.text_column],
        progress=True
    )
    y = torch.Tensor(data[info.train_split][info.label_column])

    if info.train_limit:
        idxs = list(range(X.shape[0]))
        np.random.shuffle(idxs)
        X = X[idxs, :]
        y = y[idxs]

    if info.has_splits:
        Xtest = torch.concat([
            tokenizer.get_token_counts(
                data[test][info.text_column],
                progress=True
            ) for test in info.test_splits
        ])
        ytest = torch.concat([
            torch.Tensor(data[test][info.label_column]) \
                for test in info.test_splits
        ])
    else:
        tr, tst = next(StratifiedKFold(
            n_splits=2, random_state=seed, shuffle=True).split(X, y))
        Xtest = X[tst, :]
        ytest = y[tst]

        X = X[tr, :]
        y = y[tr]

    return X, y, Xtest, ytest