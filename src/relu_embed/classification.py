from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from sys import stdout
from relu_embed.embedding import NNEmbedding
from relu_embed.utils import TokenizerWrapper


class ReLUClassifier:
    def __init__(
        self, n_classes: int,
        input_size: int,
        hidden_size=256, device="cpu"
    ):
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.device = device
        self.input_size = input_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
            nn.Softmax(dim=1),
        ).to(device)
        self.last_epoch = -1

    def to(self, device):
        self.model.to(device)
        self.device = device

    def eval_accuracy(self, test_dataloader: DataLoader):
        errors = []
        for (Xt, yt) in test_dataloader:
            errors.append((
                self.model(Xt.to(self.device)).argmax(dim=1) != yt.to(self.device)
            ).to(torch.float).mean().item())
        return 1 - np.mean(errors)

    def get_dataloader(
        self,
        X: np.array, y: np.array,
        batch_size: int = 64,
        normalize_rows=True
    ):
        if normalize_rows:
            X = X / np.maximum(1, np.linalg.norm(X, axis=1, keepdims=True))
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y).to(torch.long))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        lr_decay: float = None,
    ):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
        if lr_decay is not None:
            lr_scheduler = ExponentialLR(
                optimizer, gamma=lr_decay
            )
        else:
            lr_scheduler = None

        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            print("Test accuracy: ", self.eval_accuracy(test_dataloader))

        for _ in tqdm(range(epochs)):
            for (X, y) in train_dataloader:
                output = self.model(X.to(self.device))
                loss = loss_fn(output, y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.last_epoch += 1
            if lr_scheduler is not None:
                lr_scheduler.step()
            with torch.no_grad():
                print("Test accuracy: ", self.eval_accuracy(test_dataloader))


class MultiproblemReLUClassifier:
    def __init__(
        self, n_classes: list[int],
        problem_names: list[str],
        input_size: int,
        embedding_size: int=256,
        hidden_size: int = None,
        device="cpu"
    ):
        if hidden_size is None:
            hidden_size = embedding_size
        self.problem_names = problem_names

        self.n_classes = n_classes
        self.max_n_classes = np.max(n_classes)
        self.embedding_size = embedding_size
        self.device = device
        self.input_size = input_size

        self.embedding_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        ).to(device)

        self.classification_heads = [
            nn.Sequential(
                nn.Linear(embedding_size, self.max_n_classes),
                nn.Softmax(dim=1)
            ).to(device) for _ in n_classes
        ]

        self.last_epoch = -1

    def to(self, device):
        self.embedding_model.to(device)
        for head in self.classification_heads:
            head.to(device)
        self.device = device

    def eval_accuracy(self, test_dataloader: DataLoader):
        errors = []

        problem_names = set()
        for (Xt, yt, ids) in test_dataloader:
            errors.append((
                self.forward(Xt.to(self.device), ids).argmax(dim=1) != yt.to(self.device)
            ).to(torch.float).mean().item())
            problem_names = problem_names.union(set([self.problem_names[i] for i in ids]))
        return 1 - np.mean(errors), list(problem_names)

    def get_dataloader_batch(
        self,
        Xs: list[np.array], ys: list[np.array],
        problem_ids: list[int],
        batch_size: int = 64,
        normalize_rows=True
    ):
        y = torch.concat([y.to(torch.long) for y in ys])
        X = [torch.clone(x) for x in Xs]
        ids = torch.concat([torch.ones(x.shape[0], dtype=torch.long) * id for (x, id) in zip(Xs, problem_ids)])
        if normalize_rows:
            for i in range(len(Xs)):
                X[i] = X[i] / (torch.linalg.norm(X[i], axis=1, keepdims=True) + 1e-32)

        X = torch.concat(X)

        dataset = TensorDataset(X, y, ids)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_dataloader_single(
        self,
        X: np.array, y: np.array,
        problem_id: int,
        batch_size: int = 64,
        normalize_rows=True
    ):
        if normalize_rows:
            X = X / (torch.linalg.norm(X, axis=1, keepdims=True) + 1e-32)
        y = y.to(torch.long)
        id = torch.ones(X.shape[0], dtype=torch.long) * problem_id

        dataset = TensorDataset(X, y, id)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def forward(self, X: torch.Tensor, problem_ids: torch.Tensor) -> list[torch.Tensor]:
        embeds = self.embedding_model(X)
        outputs = torch.stack(
            [head(embeds) for head in self.classification_heads], dim=2)
        choices = torch.zeros_like(outputs)
        choices[list(range(len(problem_ids))), :, problem_ids] = 1
        return torch.sum(outputs * choices, dim=2)

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloaders: list[DataLoader],
        epochs: int = 10,
        eval_interval: int=100_000,
        lr: float = 1e-4,
        lr_decay: float = None,
    ):
        optimizer = torch.optim.Adam(
            list(self.embedding_model.parameters()) + sum(
                [list(model.parameters()) for model in self.classification_heads],
                start=[]
            ), lr=lr)
        if lr_decay is not None:
            lr_scheduler = ExponentialLR(
                optimizer, gamma=lr_decay
            )
        else:
            lr_scheduler = None

        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for dataloader in test_dataloaders:
                acc, problem_names = self.eval_accuracy(dataloader)
                print(f"Test accuracy for problem(s) {problem_names}: {acc}")

        eval_interval = int(np.ceil(eval_interval / train_dataloader.batch_size))
        n_batches = 0
        for _ in tqdm(range(epochs)):
            for (X, y, ids) in train_dataloader:
                output = self.forward(X.to(self.device), ids)
                loss = loss_fn(output, y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_batches += 1
                if n_batches % eval_interval == eval_interval - 1:
                    print("\nEvaluating...")
                    with torch.no_grad():
                        for dataloader in test_dataloaders:
                            acc, problem_names = self.eval_accuracy(dataloader)
                            print(f"Test accuracy for problem(s) {problem_names}: {acc}")
                    stdout.flush()

            self.last_epoch += 1
            if lr_scheduler is not None:
                lr_scheduler.step()

        print("\nEvaluating...")
        with torch.no_grad():
            for dataloader in test_dataloaders:
                acc, problem_names = self.eval_accuracy(dataloader)
                print(f"Test accuracy for problem(s) {problem_names}: {acc}")
           
    def get_embedding_model(
        self, tokenizer: TokenizerWrapper,
        normalize_token_counts = True,
        normalize_embeds = False
    ):
        return NNEmbedding(
            tokenizer, self.embedding_model,
            self.embedding_size,
            self.device, normalize_token_counts,
            normalize_embeds
        )