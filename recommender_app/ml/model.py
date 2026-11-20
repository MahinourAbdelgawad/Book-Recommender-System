from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden: tuple[int, ...] = (128, 64)
    emb_dim: int = 64
    dropout: float = 0.1


class HybridModel(nn.Module):
    def __init__(self, config, n_features, n_users, n_books, mean):
        super().__init__()
        self.unknown_user = n_users
        self.register_buffer("mu", torch.tensor(mean, dtype=torch.float32))
        self.user_emb = nn.Embedding(n_users + 1, config.emb_dim, padding_idx=n_users)
        self.book_emb = nn.Embedding(n_books, config.emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.book_emb.weight, std=0.05)
        with torch.no_grad():
            self.user_emb.weight[n_users].zero_()

        width = n_features + 3 * config.emb_dim
        layers = []
        for hidden in config.hidden:
            layers.extend((nn.Linear(width, hidden), nn.BatchNorm1d(hidden),
                           nn.ReLU(), nn.Dropout(config.dropout)))
            width = hidden
        output = nn.Linear(width, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.mlp = nn.Sequential(*layers, output)

    def forward(self, features, users, books):
        reader, book = self.user_emb(users), self.book_emb(books)
        inputs = torch.cat((features, reader, book, reader * book), dim=1)
        return self.mu + self.mlp(inputs).squeeze(-1)

    def embedding_penalty(self, users, books):
        return (self.user_emb(users).square().sum(1) + self.book_emb(books).square().sum(1)).mean()
