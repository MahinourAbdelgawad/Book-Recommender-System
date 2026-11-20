from copy import deepcopy
from dataclasses import dataclass

import torch

from .evaluation import predict, rmse
from .model import HybridModel


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    patience: int = 5
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_l2: float = 1e-5


def fit(config, settings, data, train, validation, seed, on_epoch=print):
    torch.manual_seed(seed)
    model = HybridModel(config, len(data.columns), len(data.user_ids), len(data.book_ids), data.mean)
    embeddings = [model.user_emb.weight, model.book_emb.weight]
    optimizer = torch.optim.AdamW(
        [{"params": model.mlp.parameters(), "weight_decay": settings.weight_decay},
         {"params": embeddings, "weight_decay": 0.0}], lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    tensors = tuple(torch.from_numpy(a) for a in (train.features, train.users, train.books, train.ratings))
    best_score = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        for batch in torch.randperm(len(train.ratings)).split(settings.batch_size):
            if len(batch) < 2:
                continue  # Batch normalization needs at least two rows.
            features, users, books, ratings = (tensor[batch] for tensor in tensors)
            loss = (model(features, users, books) - ratings).square().mean()
            loss = loss + settings.embedding_l2 * model.embedding_penalty(users, books)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_score = rmse(predict(model, train.features, train.users, train.books), train.ratings)
        val_score = validation.warm_rmse(model)
        scheduler.step(val_score)
        if not (0 <= train_score <= 4 and 0 <= val_score <= 4):
            raise RuntimeError("Non-finite or invalid rating error")
        row = {"epoch": epoch, "train_rmse": train_score, "validation_rmse": val_score,
               "gap": val_score - train_score}
        history.append(row)
        on_epoch(row)
        if val_score < best_score - 1e-5:
            best_score, best_epoch = val_score, epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= settings.patience:
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch
