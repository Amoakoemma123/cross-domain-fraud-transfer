"""Generic PyTorch trainer for fraud detection models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Callable


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: str = None,
        patience: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience
        self.model.to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 512,
    ) -> list[dict]:
        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size, shuffle=False)

        history = []
        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._eval_epoch(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return history

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                batch = X_t[i : i + batch_size]
                out = self.model(batch)
                probs.append(torch.sigmoid(out).cpu().numpy())
        return np.concatenate(probs)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_batch)
            loss = self.loss_fn(out, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                out = self.model(X_batch)
                loss = self.loss_fn(out, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
    ) -> DataLoader:
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)
