from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from losses import joint_ce_supcon, supervised_contrastive_loss
from metrics import classification_metrics
from models import CNN1DClassifier, TCNClassifier


def build_model(config: dict, channels: int, classes: int) -> nn.Module:
    kwargs = dict(in_channels=channels, hidden_channels=int(config.get("hidden_channels", 16)),
                  projection_dim=int(config.get("projection_dim", 16)), num_classes=classes)
    if config.get("name", "tcn") == "cnn1d": return CNN1DClassifier(**kwargs)
    return TCNClassifier(**kwargs, levels=int(config.get("levels", 3)))


@dataclass
class FitResult:
    best_epoch: int
    validation_metrics: dict[str, object]
    history: list[dict[str, float]]


class ExperimentTrainer:
    def __init__(self, model: nn.Module, device: str, learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

    def _loader(self, x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
        return DataLoader(TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()), batch_size=batch, shuffle=shuffle)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
            epochs: int, batch_size: int, mode: str = "ce", supcon_weight: float = 0.1,
            temperature: float = 0.1) -> FitResult:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state, best_score, best_epoch = None, -1.0, -1
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            self.model.train(); losses = []
            for x, y in self._loader(train_x, train_y, batch_size, True):
                x, y = x.to(self.device), y.to(self.device)
                if mode == "ce_aug":
                    x = x + 0.01 * torch.randn_like(x)
                optimizer.zero_grad(); output = self.model(x)
                if mode == "supcon": loss = supervised_contrastive_loss(output["projection"], y, temperature)
                elif mode == "joint": loss = joint_ce_supcon(output["logits"], output["projection"], y, supcon_weight, temperature)
                else: loss = F.cross_entropy(output["logits"], y)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            metrics = self.evaluate(val_x, val_y, batch_size)
            score = float(metrics["macro_f1"])
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses)), "validation_macro_f1": score})
            if score > best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(self.model.state_dict())
        if best_state is not None: self.model.load_state_dict(best_state)
        return FitResult(best_epoch, self.evaluate(val_x, val_y, batch_size), history)

    def pretrain_supcon(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray,
                        val_y: np.ndarray, epochs: int, batch_size: int,
                        temperature: float) -> list[dict[str, float]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state, best_loss = None, float("inf")
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            self.model.train(); training = []
            for x, y in self._loader(train_x, train_y, batch_size, True):
                x, y = x.to(self.device), y.to(self.device); optimizer.zero_grad()
                loss = supervised_contrastive_loss(self.model(x)["projection"], y, temperature)
                loss.backward(); optimizer.step(); training.append(float(loss.detach()))
            validation = self.contrastive_loss(val_x, val_y, batch_size, temperature)
            history.append({"epoch": float(epoch), "loss": float(np.mean(training)), "validation_supcon_loss": validation})
            if validation < best_loss:
                best_loss, best_state = validation, copy.deepcopy(self.model.state_dict())
        if best_state is not None: self.model.load_state_dict(best_state)
        return history

    def contrastive_loss(self, x: np.ndarray, y: np.ndarray, batch_size: int, temperature: float) -> float:
        self.model.eval(); losses = []
        with torch.no_grad():
            for xb, yb in self._loader(x, y, batch_size, False):
                losses.append(float(supervised_contrastive_loss(self.model(xb.to(self.device))["projection"], yb.to(self.device), temperature)))
        return float(np.mean(losses))

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict[str, object]:
        self.model.eval(); predictions, probabilities = [], []
        with torch.no_grad():
            for xb, _ in self._loader(x, y, batch_size, False):
                logits = self.model(xb.to(self.device))["logits"]
                probabilities.append(torch.softmax(logits, 1).cpu().numpy())
                predictions.append(logits.argmax(1).cpu().numpy())
        return classification_metrics(y, np.concatenate(predictions), np.concatenate(probabilities))

    def freeze_encoder(self) -> None:
        for parameter in self.model.encoder.parameters(): parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for parameter in self.model.parameters(): parameter.requires_grad = True
