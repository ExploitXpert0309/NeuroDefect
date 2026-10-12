from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    backend: str = "numpy"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


if HAS_TORCH:

    class _TorchBPN(nn.Module):
        def __init__(self, n_in: int, hidden: tuple[int, int], dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[1], 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)
else:
    _TorchBPN = None  # type: ignore[misc, assignment]


class BPNClassifier(BaseEstimator, ClassifierMixin):
    """Feed-forward defect predictor: Dense(64)-ReLU-Dropout-Dense(32)-ReLU-Dropout-Dense(1)."""

    def __init__(
        self,
        hidden_sizes: tuple[int, int] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        optimizer: str = "sgd_momentum",
        momentum: float = 0.9,
        epochs: int = 40,
        batch_size: int = 32,
        patience: int = 8,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.history_ = TrainHistory()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, X_val=None, y_val=None, callback=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_val is None or y_val is None:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(X))
            cut = max(int(0.15 * len(X)), 8)
            val_idx, train_idx = idx[:cut], idx[cut:]
            X_val, y_val = X[val_idx], y[val_idx]
            X, y = X[train_idx], y[train_idx]
        else:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

        if HAS_TORCH:
            self._fit_torch(X, y, X_val, y_val, callback)
        else:
            self._fit_numpy(X, y, X_val, y_val, callback)
        return self

    def _fit_torch(self, X, y, X_val, y_val, callback):
        torch.manual_seed(self.random_state)
        device = torch.device("cpu")
        model = _TorchBPN(X.shape[1], self.hidden_sizes, self.dropout).to(device)
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if self.optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            opt = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum if "momentum" in self.optimizer else 0.0,
            )
        best_state = None
        best_val = float("inf")
        stall = 0
        history = TrainHistory(backend="torch")
        n = len(X)
        order = np.arange(n)
        rng = np.random.default_rng(self.random_state)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)

        for epoch in range(self.epochs):
            model.train()
            rng.shuffle(order)
            losses = []
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                xb = X_t[batch]
                yb = y_t[batch]
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                tr_logits = model(X_t)
                va_logits = model(Xv)
                tr_loss = float(loss_fn(tr_logits, y_t).item())
                va_loss = float(loss_fn(va_logits, yv).item())
                tr_prob = torch.sigmoid(tr_logits).numpy()
                va_prob = torch.sigmoid(va_logits).numpy()
            tr_pred = (tr_prob >= 0.5).astype(int)
            va_pred = (va_prob >= 0.5).astype(int)
            tr_f1 = float(f1_score(y, tr_pred, zero_division=0))
            va_f1 = float(f1_score(y_val, va_pred, zero_division=0))
            tr_acc = float((tr_pred == y).mean())
            history.train_loss.append(tr_loss)
            history.val_loss.append(va_loss)
            history.train_f1.append(tr_f1)
            history.val_f1.append(va_f1)
            history.train_acc.append(tr_acc)
            if callback:
                callback(epoch + 1, self.epochs, history)
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                stall = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.backend_ = "torch"
        self.n_features_in_ = X.shape[1]
        self.history_ = history

    def _fit_numpy(self, X, y, X_val, y_val, callback):
        rng = np.random.default_rng(self.random_state)
        n_in = X.shape[1]
        h1, h2 = self.hidden_sizes
        W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, h1)).astype(np.float32)
        b1 = np.zeros(h1, dtype=np.float32)
        W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2)).astype(np.float32)
        b2 = np.zeros(h2, dtype=np.float32)
        W3 = rng.normal(0, np.sqrt(2 / h2), (h2, 1)).astype(np.float32)
        b3 = np.zeros(1, dtype=np.float32)
        params = [W1, b1, W2, b2, W3, b3]
        use_adam = self.optimizer == "adam"
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        vel = [np.zeros_like(p) for p in params]
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        pos_weight = neg / pos
        best = None
        best_val = float("inf")
        stall = 0
        history = TrainHistory(backend="numpy")
        t_step = 0
        n = len(X)

        def forward(xb, train=True):
            z1 = xb @ W1 + b1
            a1 = _relu(z1)
            if train and self.dropout > 0:
                mask1 = (rng.random(a1.shape) > self.dropout).astype(np.float32)
                a1 = a1 * mask1 / (1.0 - self.dropout)
            else:
                mask1 = None
            z2 = a1 @ W2 + b2
            a2 = _relu(z2)
            if train and self.dropout > 0:
                mask2 = (rng.random(a2.shape) > self.dropout).astype(np.float32)
                a2 = a2 * mask2 / (1.0 - self.dropout)
            else:
                mask2 = None
            z3 = a2 @ W3 + b3
            p = _sigmoid(z3.reshape(-1))
            cache = (xb, z1, a1, mask1, z2, a2, mask2, z3, p)
            return p, cache

        def bce(p, yb):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.mean(-(pos_weight * yb * np.log(p) + (1 - yb) * np.log(1 - p))))

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                xb, yb = X[batch], y[batch]
                p, cache = forward(xb, train=True)
                p = np.clip(p, 1e-6, 1 - 1e-6)
                xb, z1, a1, mask1, z2, a2, mask2, z3, _ = cache
                dz3 = ((pos_weight * (p - yb)) / len(yb)).reshape(-1, 1)
                gW3 = a2.T @ dz3
                gb3 = dz3.sum(axis=0)
                da2 = dz3 @ W3.T
                if mask2 is not None:
                    da2 = da2 * mask2 / (1.0 - self.dropout)
                dz2 = da2 * (z2 > 0)
                gW2 = a1.T @ dz2
                gb2 = dz2.sum(axis=0)
                da1 = dz2 @ W2.T
                if mask1 is not None:
                    da1 = da1 * mask1 / (1.0 - self.dropout)
                dz1 = da1 * (z1 > 0)
                gW1 = xb.T @ dz1
                gb1 = dz1.sum(axis=0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3]
                t_step += 1
                if use_adam:
                    beta1, beta2, eps = 0.9, 0.999, 1e-8
                    for i, g in enumerate(grads):
                        m[i] = beta1 * m[i] + (1 - beta1) * g
                        v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                        mhat = m[i] / (1 - beta1**t_step)
                        vhat = v[i] / (1 - beta2**t_step)
                        params[i] -= self.learning_rate * mhat / (np.sqrt(vhat) + eps)
                else:
                    mu = self.momentum if "momentum" in self.optimizer else 0.0
                    for i, g in enumerate(grads):
                        vel[i] = mu * vel[i] - self.learning_rate * g
                        params[i] += vel[i]
                W1, b1, W2, b2, W3, b3 = params

            tr_p, _ = forward(X, train=False)
            va_p, _ = forward(X_val, train=False)
            tr_loss, va_loss = bce(tr_p, y), bce(va_p, y_val)
            tr_pred = (tr_p >= 0.5).astype(int)
            va_pred = (va_p >= 0.5).astype(int)
            tr_f1 = float(f1_score(y, tr_pred, zero_division=0))
            va_f1 = float(f1_score(y_val, va_pred, zero_division=0))
            history.train_loss.append(tr_loss)
            history.val_loss.append(va_loss)
            history.train_f1.append(tr_f1)
            history.val_f1.append(va_f1)
            history.train_acc.append(float((tr_pred == y).mean()))
            if callback:
                callback(epoch + 1, self.epochs, history)
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                stall = 0
                best = [p.copy() for p in params]
            else:
                stall += 1
                if stall >= self.patience:
                    break
        if best is not None:
            self.weights_ = best
        else:
            self.weights_ = params
        self.backend_ = "numpy"
        self.n_features_in_ = n_in
        self.history_ = history

    def _numpy_proba(self, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2, W3, b3 = self.weights_
        a1 = _relu(X @ W1 + b1)
        a2 = _relu(a1 @ W2 + b2)
        return _sigmoid((a2 @ W3 + b3).reshape(-1))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if getattr(self, "backend_", "numpy") == "torch" and HAS_TORCH:
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.tensor(X, dtype=torch.float32))
                p1 = torch.sigmoid(logits).numpy().reshape(-1)
        else:
            p1 = self._numpy_proba(X)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save_weights(self) -> dict:
        if getattr(self, "backend_", "numpy") == "torch" and HAS_TORCH:
            return {
                "backend": "torch",
                "state_dict": {k: v.cpu().numpy() for k, v in self.model_.state_dict().items()},
                "n_features": int(self.n_features_in_),
                "hidden": self.hidden_sizes,
                "dropout": self.dropout,
            }
        return {
            "backend": "numpy",
            "weights": [w.copy() for w in self.weights_],
            "n_features": int(self.n_features_in_),
            "hidden": self.hidden_sizes,
            "dropout": self.dropout,
        }

    @classmethod
    def from_weights(cls, payload: dict, **kwargs) -> "BPNClassifier":
        model = cls(hidden_sizes=tuple(payload.get("hidden", (64, 32))), **kwargs)
        model.n_features_in_ = payload["n_features"]
        model.backend_ = payload["backend"]
        if payload["backend"] == "torch" and HAS_TORCH:
            net = _TorchBPN(payload["n_features"], tuple(payload["hidden"]), payload.get("dropout", 0.2))
            state = {k: torch.tensor(v) for k, v in payload["state_dict"].items()}
            net.load_state_dict(state)
            net.eval()
            model.model_ = net
        else:
            model.weights_ = payload["weights"]
            model.backend_ = "numpy"
        return model
