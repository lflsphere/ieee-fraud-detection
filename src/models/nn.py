"""Phase 5/6 — feed-forward neural network for tabular fraud data.

Architecture rationale (every choice below is a decision, not a default)
-----------------------------------------------------------------------
**Embeddings for categoricals, not one-hot.** `card1_code` has 13,553 levels.
One-hot would make the first layer a 13,553-wide sparse block whose weights
each see a handful of rows. A learned embedding of width 16 shares statistical
strength across levels and costs 217k parameters instead of 3.5M. Embedding
width follows `min(16, ceil(1.6 * cardinality^0.56))` — sublinear growth, so
`ProductCD` (5 levels) gets 4 dimensions and `card1` gets the cap.

**Two hidden layers, 256 → 128.** Depth is set by what the data can support,
not by what is fashionable. There are 20,663 positives in total and ~13,000 in
a typical training fold. A 256→128 net has ~250k dense parameters, already an
order of magnitude more than the number of positive examples; going deeper or
wider makes the capacity/positive-count ratio worse, and Phase 6's ablations
confirm that widening to 512 buys nothing. Tabular data with mostly axis-aligned
structure does not reward depth the way images do.

**BatchNorm then Dropout, in that order.** Inputs span standardised numerics and
freshly-initialised embeddings with very different scales; BatchNorm makes the
first epochs trainable at a usable learning rate. Dropout comes after so it
perturbs the normalised activations rather than the statistics BatchNorm is
trying to estimate.

**Class-weighted BCE (`pos_weight = n_neg/n_pos`), not resampling.** Same
reasoning as the GBM: oversampling positives inside a chronological fold
duplicates whole fraud bursts, which is precisely the structure the
chronological split exists to keep honest.

**Early stopping on validation PR-AUC, not validation loss.** With a 3.5% base
rate, weighted BCE keeps improving on the majority class after the ranking has
stopped improving. Phase 6 shows the two diverge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.models.base import CATEGORICAL_FEATURES, FraudModel

torch.set_num_threads(4)


def embedding_width(cardinality: int, cap: int = 16) -> int:
    return int(min(cap, max(2, math.ceil(1.6 * cardinality ** 0.56))))


class TabularMLP(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int],
                 hidden: tuple[int, ...] = (256, 128), dropout: float = 0.3):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, embedding_width(c)) for c in cardinalities])
        emb_dim = sum(embedding_width(c) for c in cardinalities)
        # Normalising the numeric block again inside the net (on top of the
        # StandardScaler) keeps it on the same scale as the embeddings, whose
        # initial variance is set by the initialiser rather than by the data.
        self.num_norm = nn.BatchNorm1d(n_numeric) if n_numeric else None

        layers: list[nn.Module] = []
        d = n_numeric + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.num_norm is not None and x_num.shape[1]:
            parts.append(self.num_norm(x_num))
        for i, emb in enumerate(self.embeddings):
            parts.append(emb(x_cat[:, i]))
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)


@dataclass
class NeuralNet(FraudModel):
    name: str = "nn"
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    max_epochs: int = 30
    patience: int = 5
    seed: int = config.RANDOM_SEED
    verbose: bool = False

    history_: list[dict] = field(default_factory=list)
    best_epoch_: int = -1
    _model: TabularMLP | None = None
    _imputer: SimpleImputer | None = None
    _scaler: StandardScaler | None = None
    _num_cols: list[str] = field(default_factory=list)
    _cat_cols: list[str] = field(default_factory=list)
    _cardinalities: list[int] = field(default_factory=list)

    # -- preprocessing ------------------------------------------------------
    def _prepare_fit(self, X: pd.DataFrame) -> None:
        self._cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        self._num_cols = [c for c in X.columns if c not in set(self._cat_cols)]
        # Cardinality from the *training fold only*. Codes outside this range at
        # predict time (an unseen level in a later fold) fold into index 0.
        self._cardinalities = [int(X[c].max()) + 2 for c in self._cat_cols]
        Xn = X[self._num_cols].replace([np.inf, -np.inf], np.nan)
        self._imputer = SimpleImputer(strategy="median").fit(Xn)
        self._scaler = StandardScaler().fit(self._imputer.transform(Xn))

    def _to_tensors(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        Xn = X[self._num_cols].replace([np.inf, -np.inf], np.nan)
        num = self._scaler.transform(self._imputer.transform(Xn))
        # Clipping guards the net against the extreme tails of the C*/V* blocks
        # (a single 30-sigma row otherwise dominates a batch's gradient).
        num = np.clip(num, -10, 10).astype(np.float32)

        cat = np.zeros((len(X), len(self._cat_cols)), dtype=np.int64)
        for j, c in enumerate(self._cat_cols):
            v = X[c].to_numpy()
            v = np.where(np.isfinite(v.astype(np.float64)), v, -1).astype(np.int64) + 1
            cat[:, j] = np.clip(v, 0, self._cardinalities[j] - 1)
        return torch.from_numpy(num), torch.from_numpy(cat)

    # -- training -----------------------------------------------------------
    def fit(self, X, y, X_valid=None, y_valid=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._prepare_fit(X)
        xn, xc = self._to_tensors(X)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))

        has_val = X_valid is not None and y_valid is not None
        if has_val:
            vn, vc = self._to_tensors(X_valid)
            vy = np.asarray(y_valid)

        self._model = TabularMLP(len(self._num_cols), self._cardinalities,
                                 self.hidden, self.dropout)
        pos = float((yt == 1).sum()); neg = float((yt == 0).sum())
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(neg / max(pos, 1.0), dtype=torch.float32))
        # AdamW rather than Adam: with weight_decay on plain Adam the decay is
        # folded into the adaptive denominator and is not true L2.
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        n = len(yt)
        best_score, best_state, bad = -np.inf, None, 0
        self.history_ = []

        for epoch in range(self.max_epochs):
            self._model.train()
            # Shuffling *within* the training fold is fine and necessary — the
            # chronological guarantee is about which rows are in the fold, not
            # about the order gradients see them in.
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:      # BatchNorm needs >1 row
                    continue
                opt.zero_grad()
                out = self._model(xn[idx], xc[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / n

            rec = {"epoch": epoch, "train_loss": train_loss}
            if has_val:
                self._model.eval()
                with torch.no_grad():
                    vout = self._model(vn, vc)
                    rec["valid_loss"] = float(loss_fn(vout, torch.from_numpy(
                        vy.astype(np.float32))))
                    vp = torch.sigmoid(vout).numpy()
                rec["valid_pr_auc"] = float(average_precision_score(vy, vp))
                score = rec["valid_pr_auc"]
                if score > best_score + 1e-5:
                    best_score, bad = score, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self._model.state_dict().items()}
                    self.best_epoch_ = epoch
                else:
                    bad += 1
            self.history_.append(rec)
            if self.verbose:
                print(rec)
            if has_val and bad >= self.patience:
                break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self._model.eval()
        xn, xc = self._to_tensors(X)
        outs = []
        with torch.no_grad():
            for i in range(0, len(xn), 8192):
                outs.append(torch.sigmoid(
                    self._model(xn[i:i + 8192], xc[i:i + 8192])).numpy())
        return np.concatenate(outs)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)
