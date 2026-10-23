"""Unified TransferAdapter — wraps all domain adaptation methods.

Supported methods
-----------------
no_adapt      Train RF on source features (zero_fill), predict on target.
              Uses sklearn RF — no neural network.

coral_rf      Closed-form CORAL feature alignment + RF.
              Whitens source, colors it to match target covariance, then trains RF.

coral_deep    MLP encoder trained with CORAL alignment loss.
              Unsupervised DA: needs only unlabeled target.

mmd           MLP encoder trained with MMD alignment loss.
              Unsupervised DA: needs only unlabeled target.

dann          Domain-Adversarial NN with gradient reversal layer.
              Unsupervised DA: domain labels = 0 (src), 1 (tgt).

finetune      MLP pretrained on source, then fine-tuned on labeled target.
              Requires label_frac > 0.

contrastive   Supervised contrastive pre-training (cross-domain),
              followed by classifier head fine-tuning.
              Benefits from labeled target; works with source-only at 0%.

Usage
-----
    adapter = TransferAdapter(method="coral_deep", input_dim=32, device="cuda")
    adapter.fit(
        X_src_tr, y_src_tr,           # labeled source train
        X_tgt_unlab,                  # unlabeled target (all target train)
        X_tgt_lab=X_tgt_few,          # labeled target (label_frac subset)
        y_tgt_lab=y_tgt_few,
        X_val=X_tgt_val, y_val=y_tgt_val,  # target val for early stopping
    )
    y_prob = adapter.predict_proba(X_tgt_test)
    adapter.save(path)
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier

from src.models.transfer.encoder    import FraudEncoder
from src.models.transfer.coral      import CORALAdapter, coral_loss
from src.models.transfer.mmd        import mmd_loss
from src.models.transfer.dann       import DANN
from src.models.transfer.contrastive import ContrastiveEncoder, supervised_contrastive_loss
from src.models.transfer.tradaboost  import TrAdaBoost


METHODS = ("no_adapt", "coral_rf", "coral_deep", "mmd", "dann", "finetune", "contrastive", "tradaboost")


class TransferAdapter:
    """Unified interface for all transfer learning methods."""

    def __init__(
        self,
        method: str,
        input_dim: int,
        # Deep model architecture
        hidden_dims: tuple = (128, 64),
        repr_dim: int = 32,
        dropout: float = 0.3,
        # Training hyperparams
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 10,
        # Domain alignment weight (lambda)
        lambda_da: float = 1.0,
        # RF hyperparams (no_adapt / coral_rf)
        rf_n_estimators: int = 200,
        rf_min_samples_leaf: int = 4,
        device: str = None,
        seed: int = 42,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method        = method
        self.input_dim     = input_dim
        self.hidden_dims   = hidden_dims
        self.repr_dim      = repr_dim
        self.dropout       = dropout
        self.lr            = lr
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.patience      = patience
        self.lambda_da     = lambda_da
        self.rf_n_estimators       = rf_n_estimators
        self.rf_min_samples_leaf   = rf_min_samples_leaf
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed          = seed

        # Will be set by fit()
        self._model = None
        self._coral_adapter = None  # for coral_rf

    # ── public ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_src: np.ndarray,
        y_src: np.ndarray,
        X_tgt_unlab: np.ndarray,             # all unlabeled target train rows
        X_tgt_lab: np.ndarray = None,        # labeled target subset (may be empty)
        y_tgt_lab: np.ndarray = None,
        X_val: np.ndarray = None,            # target val for early stopping
        y_val: np.ndarray = None,
    ) -> "TransferAdapter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.method == "no_adapt":
            self._fit_no_adapt(X_src, y_src)
        elif self.method == "coral_rf":
            self._fit_coral_rf(X_src, y_src, X_tgt_unlab)
        elif self.method == "coral_deep":
            self._fit_deep(X_src, y_src, X_tgt_unlab, X_val, y_val, da_loss="coral")
        elif self.method == "mmd":
            self._fit_deep(X_src, y_src, X_tgt_unlab, X_val, y_val, da_loss="mmd")
        elif self.method == "dann":
            self._fit_dann(X_src, y_src, X_tgt_unlab, X_val, y_val)
        elif self.method == "finetune":
            self._fit_finetune(X_src, y_src, X_tgt_lab, y_tgt_lab, X_val, y_val)
        elif self.method == "contrastive":
            self._fit_contrastive(X_src, y_src, X_tgt_unlab, X_tgt_lab, y_tgt_lab, X_val, y_val)
        elif self.method == "tradaboost":
            self._fit_tradaboost(X_src, y_src, X_tgt_lab, y_tgt_lab)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) positive-class probabilities."""
        if self.method in ("no_adapt", "coral_rf"):
            X_in = self._coral_adapter.transform(X) if self.method == "coral_rf" else X
            return self._model.predict_proba(X_in)[:, 1]
        elif self.method == "tradaboost":
            return self._model.predict_proba(X)
        elif self.method == "dann":
            return self._dann_predict_proba(X)
        elif self.method == "contrastive":
            return self._model.predict_proba(
                torch.FloatTensor(X).to(self.device)
            ).cpu().numpy()
        else:
            return self._model.predict_proba(X, device=self.device)

    def save(self, path: str):
        import joblib
        payload = {
            "method": self.method,
            "input_dim": self.input_dim,
            "seed": self.seed,
        }
        if self.method in ("no_adapt", "coral_rf"):
            payload["rf"] = self._model
            if self._coral_adapter:
                payload["coral_adapter"] = self._coral_adapter
        elif self.method == "tradaboost":
            payload["tradaboost"] = self._model
        else:
            payload["state_dict"] = self._model.state_dict()
            payload["model_class"] = type(self._model).__name__
            payload["hidden_dims"] = self.hidden_dims
            payload["repr_dim"] = self.repr_dim
            payload["dropout"] = self.dropout
        joblib.dump(payload, path)

    # ── no_adapt ──────────────────────────────────────────────────────────────

    def _fit_no_adapt(self, X_src, y_src):
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            min_samples_leaf=self.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.seed,
        )
        rf.fit(X_src, y_src)
        self._model = rf

    # ── coral_rf ──────────────────────────────────────────────────────────────

    def _fit_coral_rf(self, X_src, y_src, X_tgt):
        adapter = CORALAdapter()
        adapter.fit(X_src, X_tgt)
        X_src_aligned = adapter.transform(X_src)
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            min_samples_leaf=self.rf_min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.seed,
        )
        rf.fit(X_src_aligned, y_src)
        self._coral_adapter = adapter
        self._model = rf

    # ── deep (CORAL / MMD) ────────────────────────────────────────────────────

    def _fit_deep(self, X_src, y_src, X_tgt_unlab, X_val, y_val, da_loss: str):
        model = FraudEncoder(self.input_dim, self.hidden_dims, self.repr_dim, self.dropout)
        model.to(self.device)

        n_pos = int(y_src.sum())
        n_neg = len(y_src) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)
        cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        src_loader = self._make_loader(X_src, y_src, shuffle=True)
        tgt_loader = self._make_unlabeled_loader(X_tgt_unlab, shuffle=True)

        best_val_loss = float("inf")
        patience_cnt = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            tgt_iter = iter(tgt_loader)

            for X_s, y_s in src_loader:
                X_s, y_s = X_s.to(self.device), y_s.to(self.device)
                try:
                    (X_t,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    (X_t,) = next(tgt_iter)
                X_t = X_t.to(self.device)

                optimizer.zero_grad()
                z_s = model.encode(X_s)
                z_t = model.encode(X_t)
                logit_s = model.head(z_s).squeeze(-1)

                loss_cls = cls_loss_fn(logit_s, y_s)
                if da_loss == "coral":
                    loss_da = coral_loss(z_s, z_t)
                else:
                    loss_da = mmd_loss(z_s, z_t)
                loss = loss_cls + self.lambda_da * loss_da
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validate
            if X_val is not None and y_val is not None:
                val_loss = self._eval_val_loss(model, cls_loss_fn, X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model

    # ── DANN ──────────────────────────────────────────────────────────────────

    def _fit_dann(self, X_src, y_src, X_tgt_unlab, X_val, y_val):
        model = DANN(
            input_dim=self.input_dim,
            encoder_dims=list(self.hidden_dims),
            latent_dim=self.repr_dim,
            dropout=self.dropout,
        )
        model.to(self.device)

        n_pos = int(y_src.sum())
        n_neg = len(y_src) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)
        cls_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=pw)
        dom_loss_fn  = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        src_loader = self._make_loader(X_src, y_src, shuffle=True)
        tgt_loader = self._make_unlabeled_loader(X_tgt_unlab, shuffle=True)

        best_val_loss = float("inf")
        patience_cnt = 0
        best_state = None
        total_steps = self.epochs * len(src_loader)
        step = 0

        for epoch in range(self.epochs):
            model.train()
            tgt_iter = iter(tgt_loader)

            for X_s, y_s in src_loader:
                X_s, y_s = X_s.to(self.device), y_s.to(self.device)
                try:
                    (X_t,) = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    (X_t,) = next(tgt_iter)
                X_t = X_t.to(self.device)

                # Linearly increase lambda from 0→1
                p = step / max(total_steps, 1)
                lam = 2.0 / (1 + np.exp(-10 * p)) - 1  # Ganin schedule
                step += 1

                optimizer.zero_grad()
                # Source: fraud + domain loss
                fraud_s, dom_s = model(X_s, lam)
                loss_fraud = cls_loss_fn(fraud_s, y_s)
                loss_dom_s = dom_loss_fn(dom_s, torch.zeros_like(dom_s))
                # Target: domain loss only
                _, dom_t = model(X_t, lam)
                loss_dom_t = dom_loss_fn(dom_t, torch.ones_like(dom_t))

                loss = loss_fraud + self.lambda_da * (loss_dom_s + loss_dom_t) * 0.5
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                val_loss = self._eval_dann_val_loss(model, cls_loss_fn, X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model

    def _dann_predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), 2048):
                fraud_logit, _ = self._model(X_t[i : i + 2048], lambda_=0.0)
                probs.append(torch.sigmoid(fraud_logit).cpu().numpy())
        return np.concatenate(probs)

    # ── Fine-tune ─────────────────────────────────────────────────────────────

    def _fit_finetune(self, X_src, y_src, X_tgt_lab, y_tgt_lab, X_val, y_val):
        # Phase 1: train on source
        model = FraudEncoder(self.input_dim, self.hidden_dims, self.repr_dim, self.dropout)
        model.to(self.device)
        n_pos = int(y_src.sum())
        n_neg = len(y_src) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)
        cls_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        self._train_classifier(model, opt, cls_fn, X_src, y_src, X_val, y_val, self.epochs)

        # Phase 2: fine-tune on labeled target (smaller LR)
        if X_tgt_lab is not None and len(X_tgt_lab) > 0:
            n_pos_t = int(y_tgt_lab.sum())
            n_neg_t = len(y_tgt_lab) - n_pos_t
            pw_t = torch.tensor([n_neg_t / max(n_pos_t, 1)], device=self.device)
            cls_fn_t = nn.BCEWithLogitsLoss(pos_weight=pw_t)
            opt_ft = torch.optim.Adam(model.parameters(), lr=self.lr * 0.1)
            self._train_classifier(model, opt_ft, cls_fn_t,
                                   X_tgt_lab, y_tgt_lab, X_val, y_val,
                                   epochs=20, patience=5)
        self._model = model

    # ── TrAdaBoost ────────────────────────────────────────────────────────────

    def _fit_tradaboost(self, X_src, y_src, X_tgt_lab, y_tgt_lab):
        if X_tgt_lab is None or len(X_tgt_lab) == 0:
            raise ValueError("tradaboost requires labeled target data (label_frac > 0)")
        model = TrAdaBoost(seed=self.seed)
        model.fit(X_src, y_src, X_tgt_lab, y_tgt_lab)
        self._model = model

    # ── Contrastive ───────────────────────────────────────────────────────────

    def _fit_contrastive(self, X_src, y_src, X_tgt_unlab, X_tgt_lab, y_tgt_lab, X_val, y_val):
        model = ContrastiveEncoder(
            input_dim=self.input_dim,
            encoder_dims=list(self.hidden_dims),
            latent_dim=self.repr_dim,
            dropout=self.dropout,
        )
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Phase 1: contrastive pre-training on source (+ labeled target if available)
        all_X = X_src
        all_y = y_src
        if X_tgt_lab is not None and len(X_tgt_lab) > 0:
            all_X = np.vstack([X_src, X_tgt_lab])
            all_y = np.concatenate([y_src, y_tgt_lab])

        loader = self._make_loader(all_X, all_y, shuffle=True)
        for epoch in range(min(self.epochs, 30)):
            model.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                z = model.project(X_b)
                loss = supervised_contrastive_loss(z, y_b.long())
                if not loss.isnan():
                    loss.backward()
                    opt.step()

        # Phase 2: freeze encoder, train classifier head on source
        for p in model.encoder.parameters():
            p.requires_grad = False
        n_pos = int(all_y.sum())
        n_neg = len(all_y) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)
        cls_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt_head = torch.optim.Adam(model.classifier.parameters(), lr=self.lr)

        head_data_X = all_X if (X_tgt_lab is not None and len(X_tgt_lab) > 0) else X_src
        head_data_y = all_y if (X_tgt_lab is not None and len(X_tgt_lab) > 0) else y_src
        self._train_contrastive_head(model, opt_head, cls_fn,
                                     head_data_X, head_data_y, X_val, y_val)

        # Unfreeze
        for p in model.encoder.parameters():
            p.requires_grad = True
        self._model = model

    def _train_contrastive_head(self, model, opt, cls_fn, X, y, X_val, y_val, epochs=20):
        loader = self._make_loader(X, y, shuffle=True)
        best_loss = float("inf")
        patience_cnt = 0
        best_state = None
        for _ in range(epochs):
            model.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                logit = model.classifier(model.encoder(X_b)).squeeze(-1)
                loss = cls_fn(logit, y_b)
                loss.backward()
                opt.step()
            if X_val is not None and y_val is not None:
                val_loss = self._eval_contrastive_val(model, cls_fn, X_val, y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_cnt = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_cnt += 1
                    if patience_cnt >= 5:
                        break
        if best_state:
            model.load_state_dict(best_state)

    def _eval_contrastive_val(self, model, cls_fn, X_val, y_val) -> float:
        model.eval()
        X_t = torch.FloatTensor(X_val).to(self.device)
        y_t = torch.FloatTensor(y_val).to(self.device)
        with torch.no_grad():
            logit = model.classifier(model.encoder(X_t)).squeeze(-1)
            return cls_fn(logit, y_t).item()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _train_classifier(self, model, opt, cls_fn, X, y, X_val, y_val, epochs, patience=None):
        patience = patience or self.patience
        loader = self._make_loader(X, y, shuffle=True)
        best_loss = float("inf")
        patience_cnt = 0
        best_state = None
        for _ in range(epochs):
            model.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                loss = cls_fn(model(X_b), y_b)
                loss.backward()
                opt.step()
            if X_val is not None and y_val is not None:
                val_loss = self._eval_val_loss(model, cls_fn, X_val, y_val)
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_cnt = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break
        if best_state:
            model.load_state_dict(best_state)

    def _eval_val_loss(self, model, cls_fn, X_val, y_val) -> float:
        model.eval()
        X_t = torch.FloatTensor(X_val).to(self.device)
        y_t = torch.FloatTensor(y_val).to(self.device)
        with torch.no_grad():
            return cls_fn(model(X_t), y_t).item()

    def _eval_dann_val_loss(self, model, cls_fn, X_val, y_val) -> float:
        model.eval()
        X_t = torch.FloatTensor(X_val).to(self.device)
        y_t = torch.FloatTensor(y_val).to(self.device)
        with torch.no_grad():
            logit, _ = model(X_t, lambda_=0.0)
            return cls_fn(logit, y_t).item()

    def _make_loader(self, X, y, shuffle=True) -> DataLoader:
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def _make_unlabeled_loader(self, X, shuffle=True) -> DataLoader:
        ds = TensorDataset(torch.FloatTensor(X))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)
