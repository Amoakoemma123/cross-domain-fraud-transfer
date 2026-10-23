"""Feature alignment strategies for cross-domain transfer.

Three strategies
────────────────
semantic     Two canonical features guaranteed across all datasets:
             amount_log, hour_of_day.

intersection All feature names common to both preprocessed source and target
             (after canonical renaming).  Requires target to be preprocessed
             independently and the column lists to be compared.

zero_fill    Use the full source feature space.  Target columns that are
             absent from source features are dropped; source features absent
             in the target are filled with 0.

Canonical rename map
────────────────────
Some preprocessors produce dataset-specific names (e.g. ULB's Amount_log vs
PaySim's amount_log).  `canonicalize()` applies a deterministic rename so
that cross-dataset comparison is meaningful.
"""

import numpy as np

# ── canonical name map ────────────────────────────────────────────────────────

_RENAME: dict[str, str] = {
    # ULB
    "Amount":        "amount",
    "Amount_log":    "amount_log",
    "Time":          "time_elapsed",
    # IEEE-CIS
    "TransactionAmt":     "amount",
    "TransactionAmt_log": "amount_log",
    "TransactionDT":      "time_elapsed",
    # PaySim / BankSim already lowercase; no rename needed
}

# Features guaranteed to be produced (post-engineering) across all 4 datasets
SEMANTIC_FEATURES = ["amount_log", "hour_of_day"]


def canonicalize(feature_names: list[str]) -> list[str]:
    """Apply canonical renaming to a list of preprocessor output feature names."""
    return [_RENAME.get(n, n) for n in feature_names]


# ── alignment strategies ──────────────────────────────────────────────────────

class FeatureAligner:
    """Aligns source and target feature matrices for cross-domain transfer.

    Parameters
    ----------
    strategy : {'semantic', 'intersection', 'zero_fill'}
    source_names : list[str]   Preprocessor output names for source dataset.
    target_names : list[str]   Preprocessor output names for target dataset.
    """

    def __init__(
        self,
        strategy: str,
        source_names: list[str],
        target_names: list[str],
    ):
        if strategy not in ("semantic", "intersection", "zero_fill"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy

        # Apply canonical renaming
        self._src_names = canonicalize(source_names)
        self._tgt_names = canonicalize(target_names)

        self._src_index = {n: i for i, n in enumerate(self._src_names)}
        self._tgt_index = {n: i for i, n in enumerate(self._tgt_names)}

        self._aligned_names: list[str] = self._compute_aligned_names()

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        return self._aligned_names

    @property
    def n_features(self) -> int:
        return len(self._aligned_names)

    def align_source(self, X_src: np.ndarray) -> np.ndarray:
        """Project source matrix onto the aligned feature space."""
        return self._project(X_src, self._src_index)

    def align_target(self, X_tgt: np.ndarray) -> np.ndarray:
        """Project target matrix onto the aligned feature space."""
        return self._project(X_tgt, self._tgt_index)

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute_aligned_names(self) -> list[str]:
        if self.strategy == "semantic":
            present = [f for f in SEMANTIC_FEATURES
                       if f in self._src_index and f in self._tgt_index]
            if not present:
                raise ValueError(
                    "No semantic features found in both source and target. "
                    f"Expected {SEMANTIC_FEATURES}. "
                    f"Source: {self._src_names[:10]}... "
                    f"Target: {self._tgt_names[:10]}..."
                )
            return present

        if self.strategy == "intersection":
            src_set = set(self._src_names)
            tgt_set = set(self._tgt_names)
            common = src_set & tgt_set
            # Preserve source ordering for reproducibility
            return [n for n in self._src_names if n in common]

        # zero_fill: full source space
        return list(self._src_names)

    def _project(self, X: np.ndarray, index: dict[str, int]) -> np.ndarray:
        """Select / fill columns to match self._aligned_names."""
        n = X.shape[0]
        out = np.zeros((n, self.n_features), dtype=np.float32)
        for col_out, feat in enumerate(self._aligned_names):
            col_in = index.get(feat, None)
            if col_in is not None:
                out[:, col_out] = X[:, col_in]
            # else: leave as 0 (zero_fill for absent target features)
        return out


# ── convenience factory ───────────────────────────────────────────────────────

def make_aligner(
    strategy: str,
    source_names: list[str],
    target_names: list[str],
) -> FeatureAligner:
    return FeatureAligner(strategy, source_names, target_names)
