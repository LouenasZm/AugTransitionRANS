"""
Shared loss utilities for the bi-level optimisation framework.

Both the follower problem (training split) and the leader problem (validation
split) evaluate a normalised, masked MSE of the form:

    L = (1 / N_cases / N_vars) * sum_{c,v}  MSE_masked(c,v) / baseline(c,v)

The helpers here implement that computation in a fully vectorised way so that
both levels can reuse the same logic.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level building block
# ---------------------------------------------------------------------------

def masked_normalised_mse(
    preds:     np.ndarray,   # (n_points, n_candidates)
    les_val:   np.ndarray,   # (n_points,)
    s_star:    np.ndarray,   # (n_points,)  — region selector, True where active
    baseline:  float,
    *,
    s_star_threshold: float = 0.8,
) -> np.ndarray:
    """Vectorised masked, baseline-normalised MSE.

    Parameters
    ----------
    preds:
        Surrogate predictions shaped (n_points, n_candidates).
        Callers are responsible for transposing if the surrogate returns
        (n_candidates, n_points).
    les_val:
        Reference (LES) values shaped (n_points,).
    s_star:
        Strain-rate-based selector values shaped (n_points,).
        Only points where s_star < s_star_threshold are included.
    baseline:
        Baseline MSE used to normalise.  Replaced by 1.0 if zero.
    s_star_threshold:
        Threshold for the S* mask (default 0.8).

    Returns
    -------
    mse_per_candidate : np.ndarray, shape (n_candidates,)
    """
    finite_mask = np.isfinite(les_val)
    if not finite_mask.any():
        return np.zeros(preds.shape[1])

    s_star_mask = s_star[finite_mask] < s_star_threshold

    les_masked  = les_val[finite_mask][s_star_mask, np.newaxis]  # (n_active, 1)
    preds_masked = preds[s_star_mask]               # (n_active, n_cand)

    mse = ((preds_masked - les_masked) ** 2).mean(axis=0)        # (n_cand,)

    denom = baseline if baseline != 0.0 else 1.0
    return mse / denom


# ---------------------------------------------------------------------------
# Dataset-level aggregator
# ---------------------------------------------------------------------------

def compute_normalised_loss(
    x:         np.ndarray,                   # (n_candidates, n_features)
    cases:     List[str],
    variables: List[str],
    data:      Dict[str, Any],
    *,
    s_star_threshold: float = 0.8,
) -> np.ndarray:
    """Aggregate masked, normalised MSE over all cases and variables.

    Parameters
    ----------
    x:
        Candidate parameter vectors shaped (n_candidates, n_features).
    cases:
        List of flow-case identifiers to iterate over.
    variables:
        List of physical quantity names to iterate over.
    data:
        A dict that must contain:

        * "surrogates" — {case: {var: model}}
        * "les"        — {case: {var: array, "S_star": array}}
        * "baseline"   — {"loss": {case: {var: float}}}  *or*
                              {case: {var: float}} (leader form)
        * "scalers"    — {case: scaler}
    s_star_threshold:
        Threshold for the S* region mask.

    Returns
    -------
    loss_value : np.ndarray, shape (n_candidates,)
    """
    n_candidates = x.shape[0]
    loss_value   = np.zeros(n_candidates)

    # Support both "baseline.loss.case.var" (follower) and "baseline.case.var" (leader)
    def _get_baseline(case: str, var: str) -> float:
        b = data["baseline"]
        if "loss" in b:
            return b["loss"][case][var]
        return b[case][var]

    for case in cases:
        scaler      = data["scalers"][case]
        x_scaled    = scaler.transform(x)
        les_case    = data["les"][case]
        surr_case   = data["surrogates"][case]
        s_star_full = les_case["S_star"].flatten()

        for var in variables:
            preds = surr_case[var].predict(x_scaled)  # shape may vary by model

            # Normalise to (n_points, n_candidates)
            if preds.ndim == 1:
                # Scalar-per-candidate output: shape (n_candidates,) → (1, n_candidates)
                preds = preds[np.newaxis, :]
            elif preds.shape == (n_candidates, 1):
                # Column-vector output from single-output regressors
                preds = preds.T                        # (1, n_candidates)
            elif preds.ndim == 2 and preds.shape[0] == n_candidates and preds.shape[1] != n_candidates:
                preds = preds.T  # (n_candidates, n_points) → (n_points, n_candidates)

            les_val  = les_case[var].flatten()
            baseline = _get_baseline(case, var)

            if not np.isfinite(les_val).any():
                logger.warning("No valid LES data for case=%s var=%s — skipping.", case, var)
                continue

            loss_value += masked_normalised_mse(
                preds, les_val, s_star_full, baseline,
                s_star_threshold=s_star_threshold,
            )

    n_terms = len(cases) * len(variables)
    return loss_value / n_terms if n_terms > 0 else loss_value
