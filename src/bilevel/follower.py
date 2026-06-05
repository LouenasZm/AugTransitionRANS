"""

    Author: Louenas Zemmour, Novemebr 2025, refactored on May 2026

Follower (lower-level) problem for the bi-level optimisation framework.

The follower minimises an elastic-net regularised, surrogate-based MSE loss
over the correction coefficients **Γ**:

    L̃(Gamma, alpha, beta) = L_MSE(Gamma) + alpha ‖Gamma‖₁  +  beta · ½‖Gamma‖₂²

where the MSE is computed against LES reference data filtered by a
strain-rate-based mask (S* < threshold).

Usage example
-------------
::

    from bilevel_optim.follower import ElasticNetFollower, OptimizationCallback
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize

    problem  = ElasticNetFollower(
        hyperparams=(alpha, beta),
        data=follower_data,
        bounds=(l_bounds, u_bounds),
        training_cases=config["optim"]["training_cases"],
        variables=config["surrogate"]["qty"],
    )
    callback = OptimizationCallback()
    res      = minimize(problem, GA(pop_size=300), ("n_gen", 250),
                        callback=callback, verbose=False)
    best_coefficients = res.X
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.problem  import Problem

from .loss import compute_normalised_loss

logger = logging.getLogger(__name__)


class ElasticNetFollower(Problem):
    """Elastic-net regularised follower problem.

    Parameters
    ----------
    hyperparams:
        Tuple ``(alpha, beta)`` — regularisation strengths for the L1 and
        (half) L2 terms, respectively.
    data:
        Dictionary with keys ``"surrogates"``, ``"les"``, ``"baseline"``,
        and ``"scalers"``.  See :mod:`bilevel_optim.types` for the expected
        layout.
    bounds:
        Tuple ``(lower, upper)`` of array-like, each of length ``n_var``.
    training_cases:
        List of flow-case identifiers used for the follower loss.
    variables:
        List of physical quantity names (e.g. ``["uu", "vv", "uv"]``).
    n_var:
        Number of correction coefficients (design variables).
    s_star_threshold:
        Threshold on S* below which points are included in the MSE.
    """

    def __init__(
        self,
        hyperparams:      Tuple[float, float],
        data:             Dict[str, Any],
        bounds:           Tuple[Sequence[float], Sequence[float]],
        training_cases:   List[str],
        variables:        List[str],
        n_var:            int  = 17,
        s_star_threshold: float = 0.8,
    ) -> None:
        lower, upper = bounds
        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=np.asarray(lower, dtype=float),
            xu=np.asarray(upper, dtype=float),
        )
        self.alpha            = float(hyperparams[0])
        self.beta             = float(hyperparams[1])
        self.data             = data
        self.training_cases   = training_cases
        self.variables        = variables
        self.s_star_threshold = s_star_threshold

    # ------------------------------------------------------------------
    # pymoo interface
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate elastic-net objective for a population ``x`` of shape
        ``(n_candidates, n_var)``."""
        mse_term = compute_normalised_loss(
            x,
            cases=self.training_cases,
            variables=self.variables,
            data=self.data,
            s_star_threshold=self.s_star_threshold,
        )
        l1_term = np.linalg.norm(x, axis=1, ord=1)
        l2_term = 0.5 * np.linalg.norm(x, axis=1, ord=2)

        out["F"] = (mse_term + self.alpha * l1_term + self.beta * l2_term)[:, np.newaxis]


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class OptimizationCallback(Callback):
    """Records per-generation statistics (best / worst / mean objective).

    After the run, ``callback.history`` is a list of dicts with keys:

    * ``generation``
    * ``best_candidate``, ``best_objective``
    * ``worst_candidate``, ``worst_objective``
    * ``mean_objective``
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: List[dict] = []

    def notify(self, algorithm) -> None:
        pop        = algorithm.pop
        objectives = pop.get("F")          # (pop_size, n_obj)
        f_1d       = objectives[:, 0]

        best_idx  = f_1d.argmin()
        worst_idx = f_1d.argmax()

        self.history.append({
            "generation":     algorithm.n_gen,
            "best_candidate": pop[best_idx].X.tolist(),
            "best_objective": float(f_1d[best_idx]),
            "worst_candidate": pop[worst_idx].X.tolist(),
            "worst_objective": float(f_1d[worst_idx]),
            "mean_objective":  float(f_1d.mean()),
        })

        logger.debug(
            "Gen %d | best=%.6f  worst=%.6f  mean=%.6f",
            algorithm.n_gen,
            f_1d[best_idx], f_1d[worst_idx], f_1d.mean(),
        )
