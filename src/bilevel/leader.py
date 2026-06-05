"""

    Author: Louenas Zemmour, Novemebr 2025, refactored on May 2026

Leader (upper-level) problem for the bi-level optimisation framework.

The leader minimises the validation MSE by choosing the regularisation
hyperparameters (alpha, beta) that yield the best follower response:

    min_{alpha,beta}  L_MSE(Gamma*(alpha,beta), Q_val)
    s.t.  Gamma*(alpha,beta) = argmin_Gamma  L̃(Gamma, alpha, beta)_train

Usage example
-------------
::

    from bilevel_optim.leader import LeaderGA, run_follower_worker
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize

    leader   = LeaderGA.from_config(config)
    res      = minimize(leader, GA(pop_size=40), ("n_gen", 25), verbose=True)
    print("Best hyperparams:", res.X)
    print("Best coefficients:", leader.best_coefficients)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed, dump, load
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from .follower import ElasticNetFollower, OptimizationCallback
from .loss import compute_normalised_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone worker (must be module-level for joblib / multiprocessing)
# ---------------------------------------------------------------------------

def run_follower_worker(
    hyperparams:    Tuple[float, float],
    follower_data:  Dict[str, Any],
    training_cases: List[str],
    variables:      List[str],
    n_var:          int,
    bounds:         Tuple[List[float], List[float]],
    *,
    pop_size:       int   = 300,
    n_gen:          int   = 250,
    s_star_threshold: float = 0.8,
) -> np.ndarray:
    """Solve one follower problem and return the optimal coefficients.

    Designed to be called from joblib.Parallel; all arguments are explicit
    to avoid pickling self.
    """
    problem = ElasticNetFollower(
        hyperparams=hyperparams,
        data=follower_data,
        bounds=bounds,
        training_cases=training_cases,
        variables=variables,
        n_var=n_var,
        s_star_threshold=s_star_threshold,
    )
    algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", n_gen), verbose=False)
    return res.X


# ---------------------------------------------------------------------------
# Leader problem
# ---------------------------------------------------------------------------

class LeaderGA(Problem):
    """Leader problem that drives bi-level hyperparameter search.

    Parameters
    ----------
    leader_data:
        Validation-split data dict with keys "surrogates", "les",
        "baseline", "scalers".
    follower_data:
        Training-split data dict with the same structure plus an additional
        "baseline" sub-key "loss".
    validation_cases:
        Flow cases used to evaluate the leader objective.
    training_cases:
        Flow cases passed to the follower.
    variables:
        Physical quantities (surrogate outputs).
    n_coefficients:
        Number of correction coefficients (n_var of the follower).
    bounds:
        Coefficient bounds (lower, upper) passed to the follower.
    hyperparam_bounds:
        (lower, upper) for (alpha, beta).  Defaults to ([0,0],[1,1]).
    follower_pop_size:
        GA population size for each follower solve.
    follower_n_gen:
        Number of generations for each follower solve.
    n_jobs:
        Number of parallel jobs for follower solves.  Use 1 to disable
        parallelism (safer on memory-constrained machines).
    s_star_threshold:
        S* mask threshold shared by both levels.
    use_memmap:
        If True, dump follower_data to a memory-mapped file so worker
        processes share it without duplicating RAM.
    """

    def __init__(
        self,
        leader_data:       Dict[str, Any],
        follower_data:     Dict[str, Any],
        validation_cases:  List[str],
        training_cases:    List[str],
        variables:         List[str],
        n_coefficients:    int,
        bounds:            Tuple[List[float], List[float]],
        *,
        hyperparam_bounds: Tuple[List[float], List[float]] = ([0.0, 0.0], [1.0, 1.0]),
        follower_pop_size: int   = 300,
        follower_n_gen:    int   = 250,
        n_jobs:            int   = 1,
        s_star_threshold:  float = 0.8,
        use_memmap:        bool  = True,
    ) -> None:
        hl, hu = hyperparam_bounds
        super().__init__(
            n_var=len(hl),
            n_obj=1,
            xl=np.asarray(hl, dtype=float),
            xu=np.asarray(hu, dtype=float),
        )
        self.leader_data       = leader_data
        self.validation_cases  = validation_cases
        self.training_cases    = training_cases
        self.variables         = variables
        self.n_coefficients    = n_coefficients
        self.coeff_bounds      = bounds
        self.follower_pop_size = follower_pop_size
        self.follower_n_gen    = follower_n_gen
        self.n_jobs            = n_jobs
        self.s_star_threshold  = s_star_threshold

        # Best-so-far tracking
        self.best_objective:    float     = np.inf
        self.best_coefficients: Optional[np.ndarray] = None

        # Optionally memory-map follower data for parallel workers
        self._temp_folder: Optional[str] = None
        if use_memmap and n_jobs != 1:
            self._temp_folder   = tempfile.mkdtemp()
            _mmap_path          = os.path.join(self._temp_folder, "follower_data.mmap")
            dump(follower_data, _mmap_path)
            self.follower_data  = load(_mmap_path, mmap_mode="r")
            logger.info("Follower data memory-mapped at %s", _mmap_path)
        else:
            self.follower_data = follower_data

    # ------------------------------------------------------------------
    # pymoo interface
    # ------------------------------------------------------------------

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """Evaluate the leader objective for a population of hyperparameter
        vectors x of shape (n_candidates, 2)."""
        out["F"] = self._compute_leader_loss(x)[:, np.newaxis]

    # ------------------------------------------------------------------
    # Leader loss
    # ------------------------------------------------------------------

    def _compute_leader_loss(self, x: np.ndarray) -> np.ndarray:
        """Solve all follower problems, then evaluate validation loss.

        Parameters
        ----------
        x : np.ndarray, shape (n_candidates, 2)
            Each row is [alpha, beta].

        Returns
        -------
        val_loss : np.ndarray, shape (n_candidates,)
        """
        n_candidates = x.shape[0]

        # ---------- Solve follower for each candidate ----------
        if self.n_jobs == 1:
            follower_responses = np.vstack([
                self._solve_single_follower(x[i])
                for i in range(n_candidates)
            ])
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(run_follower_worker)(
                    tuple(x[i]),
                    self.follower_data,
                    self.training_cases,
                    self.variables,
                    self.n_coefficients,
                    self.coeff_bounds,
                    pop_size=self.follower_pop_size,
                    n_gen=self.follower_n_gen,
                    s_star_threshold=self.s_star_threshold,
                )
                for i in range(n_candidates)
            )
            follower_responses = np.vstack(results)

        # ---------- Evaluate validation loss ----------
        # Build a "leader-compatible" data dict using follower responses as
        # scaled inputs for the validation surrogates.
        val_loss = self._evaluate_validation_loss(follower_responses)

        # ---------- Track best ----------
        best_idx = int(np.argmin(val_loss))
        if val_loss[best_idx] < self.best_objective:
            self.best_objective    = float(val_loss[best_idx])
            self.best_coefficients = follower_responses[best_idx].copy()
            logger.info(
                "New best: obj=%.6f  hyperparams=%s",
                self.best_objective, x[best_idx].tolist(),
            )

        return val_loss

    def _evaluate_validation_loss(self, follower_responses: np.ndarray) -> np.ndarray:
        """Compute normalised validation MSE given the matrix of follower responses.

        Parameters
        ----------
        follower_responses : np.ndarray, shape (n_candidates, n_coefficients)

        Returns
        -------
        val_loss : np.ndarray, shape (n_candidates,)
        """
        n_candidates = follower_responses.shape[0]
        val_loss     = np.zeros(n_candidates)

        for case in self.validation_cases:
            scaler           = self.leader_data["scalers"][case]
            responses_scaled = scaler.transform(follower_responses)
            les_case         = self.leader_data["les"][case]
            surr_case        = self.leader_data["surrogates"][case]
            s_star_full      = les_case["S_star"].flatten()

            for var in self.variables:
                preds = surr_case[var].predict(responses_scaled)

                # Normalise to (n_points, n_candidates)
                if preds.ndim == 1:
                    preds = preds[np.newaxis, :]
                elif preds.shape == (n_candidates, 1):
                    preds = preds.T
                elif preds.ndim == 2 and preds.shape[0] == n_candidates and preds.shape[1] != n_candidates:
                    preds = preds.T

                les_val  = les_case[var].flatten()
                baseline = self.leader_data["baseline"][case][var]

                if not np.isfinite(les_val).any():
                    logger.warning(
                        "No valid LES data for case=%s var=%s — skipping.", case, var
                    )
                    continue

                from .loss import masked_normalised_mse
                val_loss += masked_normalised_mse(
                    preds, les_val, s_star_full, baseline,
                    s_star_threshold=self.s_star_threshold,
                )

        n_terms = len(self.validation_cases) * len(self.variables)
        return val_loss / n_terms if n_terms > 0 else val_loss

    def _solve_single_follower(self, hyperparams: np.ndarray) -> np.ndarray:
        """Solve the follower for a single hyperparameter vector."""
        problem = ElasticNetFollower(
            hyperparams=tuple(hyperparams),
            data=self.follower_data,
            bounds=self.coeff_bounds,
            training_cases=self.training_cases,
            variables=self.variables,
            n_var=self.n_coefficients,
            s_star_threshold=self.s_star_threshold,
        )
        algorithm = GA(
            pop_size=self.follower_pop_size,
            eliminate_duplicates=True,
        )
        res = minimize(
            problem, algorithm,
            ("n_gen", self.follower_n_gen),
            verbose=False,
        )
        return res.X

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the temporary memory-mapped folder if it was created."""
        if self._temp_folder and os.path.exists(self._temp_folder):
            try:
                shutil.rmtree(self._temp_folder)
                logger.info("Cleaned up temp folder: %s", self._temp_folder)
            except OSError as exc:
                logger.error("Could not remove temp folder %s: %s", self._temp_folder, exc)

    def __del__(self) -> None:
        self.cleanup()
