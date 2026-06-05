"""
Example script — bi-level elastic-net hyperparameter optimisation
using surrogate-based RANS model correction.

Author : Louenas Zemmour
Date   : November, 2025. Refactored in June 2026

Usage
-----
    python run_bilevel.py                          # uses default config path
    python run_bilevel.py path/to/config.json      # explicit config

Config file layout
------------------
See the annotated example at the bottom of this file (``EXAMPLE_CONFIG``).
The most important fields are:

    cases.training    — list of case names used by the follower
    cases.validation  — list of case names used by the leader (leave-one-out)
    variables         — physical quantities predicted by the surrogates
    paths.surrogates  — {case: {var: "/path/to/surrogate.pkl"}}
    paths.scalers     — {case: "/path/to/cand_scaler.pkl"}
    paths.les         — {case: "path/to/les/"}
    paths.baseline    — {case: ""path/to/baseline/"}
    coefficients.n    — number of correction coefficients (Gamma dimension)
    coefficients.l_bounds / u_bounds  — per-coefficient bounds
    ga.leader.*       — pop_size, n_gen for the leader GA
    ga.follower.*     — pop_size, n_gen for the follower GA
    n_jobs            — parallel follower solves (-1 = all cores, 1 = serial)
    s_star_threshold  — S* mask threshold (default 0.8)
    output_dir        — where to write results JSON and logs
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
import warnings

from pymoo.algorithms.soo.nonconvex.ga  import GA
from pymoo.core.callback                import Callback
from pymoo.optimize                     import minimize

from utils import (setup_logging, build_data_dicts)

# ----------------------------------------------------------------------------
from bilevel_optim.leader import LeaderGA

warnings.filterwarnings("ignore", category=RuntimeWarning)
# ============================================================================
# 3.  Progress callback
# ============================================================================

class ProgressCallback(Callback):
    """Logs per-generation leader statistics and records history."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict] = []
        self._t0 = time.time()

    def notify(self, algorithm) -> None:
        F   = algorithm.pop.get("F")[:, 0]
        X   = algorithm.pop.get("X")
        best_i = int(F.argmin())

        entry = {
            "generation": algorithm.n_gen,
            "best_alpha": float(X[best_i, 0]),
            "best_beta":  float(X[best_i, 1]),
            "best_f":     float(F[best_i]),
            "mean_f":     float(F.mean()),
            "elapsed_s":  round(time.time() - self._t0, 1),
        }
        self.history.append(entry)
        logging.info(
            "Leader gen %3d | best_f=%.6f  (α=%.4f  β=%.4f) | "
            "mean_f=%.6f | elapsed=%ss",
            entry["generation"], entry["best_f"],
            entry["best_alpha"], entry["best_beta"],
            entry["mean_f"], entry["elapsed_s"],
        )


# ============================================================================
# 4.  Main optimisation routine
# ============================================================================

def run(config_: dict) -> dict:
    """
    Full bi-level optimisation run.

    Parameters
    ----------
    config_ : dict
        Parsed JSON configuration.

    Returns
    -------
    results : dict
        Serialisable dict with hyperparameters, coefficients, and history.
    """
    output_dir = Path(config_.get("output_dir", "results"))
    setup_logging(output_dir)

    logging.info("=" * 65)
    logging.info("  Bi-level elastic-net RANS correction optimisation")
    logging.info("=" * 65)
    logging.info("Training cases   : %s", config_["cases"]["training"])
    logging.info("Validation cases : %s", config_["cases"]["validation"])
    logging.info("Variables        : %s", config_["variables"])

    # ------------------------------------------------------------------
    # 4a.  Load data
    # ------------------------------------------------------------------
    leader_data, follower_data = build_data_dicts(config_)

    n_coeff  = config_["coefficients"]["n"]
    l_bounds = config_["coefficients"]["l_bounds"]
    u_bounds = config_["coefficients"]["u_bounds"]

    # ------------------------------------------------------------------
    # 4b.  Build LeaderGA
    # ------------------------------------------------------------------
    logging.info("── Initialising LeaderGA ──")
    leader = LeaderGA(
        leader_data      = leader_data,
        follower_data    = follower_data,
        validation_cases = config_["cases"]["validation"],
        training_cases   = config_["cases"]["training"],
        variables        = config_["variables"],
        n_coefficients   = n_coeff,
        bounds           = (l_bounds, u_bounds),
        follower_pop_size = config_["ga"]["follower"]["pop_size"],
        follower_n_gen    = config_["ga"]["follower"]["n_gen"],
        n_jobs            = config_.get("n_jobs", 1),
        s_star_threshold  = config_.get("s_star_threshold", 0.8),
        use_memmap        = config_.get("n_jobs", 1) != 1,
    )

    # ------------------------------------------------------------------
    # 4c.  Run the leader GA
    # ------------------------------------------------------------------
    leader_algo = GA(
        pop_size=config_["ga"]["leader"]["pop_size"],
        eliminate_duplicates=True,
    )
    callback = ProgressCallback()

    logging.info(
        "── Starting leader GA  (pop=%d  n_gen=%d) ──",
        config_["ga"]["leader"]["pop_size"],
        config_["ga"]["leader"]["n_gen"],
    )
    t_start = time.time()

    try:
        res = minimize(
            leader,
            leader_algo,
            ("n_gen", config_["ga"]["leader"]["n_gen"]),
            callback=callback,
            verbose=False,          # we handle logging ourselves
        )
    finally:
        leader.cleanup()

    elapsed = time.time() - t_start
    logging.info("── Optimisation complete in %.1f s ──", elapsed)
    logging.info("Best α = %.6f   β = %.6f", res.X[0], res.X[1])
    logging.info("Best validation loss = %.6f", leader.best_objective)

    # ------------------------------------------------------------------
    # 4d.  Collect and save results
    # ------------------------------------------------------------------
    results = {
        "hyperparameters": {
            "alpha": float(res.X[0]),
            "beta":  float(res.X[1]),
        },
        "best_val_loss":  leader.best_objective,
        "coefficients":   (
            leader.best_coefficients.tolist()
            if leader.best_coefficients is not None else None
        ),
        "generation_history": callback.history,
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = output_dir / "bilevel_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logging.info("Results saved → %s", out_path)

    return results


# ============================================================================
# 5.  Entry point
# ============================================================================
if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    results = run(config)

    print("\n" + "=" * 55)
    print(f"  alpha = {results['hyperparameters']['alpha']:.6f}")
    print(f"  beta = {results['hyperparameters']['beta']:.6f}")
    print(f"  validation loss = {results['best_val_loss']:.6f}")
    print(f"  elapsed         = {results['elapsed_seconds']} s")
    print("=" * 55)
