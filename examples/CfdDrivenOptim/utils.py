"""
    This module contains utility functions for the ercoftacTRANS example.
    These functions are the ones requiring adaptation to the specific case
    and are called by the main script

    This example requires:
        - Downloading the high-fidelity numerical
          database from https://doi.org/10.5281/zenodo.17166216
        - Installing the postprocessing
          package from: https://github.com/LouenasZm/ppMusicaa.git

    Author: Louenas Zemmour, June 2026
"""


import pickle
import joblib
import logging
import sys
import numpy            as np
from typing             import Any
from pathlib            import Path
from sklearn.metrics    import mean_squared_error
from ppModule.interface import PostProcessMusicaa
from preprocess          import ErcoftacPreprocess

from AugCfd.surrogates.surrogates import MusicaaSurrogate
# ============================================================================
#                          1.  Logging setup
# ============================================================================
def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "bilevel.log"
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Logging to %s", log_path)


# ============================================================================
#                          2.  Data loading
# ============================================================================

def load_surrogates(
                cases:          list[str],
                variables:      list[str],
                surrogate_paths: dict,          # {case: {var: path_str}}
            ) -> dict[str, dict[str, Any]]:
    """Load pickle surrogates for a given set of cases."""
    surrogates: dict = {}
    for case in cases:
        surrogates[case] = {}
        for var in variables:
            path = surrogate_paths[case][var]
            with open(path, "rb") as f:
                surrogates[case][var] = pickle.load(f)
            logging.info("  Loaded surrogate  case=%-8s  var=%s  (%s)", case, var, path)
    return surrogates


def load_scalers(
                cases: list[str],
                scaler_paths: dict,         # {case: path_str}
            ) -> dict[str, Any]:
    """Load joblib scalers (one per case)."""
    scalers: dict = {}
    for case in cases:
        scalers[case] = joblib.load(scaler_paths[case])
        logging.info("  Loaded scaler     case=%-8s  (%s)", case, scaler_paths[case])
    return scalers


def load_les(
                cases: list[str],
                variables: list[str],
                les_cfd: dict,
                baseline_cfg: dict,
            ) -> dict[str, dict[str, np.ndarray]]:
    """Load LES reference arrays and S* mask fields."""
    les: dict = {}
    for case in cases:
        pp = PostProcessMusicaa(config=les_cfd[case])
        pp.compute_qty(qty="tauw")
        les_data = pp.return_stats()
        pp_obj      = ErcoftacPreprocess(case=case,
                                         config=baseline_cfg)
        les[case]   = pp_obj.preprocess_les(grid=pp.config["grid"],
                                    stats=les_data, keylist=variables)
        logging.info("  Loaded LES data   case=%s", case)
    return les

def compute_baseline_losses(
                cases:        list[str],
                variables:    list[str],
                baseline_cfg: dict,             # {case: path}
                les: dict,
            ) -> dict[str, dict[str, float]]:
    """
    Resolve baseline MSE values.
    """
    baseline_data = {}
    losses: dict = {}
    for case in cases:
        pp = PostProcessMusicaa(config=baseline_cfg[case])
        pp.compute_qty(qty="tauw")
        baseline_stats = pp.return_stats()
        pp_obj = ErcoftacPreprocess(case=case, config=baseline_cfg)
        baseline_data[case]   = pp_obj.preprocess_rans(grid=pp.config["grid"],
                    stats=baseline_stats,
                    keylist=variables)
        losses[case] = {}
        for var in variables:
            losses[case][var] = mean_squared_error(
                baseline_data[case][var][np.isfinite(les[case][var])].flatten(),
                les[case][var][np.isfinite(les[case][var])].flatten())
    return losses


def build_data_dicts(config: dict) -> tuple[dict, dict]:
    """
    Assemble the leader_data and follower_data dicts expected by
    :class:`~bilevel_optim.leader.LeaderGA`.

    Returns
    -------
    leader_data : dict
        Keys: surrogates, les, scalers, baseline
        All keyed by validation case names.

    follower_data : dict
        Keys: surrogates, les, scalers, baseline
        baseline has sub-key "loss" as required by the follower.
        All keyed by training case names.
    """
    training_cases   = config["cases"]["training"]
    validation_cases = config["cases"]["validation"]
    variables        = config["variables"]
    paths            = config["paths"]

    logging.info("── Loading validation data (%s) ──", validation_cases)
    val_les        = load_les(validation_cases, variables.copy(), les_cfd=config["les"],
                              baseline_cfg=config["baseline"])
    val_surrogates = load_surrogates(validation_cases, variables.copy(), paths["surrogates"])
    val_scalers    = load_scalers(validation_cases, paths["scalers"])
    val_baseline   = compute_baseline_losses(
        validation_cases, variables.copy(), config["baseline"], val_les
    )

    leader_data = {
        "surrogates": val_surrogates,
        "les":        val_les,
        "scalers":    val_scalers,
        "baseline":   val_baseline,          # {case: {var: float}}
    }

    logging.info("── Loading training data (%s) ──", training_cases)
    train_les        = load_les(training_cases, variables.copy(), les_cfd=config["les"],
                                baseline_cfg=config["baseline"])
    train_surrogates = load_surrogates(training_cases, variables.copy(), paths["surrogates"])
    train_scalers    = load_scalers(training_cases, paths["scalers"])
    train_baseline   = compute_baseline_losses(
        training_cases, variables.copy(), config["baseline"], train_les
    )

    follower_data = {
        "surrogates": train_surrogates,
        "les":        train_les,
        "scalers":    train_scalers,
        "baseline":   {"loss": train_baseline},  # follower expects this nesting
    }

    return leader_data, follower_data
