"""
Checkpoint / resume support for the leader GA.

Broadly follows pymoo's documented checkpointing pattern
(https://pymoo.org/misc/checkpoint.html) -- pickle the ``Algorithm`` object
after every generation and resume by unpickling it and continuing the
``while algorithm.has_next(): algorithm.next()`` loop -- with two
adjustments forced by this specific problem:

1. We use ``dill`` instead of stdlib ``pickle``. pymoo 0.6's default
   operators (e.g. ``TournamentSelection.func_comp``) are wrapped by a
   decorator that produces a local closure, which plain ``pickle`` cannot
   serialize but ``dill`` can. This matches what pymoo's own docs use, for
   the same reason.

2. Unlike pymoo's docs, we do **not** pickle ``algorithm.problem``.
   ``algorithm.problem`` here is a ``LeaderGA`` instance, and whenever
   ``n_jobs != 1`` (the normal case on a cluster, e.g. ``n_jobs: "auto"``)
   its ``follower_data`` is joblib-memory-mapped (see ``leader.py``). That
   memmapping embeds a raw ``mmap.mmap`` object somewhere in the object
   graph, which *no* pickler -- dill included -- can serialize, since it
   wraps a live OS file mapping rather than plain data. So ``problem`` is
   detached before pickling and must be reattached by the caller after
   loading; the caller also needs to separately track and restore
   ``leader.best_objective``/``best_coefficients``, since those live on the
   (excluded) problem instance rather than on the algorithm itself.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import dill

logger = logging.getLogger(__name__)


def save_checkpoint(path: Union[str, Path], algorithm: Any, leader: Any) -> None:
    """Atomically persist ``algorithm`` (minus its problem) and best-so-far state.

    Writes to a sibling ``.tmp`` file and ``os.replace``s it into place so a
    process killed mid-write (e.g. a SLURM time-limit SIGKILL) never leaves a
    corrupt checkpoint behind.
    """
    path = Path(path)
    problem = algorithm.problem
    algorithm.problem = None
    try:
        state: Dict[str, Any] = {
            "algorithm":         algorithm,
            "best_objective":    leader.best_objective,
            "best_coefficients": leader.best_coefficients,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            dill.dump(state, f)
        os.replace(tmp_path, path)
    finally:
        algorithm.problem = problem
    logger.debug("Checkpoint saved at generation %s -> %s", algorithm.n_gen, path)


def load_checkpoint(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load a checkpoint saved by :func:`save_checkpoint`, or ``None`` if absent.

    The returned ``state["algorithm"]`` has ``problem is None`` -- the
    caller must reattach a freshly constructed problem instance (and
    restore ``best_objective``/``best_coefficients`` onto it from
    ``state``) before calling ``.next()`` again.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        state = dill.load(f)
    logger.info(
        "Loaded checkpoint from %s (generation %s)", path, state["algorithm"].n_gen
    )
    return state
