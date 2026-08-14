"""
HPC-awareness helpers for the bi-level optimisation framework.

The leader's follower solves are parallelised locally via ``joblib`` (see
``leader.py``). That backend works fine inside a single-node SLURM allocation —
SLURM's cgroup restricts which cores the job's process tree may use, and
``multiprocessing``/``loky`` workers spawned from within that allocation stay
inside it. The one thing that does need to adapt between a laptop and a cluster
job is *how many* workers to spawn: ``os.cpu_count()`` reports the physical
node's core count, not the (possibly smaller) cgroup allocation, so a hardcoded
``n_jobs`` or ``n_jobs=-1`` can quietly oversubscribe a shared node.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Union

logger = logging.getLogger(__name__)


def resolve_n_jobs(n_jobs: Union[int, str]) -> int:
    """Resolve ``n_jobs``, supporting ``"auto"`` for SLURM-aware core detection.

    Parameters
    ----------
    n_jobs:
        Either an ``int`` (joblib semantics: ``1`` = serial, ``-1`` = all
        detected cores, ``N`` = N workers), passed through unchanged, or the
        string ``"auto"``, which picks the number of cores actually available
        to this job:

        1. ``SLURM_CPUS_PER_TASK`` (set when the job requests
           ``--cpus-per-task``);
        2. ``SLURM_JOB_CPUS_PER_NODE`` (SLURM's per-node count, e.g. ``"20"``
           or ``"20(x2)"`` for multi-node jobs — the leading integer is used);
        3. ``len(os.sched_getaffinity(0))``, which respects cgroup/taskset
           restrictions (unlike ``os.cpu_count()``), where available;
        4. ``os.cpu_count()``;
        5. ``1`` if nothing else could be determined.
    """
    if isinstance(n_jobs, int):
        return n_jobs
    if n_jobs != "auto":
        raise ValueError(f"n_jobs must be an int or 'auto', got {n_jobs!r}")

    if "SLURM_CPUS_PER_TASK" in os.environ:
        resolved = int(os.environ["SLURM_CPUS_PER_TASK"])
        logger.info("n_jobs='auto' -> %d (from SLURM_CPUS_PER_TASK)", resolved)
        return resolved

    if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        match = re.match(r"\d+", os.environ["SLURM_JOB_CPUS_PER_NODE"])
        if match:
            resolved = int(match.group())
            logger.info(
                "n_jobs='auto' -> %d (from SLURM_JOB_CPUS_PER_NODE=%s)",
                resolved, os.environ["SLURM_JOB_CPUS_PER_NODE"],
            )
            return resolved

    if hasattr(os, "sched_getaffinity"):
        resolved = len(os.sched_getaffinity(0))
        logger.info("n_jobs='auto' -> %d (from os.sched_getaffinity)", resolved)
        return resolved

    resolved = os.cpu_count()
    if resolved:
        logger.info("n_jobs='auto' -> %d (from os.cpu_count)", resolved)
        return resolved

    logger.warning("n_jobs='auto' could not detect any core count, falling back to 1")
    return 1
