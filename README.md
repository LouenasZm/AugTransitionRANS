# Bi-Level Optimisation for RANS Model Improvement


Surrogate-assisted bi-level optimisation framework for improving RANS models from high-fidelity data.

The project combines:

- symbolic regression over a dictionary of candidate functions,
- surrogate-supported optimisation,
- elastic-net regularisation,
- and a bi-level formulation used to tune the regularisation hyperparameters automatically.

The lower level learns correction coefficients for the dictionary terms, while the upper level searches for the best regularisation parameters that generalise well on validation cases.

## Author

Louenas Zemmour, PhD student at Sorbonne Université from 2023 to 2026

Email: louenas.zemmour@sorbonne-universite.fr

## Features

- Bi-level optimisation with a follower/leader split.
- Elastic-net regularised coefficient fitting.
- Masked, normalised MSE loss based on LES reference data.
- Support for multiple flow cases and multiple target variables.
- Joblib-based parallel follower solves for larger experiments.

## Repository Layout

```text
src/
  bilevel/
    follower.py
    leader.py
    loss.py
  bilevel_optim/
    __init__.py
    follower.py
    leader.py
    loss.py
tests/
  test_bilevel.py
```

## Requirements

Install the package with pip:

```bash
pip install .
```

For development, use editable mode and include the test extra:

```bash
pip install -e ".[test]"
```

Main dependencies:

- `numpy`
- `scipy`
- `scikit-learn`
- `pymoo`
- `joblib`
- `pytest` for running the test suite

## Installation

The repository uses a source layout, so the easiest local setup is:

```bash
git clone git@github.com:LouenasZm/AugTransitionRANS.git
cd AugTransitionRANS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test]"
```

This installs the project in editable mode, so you can import `bilevel_optim` directly without setting `PYTHONPATH`.

## Quick Start

```python
from bilevel_optim import ElasticNetFollower, LeaderGA
```

Typical workflow:

1. Prepare training and validation data dictionaries with surrogate models, LES reference fields and baselines.
2. Instantiate `ElasticNetFollower` to solve the lower-level coefficient fitting problem.
3. Instantiate `LeaderGA` to optimise the hyperparameters `(alpha, beta)`.
4. Run `pymoo.optimize.minimize(...)` on either problem.

See `tests/test_bilevel.py` for a self-contained synthetic example of the full pipeline.

## Running on an HPC cluster (SLURM)

`LeaderGA` parallelises follower solves locally with `joblib` — each task is pure
Python/NumPy (surrogate `.predict()` calls plus a pymoo DE run), so this already works
correctly inside a single-node SLURM allocation: `sbatch`/`srun` just restrict which
cores the job's process tree may use, and joblib's workers stay inside that cgroup. No
MPI or multi-node distributed backend is needed to run on a cluster.

Two things do need to adapt between a laptop and a SLURM job:

- **Worker count.** Set `"n_jobs": "auto"` in your config instead of a hardcoded
  number. `LeaderGA` will then pick up the job's actual core allocation from
  `SLURM_CPUS_PER_TASK`/`SLURM_JOB_CPUS_PER_NODE` (falling back to all local cores when
  not running under SLURM), so the same config works unmodified on your laptop and on
  the cluster. See `src/bilevel/hpc_utils.py`.
- **BLAS thread oversubscription.** Each follower solve is wrapped in
  `threadpoolctl.threadpool_limits(1)`, so `n_jobs` worker processes won't each also
  spawn their own OpenBLAS/MKL threads and thrash a shared node.

See `examples/CfdDrivenOptim/submit_slurm.sh` for a template `sbatch` script.

## Testing

Run the smoke tests with:

```bash
pytest -q
```

Or execute the standalone test script directly:

```bash
python tests/test_bilevel.py
```
