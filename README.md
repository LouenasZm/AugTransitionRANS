# Bi-Level Optimization for RANS Model Improvement

Surrogate-assisted bi-level optimisation framework for improving RANS models from high-fidelity data.

The project combines:

- symbolic regression over a dictionary of candidate functions,
- surrogate-supported optimisation,
- elastic-net regularisation,
- and a bi-level formulation used to tune the regularisation hyperparameters automatically.

The lower level learns correction coefficients for the dictionary terms, while the upper level searches for the best regularisation strengths that generalise well on validation cases.

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

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
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
git clone <your-github-repository-url>
cd bi_level_optim
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

If you want the environment variable to persist, add the `PYTHONPATH` line to your shell profile.

## Quick Start

```python
from bilevel_optim import ElasticNetFollower, LeaderGA
```

Typical workflow:

1. Prepare training and validation data dictionaries with surrogate models, LES reference fields, scalers, and baselines.
2. Instantiate `ElasticNetFollower` to solve the lower-level coefficient fitting problem.
3. Instantiate `LeaderGA` to optimise the hyperparameters `(alpha, beta)`.
4. Run `pymoo.optimize.minimize(...)` on either problem.

See `tests/test_bilevel.py` for a self-contained synthetic example of the full pipeline.

## Testing

Run the smoke tests with:

```bash
PYTHONPATH="$PWD/src" pytest -q
```

Or execute the standalone test script directly:

```bash
PYTHONPATH="$PWD/src" python tests/test_bilevel.py
```
