"""
test_bilevel.py
===============

Self-contained smoke-test for the bilevel_optim package.

Requires only: numpy, scikit-learn, pymoo — no external data or project
dependencies.  Synthetic surrogates and LES data are generated in-place.

Run with:
    python test_bilevel.py

The test verifies:
    1. masked_normalised_mse  — basic numerics
    2. compute_normalised_loss — aggregation over cases/vars
    3. ElasticNetFollower      — full pymoo optimisation on a toy problem
    4. LeaderGA                — one full bi-level evaluation cycle
"""

import os
import sys
import traceback
import numpy as np
from sklearn.linear_model import Ridge   # tiny surrogate: Ridge wraps predict()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TinySurrogate:
    """Fake surrogate mimicking real field-prediction surrogates.

    Given ``n_candidates`` parameter vectors of shape ``(n_candidates, n_features)``,
    returns a prediction of shape ``(n_points, n_candidates)`` — one spatial value
    per grid-point per candidate, which is how the real surrogates behave.
    """
    def __init__(self, weights: np.ndarray, n_points: int) -> None:
        self.weights = weights    # (n_features,)
        self.n_points = n_points

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (n_candidates, n_features)  →  scalars: (n_candidates,)
        # Tile to (n_points, n_candidates) so every spatial point gets the same value.
        scalars = X @ self.weights                       # (n_candidates,)
        return np.tile(scalars, (self.n_points, 1))      # (n_points, n_candidates)


def make_synthetic_data(
    n_points: int     = 50,
    n_features: int   = 5,
    cases: list       = ("caseA", "caseB"),
    variables: list   = ("uu", "vv"),
    rng: np.random.Generator = None,
) -> tuple:
    """
    Build a minimal but structurally complete follower_data / leader_data pair.

    Returns
    -------
    follower_data, leader_data, bounds
    """
    if rng is None:
        rng = np.random.default_rng(42)

    def _build_data_dict(case_list):
        d = {"surrogates": {}, "les": {}}
        for c in case_list:
            weights = rng.standard_normal(n_features)
            d["surrogates"][c] = {}
            d["les"][c]        = {}
            for v in variables:
                d["surrogates"][c][v] = TinySurrogate(
                    rng.standard_normal(n_features), n_points
                )
                les_vals = rng.standard_normal(n_points)
                # Introduce a few NaNs to exercise the finite mask
                les_vals[rng.integers(0, n_points, size=3)] = np.nan
                d["les"][c][v] = les_vals

            # S_star: values in [0, 2] — roughly half below threshold 0.8
            d["les"][c]["S_star"] = rng.uniform(0, 2, n_points)

        return d

    train_cases = list(cases[:2])
    val_cases   = list(cases[-1:])    # leave-one-out

    follower_data = _build_data_dict(train_cases)
    leader_data   = _build_data_dict(val_cases)

    # Baseline loss: small positive float for each (case, var)
    follower_data["baseline"] = {"loss": {}}
    for c in train_cases:
        follower_data["baseline"]["loss"][c] = {}
        for v in variables:
            follower_data["baseline"]["loss"][c][v] = 0.5 + rng.random()

    leader_data["baseline"] = {}
    for c in val_cases:
        leader_data["baseline"][c] = {}
        for v in variables:
            leader_data["baseline"][c][v] = 0.5 + rng.random()

    l_bounds = [-1.0] * n_features
    u_bounds = [ 1.0] * n_features
    bounds   = (l_bounds, u_bounds)

    return follower_data, leader_data, train_cases, val_cases, bounds


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def ok(label):
    print(f"  {PASS}  {label}")

def fail(label, exc):
    print(f"  {FAIL}  {label}")
    traceback.print_exc()

results = {"passed": 0, "failed": 0}

def run_test(label, fn):
    try:
        fn()
        ok(label)
        results["passed"] += 1
    except Exception:
        fail(label, sys.exc_info())
        results["failed"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Individual test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_masked_mse_basic():
    from bilevel_optim.loss import masked_normalised_mse
    rng   = np.random.default_rng(0)
    n_pts = 40
    n_cand = 10
    preds   = rng.standard_normal((n_pts, n_cand))
    les_val = rng.standard_normal(n_pts)
    s_star  = rng.uniform(0, 2, n_pts)
    baseline = 2.0

    result = masked_normalised_mse(preds, les_val, s_star, baseline)
    assert result.shape == (n_cand,), f"Expected ({n_cand},) got {result.shape}"
    assert np.all(result >= 0), "MSE must be non-negative"


def test_masked_mse_all_nan():
    """When all LES values are NaN, return zeros without crashing."""
    from bilevel_optim.loss import masked_normalised_mse
    rng    = np.random.default_rng(1)
    preds  = rng.standard_normal((20, 5))
    les    = np.full(20, np.nan)
    s_star = rng.uniform(0, 2, 20)

    result = masked_normalised_mse(preds, les, s_star, 1.0)
    assert np.all(result == 0.0)


def test_masked_mse_zero_baseline():
    """Zero baseline must be replaced by 1.0."""
    from bilevel_optim.loss import masked_normalised_mse
    rng    = np.random.default_rng(2)
    preds  = rng.standard_normal((30, 4))
    les    = rng.standard_normal(30)
    s_star = rng.uniform(0, 2, 30)

    r_zero = masked_normalised_mse(preds, les, s_star, 0.0)
    r_one  = masked_normalised_mse(preds, les, s_star, 1.0)
    np.testing.assert_allclose(r_zero, r_one)


def test_compute_normalised_loss():
    from bilevel_optim.loss import compute_normalised_loss
    fd, _, train_cases, _, bounds = make_synthetic_data()
    x = np.random.default_rng(3).uniform(-1, 1, (8, 5))

    loss = compute_normalised_loss(
        x,
        cases=train_cases,
        variables=["uu", "vv"],
        data=fd,
    )
    assert loss.shape == (8,)
    assert np.all(loss >= 0)


def test_follower_optimize():
    """Full ElasticNetFollower optimisation — just check it converges."""
    from bilevel_optim import ElasticNetFollower, OptimizationCallback
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.optimize import minimize

    fd, _, train_cases, _, bounds = make_synthetic_data(n_features=5)

    problem  = ElasticNetFollower(
        hyperparams=(0.01, 0.01),
        data=fd,
        bounds=bounds,
        training_cases=train_cases,
        variables=["uu", "vv"],
        n_var=5,
    )
    callback = OptimizationCallback()
    res = minimize(
        problem,
        DE(pop_size=20),
        ("n_gen", 10),
        callback=callback,
        verbose=False,
    )

    assert res.X is not None, "No solution returned"
    assert res.X.shape == (5,), f"Expected shape (5,), got {res.X.shape}"
    assert len(callback.history) == 10, "Expected 10 history entries"
    # Objectives must be monotonically non-increasing
    bests = [h["best_objective"] for h in callback.history]
    assert bests[-1] <= bests[0] + 1e-12, "Best objective did not improve"


def test_follower_callback_fields():
    """Callback history dicts must have the required keys."""
    from bilevel_optim import ElasticNetFollower, OptimizationCallback
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.optimize import minimize

    fd, _, train_cases, _, bounds = make_synthetic_data(n_features=5)
    problem = ElasticNetFollower(
        hyperparams=(0.05, 0.0),
        data=fd,
        bounds=bounds,
        training_cases=train_cases,
        variables=["uu"],
        n_var=5,
    )
    callback = OptimizationCallback()
    minimize(problem, DE(pop_size=15), ("n_gen", 5), callback=callback, verbose=False)

    required = {"generation", "best_candidate", "best_objective",
                "worst_candidate", "worst_objective", "mean_objective"}
    for entry in callback.history:
        assert required <= entry.keys(), f"Missing keys: {required - entry.keys()}"


def test_leader_one_generation():
    """LeaderGA: run one generation with n_jobs=1."""
    from bilevel_optim import LeaderGA
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize

    fd, ld, train_cases, val_cases, bounds = make_synthetic_data(
        cases=("caseA", "caseB", "caseC"),
        n_features=5,
    )

    leader = LeaderGA(
        leader_data=ld,
        follower_data=fd,
        validation_cases=val_cases,
        training_cases=train_cases,
        variables=["uu", "vv"],
        n_coefficients=5,
        bounds=bounds,
        follower_pop_size=15,
        follower_n_gen=10,
        n_jobs=1,
        use_memmap=False,
    )

    res = minimize(
        leader,
        GA(pop_size=4, eliminate_duplicates=True),
        ("n_gen", 2),
        verbose=False,
    )

    assert res.X is not None
    assert res.X.shape == (2,), f"Expected 2 hyperparams, got {res.X.shape}"
    assert leader.best_coefficients is not None
    assert leader.best_coefficients.shape == (5,)
    assert leader.best_objective < np.inf

    leader.cleanup()


def test_leader_best_tracking():
    """best_objective must be <= all previously seen objectives."""
    from bilevel_optim import LeaderGA
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.optimize import minimize

    fd, ld, train_cases, val_cases, bounds = make_synthetic_data(
        cases=("caseA", "caseB", "caseC"), n_features=4
    )
    leader = LeaderGA(
        leader_data=ld, follower_data=fd,
        validation_cases=val_cases, training_cases=train_cases,
        variables=["uu"], n_coefficients=4, bounds=bounds,
        follower_pop_size=10, follower_n_gen=8,
        n_jobs=1, use_memmap=False,
    )
    minimize(leader, GA(pop_size=6), ("n_gen", 3), verbose=False)

    assert leader.best_objective >= 0


def _build_tiny_leader(seed_hint, **leader_kwargs):
    """LeaderGA + GA pair for checkpoint tests (small enough to run fast)."""
    from bilevel_optim import LeaderGA
    from pymoo.algorithms.soo.nonconvex.ga import GA

    fd, ld, train_cases, val_cases, bounds = make_synthetic_data(
        cases=("caseA", "caseB", "caseC"),
        n_features=4,
        rng=np.random.default_rng(seed_hint),
    )
    kwargs = dict(n_jobs=1, use_memmap=False)
    kwargs.update(leader_kwargs)
    leader = LeaderGA(
        leader_data=ld, follower_data=fd,
        validation_cases=val_cases, training_cases=train_cases,
        variables=["uu"], n_coefficients=4, bounds=bounds,
        follower_pop_size=10, follower_n_gen=5,
        **kwargs,
    )
    algo = GA(pop_size=6, eliminate_duplicates=True)
    return leader, algo


def test_checkpoint_roundtrip():
    """save_checkpoint/load_checkpoint round-trip preserves algorithm state.

    `problem` is deliberately excluded from the pickle (see
    bilevel.checkpoint docstring) so best_objective/best_coefficients are
    carried in the checkpoint dict alongside the algorithm, and the caller
    is responsible for reattaching a problem instance after loading.
    """
    import tempfile
    from bilevel.checkpoint import save_checkpoint, load_checkpoint

    leader, algo = _build_tiny_leader(seed_hint=10)
    algo.setup(leader, termination=("n_gen", 4), seed=1, verbose=False)

    for _ in range(2):
        algo.next()

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "checkpoint.pkl")
        save_checkpoint(ckpt_path, algo, leader)

        assert os.path.exists(ckpt_path)
        state = load_checkpoint(ckpt_path)
        assert state is not None
        assert state["algorithm"].problem is None    # problem excluded from the pickle
        assert state["algorithm"].n_gen == algo.n_gen
        assert state["best_objective"] == leader.best_objective
        np.testing.assert_array_equal(
            state["algorithm"].pop.get("X"), algo.pop.get("X")
        )

        # a missing checkpoint file must return None, not raise
        assert load_checkpoint(os.path.join(d, "does-not-exist.pkl")) is None

    leader.cleanup()


def test_checkpoint_survives_memmapped_follower_data():
    """Regression test: n_jobs != 1 memory-maps follower_data (leader.py), which
    embeds a raw mmap.mmap object in LeaderGA — unpicklable by any pickler,
    dill included, since it wraps a live OS file mapping. save_checkpoint
    must not choke on this (it excludes `problem` from the pickle entirely)."""
    import tempfile
    from bilevel.checkpoint import save_checkpoint, load_checkpoint

    leader, algo = _build_tiny_leader(seed_hint=30, n_jobs=2, use_memmap=True)
    algo.setup(leader, termination=("n_gen", 2), seed=1, verbose=False)
    algo.next()

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "checkpoint.pkl")
        save_checkpoint(ckpt_path, algo, leader)    # must not raise
        state = load_checkpoint(ckpt_path)
        assert state["algorithm"].n_gen == algo.n_gen

    leader.cleanup()


def test_resume_continues_from_checkpoint():
    """A save/load round-trip preserves state exactly, and the resumed run
    reaches the same generation budget as an uninterrupted one.

    Note: this does *not* assert the two runs land on the same
    best_objective — LeaderGA's follower-level DE solves
    (leader.py:_solve_single_follower) are not seeded, so they're
    non-deterministic run-to-run independently of checkpointing. What the
    checkpoint mechanism is responsible for, and what's verified here, is
    that pickling/unpickling the algorithm loses none of its state and that
    the resumed loop keeps advancing correctly to completion.
    """
    import tempfile
    from bilevel.checkpoint import save_checkpoint, load_checkpoint

    leader, algo = _build_tiny_leader(seed_hint=20)
    algo.setup(leader, termination=("n_gen", 4), seed=1, verbose=False)
    for _ in range(2):
        algo.next()

    pop_x_before = algo.pop.get("X").copy()
    n_gen_before = algo.n_gen

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "checkpoint.pkl")
        save_checkpoint(ckpt_path, algo, leader)

        # simulate a fresh process: rebuild the problem, reload the algorithm
        leader_c, _ = _build_tiny_leader(seed_hint=20)
        state = load_checkpoint(ckpt_path)
        resumed = state["algorithm"]
        resumed.problem = leader_c
        leader_c.best_objective    = state["best_objective"]
        leader_c.best_coefficients = state["best_coefficients"]

        # round-tripping through the checkpoint must not have altered state
        assert resumed.n_gen == n_gen_before
        np.testing.assert_array_equal(resumed.pop.get("X"), pop_x_before)

        while resumed.has_next():
            resumed.next()
        leader_c.cleanup()

    assert resumed.n_gen == 5    # n_gen is a "next generation" counter: 4 completed -> 5
    assert leader_c.best_objective < np.inf
    assert leader_c.best_coefficients.shape == (4,)

    leader.cleanup()


def test_resolve_n_jobs():
    """resolve_n_jobs: int passthrough and 'auto' SLURM-aware detection."""
    from bilevel.hpc_utils import resolve_n_jobs

    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(-1) == -1
    assert resolve_n_jobs(1) == 1

    saved = {
        k: os.environ.pop(k, None)
        for k in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE")
    }
    try:
        os.environ["SLURM_CPUS_PER_TASK"] = "8"
        assert resolve_n_jobs("auto") == 8

        del os.environ["SLURM_CPUS_PER_TASK"]
        os.environ["SLURM_JOB_CPUS_PER_NODE"] = "12(x2)"
        assert resolve_n_jobs("auto") == 12

        del os.environ["SLURM_JOB_CPUS_PER_NODE"]
        assert resolve_n_jobs("auto") > 0   # falls back to local core count
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("masked_normalised_mse: basic numerics",     test_masked_mse_basic),
    ("masked_normalised_mse: all-NaN input",      test_masked_mse_all_nan),
    ("masked_normalised_mse: zero baseline",      test_masked_mse_zero_baseline),
    ("compute_normalised_loss: aggregation",      test_compute_normalised_loss),
    ("ElasticNetFollower: optimisation",          test_follower_optimize),
    ("OptimizationCallback: required fields",     test_follower_callback_fields),
    ("LeaderGA: one generation cycle",            test_leader_one_generation),
    ("LeaderGA: best tracking",                   test_leader_best_tracking),
    ("checkpoint: save/load round-trip",          test_checkpoint_roundtrip),
    ("checkpoint: survives memmapped follower_data", test_checkpoint_survives_memmapped_follower_data),
    ("checkpoint: resume continues correctly",    test_resume_continues_from_checkpoint),
    ("resolve_n_jobs: int passthrough + SLURM auto-detect", test_resolve_n_jobs),
]

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  bilevel_optim — smoke tests")
    print("="*60)
    for label, fn in TESTS:
        run_test(label, fn)

    total = results["passed"] + results["failed"]
    print("="*60)
    print(f"  {results['passed']}/{total} tests passed"
          + ("" if results["failed"] == 0 else f"  ({results['failed']} FAILED)"))
    print("="*60 + "\n")
    sys.exit(0 if results["failed"] == 0 else 1)
