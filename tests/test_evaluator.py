import numpy as np
import pytest
from kuhn.evaluator import compute_ev, compute_exploitability
from kuhn.solver import VanillaCFR


def test_analytical_nash_has_zero_exploitability():
    # Kuhn's alpha = 0 Nash Equilibrium
    nash_profile = {
        # Player 1: always check first
        "J": np.array([1.0, 0.0]),
        "Q": np.array([1.0, 0.0]),
        "K": np.array([1.0, 0.0]),
        # Player 1 facing bet after check ('pb')
        "Jpb": np.array([1.0, 0.0]),
        "Qpb": np.array([2.0 / 3.0, 1.0 / 3.0]),
        "Kpb": np.array([0.0, 1.0]),
        # Player 2 after check ('p')
        "Jp": np.array([2.0 / 3.0, 1.0 / 3.0]),
        "Qp": np.array([1.0, 0.0]),
        "Kp": np.array([0.0, 1.0]),
        # Player 2 facing bet ('b')
        "Jb": np.array([1.0, 0.0]),
        "Qb": np.array([2.0 / 3.0, 1.0 / 3.0]),
        "Kb": np.array([0.0, 1.0]),
    }

    assert compute_ev(nash_profile) == pytest.approx(-1.0 / 18.0, abs=1e-12)
    assert compute_exploitability(nash_profile) == pytest.approx(0.0, abs=1e-12)


def test_always_pass_strategy_is_heavily_exploitable():
    # A bot that always checks/folds should get crushed by the Nemesis
    always_pass = {
        key: np.array([1.0, 0.0])
        for key in ["J", "Q", "K", "Jpb", "Qpb", "Kpb", "Jp", "Qp", "Kp", "Jb", "Qb", "Kb"]
    }
    assert compute_exploitability(always_pass) > 0.5


def test_vanilla_cfr_exploitability_converges_near_zero():
    solver = VanillaCFR()
    solver.train(iterations=20_000)
    profile = solver.get_average_strategy_profile()

    assert compute_ev(profile) == pytest.approx(-1.0 / 18.0, abs=0.005)
    assert compute_exploitability(profile) < 0.005