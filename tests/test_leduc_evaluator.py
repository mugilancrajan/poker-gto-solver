import numpy as np
import pytest
from leduc.evaluator import compute_ev, compute_exploitability
from leduc.solver import LeducCFRPlus


def test_leduc_uniform_strategy_is_exploitable():
    # Empty profile defaults to uniform random play at every infoset
    expl = compute_exploitability({})
    assert expl > 0.5


def test_leduc_cfr_plus_exact_ev_and_low_exploitability():
    solver = LeducCFRPlus()
    solver.train(iterations=1_500)
    profile = solver.get_average_strategy_profile()

    exact_ev = compute_ev(profile)
    expl = compute_exploitability(profile)

    assert exact_ev == pytest.approx(-0.08561, abs=0.005)
    assert expl < 0.015