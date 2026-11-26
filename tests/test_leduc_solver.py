import pytest
from leduc.solver import LeducCFRPlus


def test_leduc_cfr_plus_discovers_288_infosets_and_converges():
    solver = LeducCFRPlus()
    avg_ev = solver.train(iterations=1_500)

    # Leduc Hold'em has exactly 288 information sets
    assert len(solver.nodes) == 288

    # Ground-truth Player 1 Nash Equilibrium EV is ~ -0.08561
    assert avg_ev == pytest.approx(-0.08561, abs=0.005)
    