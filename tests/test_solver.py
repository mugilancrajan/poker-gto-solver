import pytest
from kuhn.solver import VanillaCFR
from kuhn.evaluator import compute_ev, compute_exploitability
from kuhn.solver import CFRPlus, VanillaCFR


def test_vanilla_cfr_converges_to_kuhn_nash():
    solver = VanillaCFR()
    avg_ev = solver.train(iterations=20_000)

    # 1. All 12 Kuhn Poker infosets must be discovered
    assert len(solver.nodes) == 12

    # 2. Player 1's average EV should converge to -1/18 (-0.05556)
    expected_ev = -1.0 / 18.0
    assert avg_ev == pytest.approx(expected_ev, abs=0.005)

    # 3. Check Player 2's unique Nash Equilibrium frequencies
    profile = solver.get_average_strategy_profile()

    # P2 calls with Queen ('Qb', action index 1 = 'b') ~ 1/3 of the time
    assert profile["Qb"][1] == pytest.approx(1.0 / 3.0, abs=0.02)

    # P2 bluffs with Jack after a check ('Jp', action index 1 = 'b') ~ 1/3 of the time
    assert profile["Jp"][1] == pytest.approx(1.0 / 3.0, abs=0.02)

    # P2 always bets King after a check ('Kp') and always calls King vs bet ('Kb')
    assert profile["Kp"][1] == pytest.approx(1.0, abs=0.01)
    assert profile["Kb"][1] == pytest.approx(1.0, abs=0.01)

def test_cfr_plus_outperforms_vanilla_cfr():
    vanilla = VanillaCFR()
    cfr_plus = CFRPlus()

    vanilla.train(iterations=1_000)
    cfr_plus.train(iterations=1_000)

    vanilla_profile = vanilla.get_average_strategy_profile()
    cfr_plus_profile = cfr_plus.get_average_strategy_profile()

    vanilla_expl = compute_exploitability(vanilla_profile)
    cfr_plus_expl = compute_exploitability(cfr_plus_profile)

    # CFR+ should lock onto -1/18 EV and beat Vanilla CFR by >10x in exploitability
    assert compute_ev(cfr_plus_profile) == pytest.approx(-1.0 / 18.0, abs=0.002)
    assert cfr_plus_expl < vanilla_expl / 10.0