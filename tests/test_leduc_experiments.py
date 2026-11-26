import pytest
from leduc.experiments import evaluate_leak, inject_p2_leak
from leduc.inspect_strategy import load_profile


def test_leaking_profiles_remain_valid_probability_distributions():
    gto = load_profile()
    for leak in ("over_fold", "calling_station", "no_bluff"):
        leaking = inject_p2_leak(gto, leak, 0.25)
        for probs in leaking.values():
            assert probs.sum() == pytest.approx(1.0, abs=1e-9)
            assert (probs >= -1e-9).all() and (probs <= 1.0 + 1e-9).all()


def test_exploitative_ev_dominates_gto_ev_against_leaks():
    gto = load_profile()
    base = evaluate_leak(gto, "over_fold", 0.0)
    assert base["left_on_table"] == pytest.approx(0.0, abs=1e-3)

    over_fold = evaluate_leak(gto, "over_fold", 0.30)
    assert over_fold["expl_ev"] > over_fold["gto_ev"]
    assert over_fold["left_on_table"] > 0.05