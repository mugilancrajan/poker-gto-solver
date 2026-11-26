import pytest
from leduc.game import (
    ALL_DEALS,
    apply_action,
    get_current_player,
    get_infoset_key,
    get_legal_actions,
    get_payoff,
    is_terminal,
)


def test_weighted_deals_sum_to_one():
    assert len(ALL_DEALS) == 24
    total_prob = sum(prob for _, prob in ALL_DEALS)
    assert total_prob == pytest.approx(1.0, abs=1e-12)


def test_legal_actions_and_round_transition():
    assert get_legal_actions("") == ("c", "r")
    assert get_legal_actions("r") == ("f", "c", "r")
    assert get_legal_actions("rr") == ("f", "c")

    # Round 1 check-check automatically transitions to Round 2
    r2_start = apply_action("c", "c")
    assert r2_start == "cc/"
    assert not is_terminal(r2_start)
    assert get_current_player(r2_start) == 0
    assert get_legal_actions(r2_start) == ("c", "r")


def test_infoset_keys():
    cards = (0, 2, 1)  # P1=J, P2=K, Board=Q
    assert get_infoset_key(cards, "") == "J|"
    assert get_infoset_key(cards, "r") == "K|r"
    assert get_infoset_key(cards, "cc/") == "J|Q|cc/"
    assert get_infoset_key(cards, "cc/r") == "K|Q|cc/r"


def test_fold_and_showdown_payoffs():
    # Fold in Round 1 after P1 bets 2 -> P2 folds, P1 wins P2's 1-chip ante
    assert get_payoff((0, 2, 1), "rf") == 1.0

    # Max pot: ante(1) + R1 cap(4) + R2 cap(8) = 13 chips each
    # P1=J, P2=K, Board=J -> P1 pairs the Jack and beats P2's King
    assert get_payoff((0, 2, 0), "rrc/rrc") == 13.0

    # High card when neither pairs: P1=Q, P2=K, Board=J -> P2 wins
    assert get_payoff((1, 2, 0), "cc/cc") == -1.0

    # Exact tie: both hold Queen on a Jack board -> split pot (0.0)
    assert get_payoff((1, 1, 0), "rc/rc") == 0.0