import pytest
from kuhn.game import (
    ALL_DEALS,
    get_current_player,
    get_infoset_key,
    get_legal_actions,
    get_payoff,
    is_terminal
)

def test_all_deals_count_and_uniqueness():
    assert len(ALL_DEALS) == 6
    assert all(p1 != p2 for p1, p2 in ALL_DEALS)

def test_terminal_states():
    for h in ['pp', 'bp', 'bb', 'pbp', 'pbb']:
        assert is_terminal(h) is True
        assert get_legal_actions(h) == ()

    for h in ['', 'p', 'b', 'pb']:
        assert is_terminal(h) is False
        assert get_legal_actions(h) == ('p', 'b')

def test_current_player_and_infoset_keys():
    # cards = (P1=Q, P2=K) -> (1, 2)
    assert get_current_player("") == 0
    assert get_infoset_key((1, 2), "") == "Q"

    assert get_current_player("p") == 1
    assert get_infoset_key((1, 2), "p") == "Kp"

    assert get_current_player("pb") == 0
    assert get_infoset_key((1, 2), "pb") == "Qpb"

    # cards = (P1=K, P2=J) -> (2, 0)
    assert get_infoset_key((2, 0), "p") == "Jp"

def test_fold_payoffs():
    # Fold payoffs depend only on who folded, never on the cards
    assert get_payoff((0, 2), "bp") == 1.0    # P1 bluffed J, P2 folded K
    assert get_payoff((2, 0), "pbp") == -1.0  # P1 checked K, P2 bluffed J, P1 folded


def test_showdown_payoffs():
    # (K, Q) -> P1 wins
    assert get_payoff((2, 1), "pp") == 1.0
    assert get_payoff((2, 1), "bb") == 2.0
    assert get_payoff((2, 1), "pbb") == 2.0

    # (J, Q) -> P1 loses
    assert get_payoff((0, 1), "pp") == -1.0
    assert get_payoff((0, 1), "bb") == -2.0
    assert get_payoff((0, 1), "pbb") == -2.0


def test_non_terminal_payoff_raises():
    with pytest.raises(ValueError):
        get_payoff((2, 1), "pb")