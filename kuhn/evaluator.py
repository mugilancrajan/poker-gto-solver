import numpy as np
from kuhn.game import (
    ALL_DEALS,
    CARD_NAMES,
    DECK,
    get_current_player,
    get_infoset_key,
    get_legal_actions,
    get_payoff,
    is_terminal,
)

StrategyProfile = dict[str, np.ndarray]


def _get_action_probs(strategy_profile: StrategyProfile, infoset_key: str) -> np.ndarray:
    """Safely fetch [prob_pass, prob_bet], defaulting to uniform 50/50 if unseen."""
    if infoset_key in strategy_profile:
        return strategy_profile[infoset_key]
    return np.array([0.5, 0.5], dtype=np.float64)


def _ev_traversal(
    cards: tuple[int, int],
    history: str,
    strategy_profile: StrategyProfile,
) -> float:
    """Exact expected value for Player 1 from a given deal and history."""
    if is_terminal(history):
        return get_payoff(cards, history)

    infoset_key = get_infoset_key(cards, history)
    probs = _get_action_probs(strategy_profile, infoset_key)
    actions = get_legal_actions(history)

    ev = 0.0
    for i, action in enumerate(actions):
        if probs[i] > 0.0:
            ev += probs[i] * _ev_traversal(cards, history + action, strategy_profile)
    return ev


def compute_ev(strategy_profile: StrategyProfile) -> float:
    """Return Player 1's exact Expected Value across all 6 equally likely deals."""
    total_ev = sum(_ev_traversal(cards, "", strategy_profile) for cards in ALL_DEALS)
    return total_ev / len(ALL_DEALS)


def _best_response_value(
    br_player: int,
    my_card: int,
    history: str,
    opp_weights: dict[int, float],
    strategy_profile: StrategyProfile,
) -> float:
    """
    Exact information-constrained Best Response value for `br_player`
    holding `my_card`, given belief weights over the opponent's possible cards.
    """
    if is_terminal(history):
        total = 0.0
        for opp_card, weight in opp_weights.items():
            if weight == 0.0:
                continue
            cards = (my_card, opp_card) if br_player == 0 else (opp_card, my_card)
            p1_payoff = get_payoff(cards, history)
            payoff = p1_payoff if br_player == 0 else -p1_payoff
            total += weight * payoff
        return total

    current_player = get_current_player(history)
    actions = get_legal_actions(history)

    if current_player == br_player:
        # Nemesis must pick ONE single best action for this infoset across all hidden cards
        return max(
            _best_response_value(
                br_player, my_card, history + action, opp_weights, strategy_profile
            )
            for action in actions
        )

    # Opponent's turn: split each hidden card's weight according to their fixed strategy
    total = 0.0
    for i, action in enumerate(actions):
        next_opp_weights = {}
        for opp_card, weight in opp_weights.items():
            opp_key = f"{CARD_NAMES[opp_card]}{history}"
            opp_probs = _get_action_probs(strategy_profile, opp_key)
            next_opp_weights[opp_card] = weight * opp_probs[i]

        total += _best_response_value(
            br_player, my_card, history + action, next_opp_weights, strategy_profile
        )
    return total


def compute_best_response_ev(br_player: int, strategy_profile: StrategyProfile) -> float:
    """Return the maximum EV `br_player` can achieve against the opponent's fixed strategy."""
    total = 0.0
    for my_card in DECK:
        # Given my_card, each of the 2 remaining opponent cards is equally likely (1/2)
        initial_opp_weights = {c: 0.5 for c in DECK if c != my_card}
        total += _best_response_value(
            br_player, my_card, "", initial_opp_weights, strategy_profile
        )
    return total / len(DECK)


def compute_exploitability(strategy_profile: StrategyProfile) -> float:
    """
    Return exact Exploitability = (BR_P1 + BR_P2) / 2.
    Equals 0.0 at a true Nash Equilibrium and > 0 for any exploitable strategy.
    """
    br_p1 = compute_best_response_ev(0, strategy_profile)
    br_p2 = compute_best_response_ev(1, strategy_profile)
    return (br_p1 + br_p2) / 2.0