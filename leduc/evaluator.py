import numpy as np
from leduc.game import (
    ALL_DEALS,
    RANK_NAMES,
    RANKS,
    apply_action,
    get_current_player,
    get_infoset_key,
    get_legal_actions,
    get_payoff,
    is_terminal,
)

StrategyProfile = dict[str, np.ndarray]


def _get_action_probs(
    strategy_profile: StrategyProfile, infoset_key: str, num_actions: int
) -> np.ndarray:
    """Safely fetch action probabilities, defaulting to uniform if unseen."""
    if infoset_key in strategy_profile:
        return strategy_profile[infoset_key]
    return np.full(num_actions, 1.0 / num_actions, dtype=np.float64)


def _ev_traversal(
    cards: tuple[int, int, int],
    history: str,
    strategy_profile: StrategyProfile,
) -> float:
    """Exact expected value for Player 1 from a given deal and history."""
    if is_terminal(history):
        return get_payoff(cards, history)

    actions = get_legal_actions(history)
    infoset_key = get_infoset_key(cards, history)
    probs = _get_action_probs(strategy_profile, infoset_key, len(actions))

    ev = 0.0
    for i, action in enumerate(actions):
        if probs[i] > 0.0:
            next_hist = apply_action(history, action)
            ev += probs[i] * _ev_traversal(cards, next_hist, strategy_profile)
    return ev


def compute_ev(strategy_profile: StrategyProfile) -> float:
    """Return Player 1's exact Expected Value across all 24 weighted deals."""
    return sum(prob * _ev_traversal(cards, "", strategy_profile) for cards, prob in ALL_DEALS)


def _best_response_value(
    br_player: int,
    my_card: int,
    board: int | None,
    history: str,
    opp_weights: dict[int, float],
    strategy_profile: StrategyProfile,
) -> float:
    """
    Exact information-constrained Best Response value for `br_player`
    holding `my_card` (and public `board` in Round 2).
    """
    if is_terminal(history):
        total = 0.0
        b = board if board is not None else 0
        for opp_card, weight in opp_weights.items():
            if weight == 0.0:
                continue
            cards = (my_card, opp_card, b) if br_player == 0 else (opp_card, my_card, b)
            p1_payoff = get_payoff(cards, history)
            payoff = p1_payoff if br_player == 0 else -p1_payoff
            total += weight * payoff
        return total

    # Chance node: Round 1 just ended without a fold -> deal the public board card
    if board is None and "/" in history:
        total = 0.0
        for board_rank in RANKS:
            board_opp_weights = {}
            for opp_card, weight in opp_weights.items():
                if weight == 0.0:
                    continue
                remaining = 2 - (1 if my_card == board_rank else 0) - (1 if opp_card == board_rank else 0)
                if remaining > 0:
                    board_opp_weights[opp_card] = weight * (remaining / 4.0)
            if board_opp_weights:
                total += _best_response_value(
                    br_player, my_card, board_rank, history, board_opp_weights, strategy_profile
                )
        return total

    current_player = get_current_player(history)
    actions = get_legal_actions(history)

    if current_player == br_player:
        # Nemesis picks ONE single best action for this infoset across all hidden opp cards
        return max(
            _best_response_value(
                br_player, my_card, board, apply_action(history, action), opp_weights, strategy_profile
            )
            for action in actions
        )

    # Opponent's turn: split each hidden card's weight according to their fixed strategy
    total = 0.0
    for i, action in enumerate(actions):
        next_opp_weights = {}
        for opp_card, weight in opp_weights.items():
            if weight == 0.0:
                continue
            if board is None:
                opp_key = f"{RANK_NAMES[opp_card]}|{history}"
            else:
                opp_key = f"{RANK_NAMES[opp_card]}|{RANK_NAMES[board]}|{history}"
            opp_probs = _get_action_probs(strategy_profile, opp_key, len(actions))
            next_opp_weights[opp_card] = weight * opp_probs[i]

        total += _best_response_value(
            br_player, my_card, board, apply_action(history, action), next_opp_weights, strategy_profile
        )
    return total


def compute_best_response_ev(br_player: int, strategy_profile: StrategyProfile) -> float:
    """Return the maximum EV `br_player` can achieve against the opponent's fixed strategy."""
    total = 0.0
    for my_card in RANKS:
        # Given my_card, 5 cards remain: 1 of my_card (1/5) and 2 of each other rank (2/5)
        initial_opp_weights = {
            c: (1.0 / 5.0 if c == my_card else 2.0 / 5.0) for c in RANKS
        }
        total += _best_response_value(
            br_player, my_card, None, "", initial_opp_weights, strategy_profile
        )
    return total / len(RANKS)


def compute_exploitability(strategy_profile: StrategyProfile) -> float:
    """
    Return exact Exploitability = (BR_P1 + BR_P2) / 2.
    Equals 0.0 at a true Nash Equilibrium and > 0 for any exploitable strategy.
    """
    br_p1 = compute_best_response_ev(0, strategy_profile)
    br_p2 = compute_best_response_ev(1, strategy_profile)
    return (br_p1 + br_p2) / 2.0