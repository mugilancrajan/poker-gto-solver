RANK_NAMES = {0: "J", 1: "Q", 2: "K"}
RANKS = (0, 1, 2)

ROUND_END_SEQUENCES = {"cc", "rc", "crc", "rrc", "crrc"}
ANTE = 1
ROUND_BET_SIZES = (2, 4)


def _build_weighted_deals() -> tuple[tuple[tuple[int, int, int], float], ...]:
    """
    Build all 24 valid (p1_rank, p2_rank, board_rank) triples with exact
    probabilities from a 6-card deck (2 suits each of J, Q, K).
    """
    deals = []
    total_ordered_deals = 6 * 5 * 4  # 120
    for p1 in RANKS:
        for p2 in RANKS:
            for board in RANKS:
                if p1 == p2 == board:
                    continue  # Only 2 of each rank exist in the deck
                ways_p1 = 2
                ways_p2 = 1 if p2 == p1 else 2
                ways_board = 1 if (board == p1 or board == p2) else 2
                prob = (ways_p1 * ways_p2 * ways_board) / total_ordered_deals
                deals.append(((p1, p2, board), prob))
    return tuple(deals)


ALL_DEALS = _build_weighted_deals()


def is_terminal(history: str) -> bool:
    """Return True if the hand has ended (fold in either round, or Round 2 showdown)."""
    if history.endswith("f"):
        return True
    if "/" in history:
        r2 = history.split("/")[1]
        return r2 in ROUND_END_SEQUENCES
    return False


def get_current_player(history: str) -> int:
    """Return 0 for Player 1, 1 for Player 2. Player 1 always acts first in each round."""
    round_hist = history.split("/")[-1]
    return len(round_hist) % 2


def get_legal_actions(history: str) -> tuple[str, ...]:
    """Return legal actions ('f', 'c', 'r') given the current round's betting sequence."""
    if is_terminal(history):
        return ()
    round_hist = history.split("/")[-1]
    if round_hist in ("", "c"):
        return ("c", "r")
    if round_hist in ("r", "cr"):
        return ("f", "c", "r")
    if round_hist in ("rr", "crr"):
        return ("f", "c")
    raise ValueError(f"Invalid Leduc history: {history!r}")


def apply_action(history: str, action: str) -> str:
    """Append an action and automatically insert '/' when Round 1 finishes without a fold."""
    next_hist = history + action
    if "/" not in next_hist and next_hist in ROUND_END_SEQUENCES:
        return next_hist + "/"
    return next_hist


def get_infoset_key(cards: tuple[int, int, int], history: str) -> str:
    """
    Return the information set key for the current player:
      - Round 1: 'J|r'
      - Round 2: 'J|K|cc/r'
    """
    p1, p2, board = cards
    player = get_current_player(history)
    my_rank = RANK_NAMES[p1 if player == 0 else p2]
    if "/" not in history:
        return f"{my_rank}|{history}"
    return f"{my_rank}|{RANK_NAMES[board]}|{history}"


def get_payoff(cards: tuple[int, int, int], history: str) -> float:
    """Return Player 1's net chip payoff at a terminal state."""
    if not is_terminal(history):
        raise ValueError(f"Cannot compute payoff for non-terminal history: {history!r}")

    contrib = [ANTE, ANTE]
    rounds = history.split("/")

    for round_idx, r_hist in enumerate(rounds):
        bet_size = ROUND_BET_SIZES[round_idx]
        current_bet = 0
        round_contrib = [0, 0]
        player = 0

        for action in r_hist:
            if action == "c":
                round_contrib[player] = current_bet
            elif action == "r":
                current_bet += bet_size
                round_contrib[player] = current_bet
            elif action == "f":
                # Folding player forfeits their total chips committed so far
                if player == 0:
                    return -float(contrib[0] + round_contrib[0])
                return float(contrib[1] + round_contrib[1])
            player = 1 - player

        contrib[0] += round_contrib[0]
        contrib[1] += round_contrib[1]

    # Showdown: both players committed equal chips
    p1, p2, board = cards
    p1_pair = p1 == board
    p2_pair = p2 == board

    if p1_pair and not p2_pair:
        return float(contrib[1])
    if p2_pair and not p1_pair:
        return -float(contrib[0])
    if p1 > p2:
        return float(contrib[1])
    if p2 > p1:
        return -float(contrib[0])
    return 0.0  # Exact tie: split pot