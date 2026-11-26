from itertools import permutations

#Internal card representation: 0 = Jack, 1 = Queen, 2 = King
CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}
DECK = (0, 1, 2)
ALL_DEALS = list(permutations(DECK, 2)) # 6 equally likely deals

ACTIONS = ('p', 'b') # 'p' = pass (check/fold), 'b' = bet (bet/call)
TERMINAL_HISTORIES = {'pp', 'bp', 'bb', 'pbp', 'pbb'} # Round has ended

def is_terminal(history: str) -> bool:
    """Return True if the hand has ended."""
    return history in TERMINAL_HISTORIES

def get_current_player(history: str) -> int:
    """Return 0 for Player 1, 1 for Player 2."""
    return len(history) % 2

def get_legal_actions(history: str) -> tuple[str, ...]:
    """Return available actions at a non-terminal history."""
    if is_terminal(history):
        return()
    return ACTIONS

def get_infoset_key(cards: tuple[int, int], history: str) -> str:
    """Return the acting player's private card + public betting history."""
    player = get_current_player(history)
    return f"{CARD_NAMES[cards[player]]}{history}"

def get_payoff(cards: tuple[int, int], history: str) -> float:
    """
    Return Player 1's chip payoff at a terminal history.
    Player 2's payoff is always -get_payoff(cards, history).
    """
    if not is_terminal(history):
        raise ValueError(f"History '{history}' is not terminal")

    # Fold outcomes (no showdown needed)
    if history == 'bp': # P1 bet, P2 folded
        return 1.0
    if history == 'pbp': # P1 checked, P2 bet, P1 folded
        return -1.0

    # Showdown outcomes ('pp' = 1 chip, 'bb' or 'pbb' = 2 chips)
    p1_card, p2_card = cards
    pot_stake = 1.0 if history == 'pp' else 2.0
    return pot_stake if p1_card > p2_card else -pot_stake