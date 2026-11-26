import numpy as np
from leduc.game import (
    ALL_DEALS,
    apply_action,
    get_current_player,
    get_infoset_key,
    get_legal_actions,
    get_payoff,
    is_terminal,
)


class Node:
    """Stores cumulative regrets and strategy sums for a single Leduc infoset."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_average_strategy(self) -> np.ndarray:
        """Return the normalized average strategy (converges to Nash Equilibrium)."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.num_actions, 1.0 / self.num_actions)


class LeducCFRPlus:
    """
    CFR+ solver for Leduc Hold'em (288 infosets):
    1. Alternating player updates (P1 then P2)
    2. Regret-Matching+ (floor cumulative regrets at 0 once per full pass)
    3. Linear iteration weighting on the average strategy
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.iteration: int = 0

    def get_node(self, infoset_key: str, num_actions: int) -> Node:
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(num_actions=num_actions)
        return self.nodes[infoset_key]

    def _get_current_strategy(self, node: Node) -> np.ndarray:
        """Compute current strategy via Regret Matching without mutating sums."""
        positive_regrets = np.maximum(node.regret_sum, 0.0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        return np.full(node.num_actions, 1.0 / node.num_actions)

    def cfr(
        self,
        cards: tuple[int, int, int],
        history: str,
        p0: float,
        p1: float,
        chance_prob: float,
        update_player: int,
        iteration: int,
        iter_regrets: dict[str, np.ndarray],
    ) -> float:
        """Recursive alternating CFR+ traversal weighted by deal probability."""
        player = get_current_player(history)

        if is_terminal(history):
            p1_payoff = get_payoff(cards, history)
            return p1_payoff if player == 0 else -p1_payoff

        actions = get_legal_actions(history)
        infoset_key = get_infoset_key(cards, history)
        node = self.get_node(infoset_key, len(actions))
        strategy = self._get_current_strategy(node)

        action_utils = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            next_history = apply_action(history, action)
            next_player = get_current_player(next_history)
            # Sign flips only if the turn changes (R1->R2 transition can keep P1 acting!)
            sign = 1.0 if next_player == player else -1.0

            if player == 0:
                action_utils[i] = sign * self.cfr(
                    cards, next_history, p0 * strategy[i], p1, chance_prob,
                    update_player, iteration, iter_regrets
                )
            else:
                action_utils[i] = sign * self.cfr(
                    cards, next_history, p0, p1 * strategy[i], chance_prob,
                    update_player, iteration, iter_regrets
                )

        node_util = float(np.dot(strategy, action_utils))

        if player == update_player:
            reach_prob = p0 if player == 0 else p1
            opponent_reach = p1 if player == 0 else p0

            # Linear iteration weighting on average strategy
            node.strategy_sum += iteration * chance_prob * reach_prob * strategy

            # Buffer counterfactual regrets weighted by chance_prob * opponent_reach
            if infoset_key not in iter_regrets:
                iter_regrets[infoset_key] = np.zeros(len(actions), dtype=np.float64)
            for i in range(len(actions)):
                regret = action_utils[i] - node_util
                iter_regrets[infoset_key][i] += chance_prob * opponent_reach * regret

        return node_util

    def _apply_regret_updates(self, iter_regrets: dict[str, np.ndarray]) -> None:
        """Apply buffered regrets and floor negative values at 0 (Regret-Matching+)."""
        for infoset_key, regret_delta in iter_regrets.items():
            node = self.nodes[infoset_key]
            node.regret_sum = np.maximum(0.0, node.regret_sum + regret_delta)

    def step(self) -> float:
        """Run one alternating CFR+ iteration across all 24 weighted deals."""
        self.iteration += 1
        p1_ev = 0.0

        # Pass 1: Update Player 1
        p1_regrets: dict[str, np.ndarray] = {}
        for cards, prob in ALL_DEALS:
            p1_ev += prob * self.cfr(
                cards, "", 1.0, 1.0, prob, update_player=0,
                iteration=self.iteration, iter_regrets=p1_regrets
            )
        self._apply_regret_updates(p1_regrets)

        # Pass 2: Update Player 2 against Player 1's fresh strategy
        p2_regrets: dict[str, np.ndarray] = {}
        for cards, prob in ALL_DEALS:
            self.cfr(
                cards, "", 1.0, 1.0, prob, update_player=1,
                iteration=self.iteration, iter_regrets=p2_regrets
            )
        self._apply_regret_updates(p2_regrets)

        return p1_ev

    def train(self, iterations: int) -> float:
        """Train for N iterations and return the average Player 1 game value."""
        ev_sum = 0.0
        for _ in range(iterations):
            ev_sum += self.step()
        return ev_sum / iterations

    def get_average_strategy_profile(self) -> dict[str, np.ndarray]:
        """Return {infoset_key: action_probs} for all visited infosets."""
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}