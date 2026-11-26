import numpy as np
from kuhn.game import (
    ALL_DEALS,
    get_current_player,
    get_infoset_key,
    get_legal_actions,
    get_payoff,
    is_terminal,
)


class Node:
    """Stores cumulative regrets and strategy sums for a single infoset."""

    def __init__(self, num_actions: int = 2):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self, realization_weight: float = 0.0) -> np.ndarray:
        """Compute current strategy via Regret Matching and accumulate it."""
        positive_regrets = np.maximum(self.regret_sum, 0.0)
        normalizing_sum = positive_regrets.sum()

        if normalizing_sum > 0:
            strategy = positive_regrets / normalizing_sum
        else:
            strategy = np.full(self.num_actions, 1.0 / self.num_actions)

        # Weight the average strategy by how likely WE were to reach this node
        self.strategy_sum += realization_weight * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        """Return the normalized average strategy (converges to Nash Equilibrium)."""
        normalizing_sum = self.strategy_sum.sum()
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        return np.full(self.num_actions, 1.0 / self.num_actions)


class VanillaCFR:
    """Vanilla Counterfactual Regret Minimization solver for Kuhn Poker."""

    def __init__(self):
        self.nodes: dict[str, Node] = {}

    def get_node(self, infoset_key: str) -> Node:
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(num_actions=2)
        return self.nodes[infoset_key]

    def cfr(
        self,
        cards: tuple[int, int],
        history: str,
        p0: float,
        p1: float,
    ) -> float:
        """
        Recursive CFR traversal.
        Returns expected utility from the perspective of the current player.
        """
        player = get_current_player(history)

        # Base case: hand is over
        if is_terminal(history):
            p1_payoff = get_payoff(cards, history)
            return p1_payoff if player == 0 else -p1_payoff

        infoset_key = get_infoset_key(cards, history)
        node = self.get_node(infoset_key)
        actions = get_legal_actions(history)

        # 1. Get current strategy using Regret Matching
        reach_prob = p0 if player == 0 else p1
        strategy = node.get_strategy(reach_prob)

        # 2. Recursively compute utility for each action
        action_utils = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            next_history = history + action
            if player == 0:
                action_utils[i] = -self.cfr(cards, next_history, p0 * strategy[i], p1)
            else:
                action_utils[i] = -self.cfr(cards, next_history, p0, p1 * strategy[i])

        # 3. Expected value of this node under current strategy
        node_util = float(np.dot(strategy, action_utils))

        # 4. Update counterfactual regrets (weighted by OPPONENT's reach prob)
        opponent_reach = p1 if player == 0 else p0
        for i in range(len(actions)):
            regret = action_utils[i] - node_util
            node.regret_sum[i] += opponent_reach * regret

        return node_util

    def step(self) -> float:
        """Run one full CFR iteration across all 6 equally likely deals."""
        total_util = 0.0
        for cards in ALL_DEALS:
            total_util += self.cfr(cards, "", 1.0, 1.0)
        return total_util / len(ALL_DEALS)

    def train(self, iterations: int) -> float:
        """Train for N iterations and return the average Player 1 game value."""
        util_sum = 0.0
        for _ in range(iterations):
            util_sum += self.step()
        return util_sum / iterations

    def get_average_strategy_profile(self) -> dict[str, np.ndarray]:
        """Return {infoset_key: [prob_pass, prob_bet]} for all visited infosets."""
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}

class CFRPlus:
    """
    CFR+ solver for Kuhn Poker:
    1. Alternating player updates (P1 then P2)
    2. Regret-matching+ (floor cumulative regrets at 0 once per full pass)
    3. Linear iteration weighting on the average strategy
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.iteration: int = 0

    def get_node(self, infoset_key: str) -> Node:
        if infoset_key not in self.nodes:
            self.nodes[infoset_key] = Node(num_actions=2)
        return self.nodes[infoset_key]

    def _get_current_strategy(self, node: Node) -> np.ndarray:
        """Compute current strategy via Regret Matching without mutating sums."""
        positive_regrets = np.maximum(node.regret_sum, 0.0)
        normalizing_sum = positive_regrets.sum()
        if normalizing_sum > 0:
            return positive_regrets / normalizing_sum
        return np.full(node.num_actions, 1.0 / node.num_actions)

    def cfr(
        self,
        cards: tuple[int, int],
        history: str,
        p0: float,
        p1: float,
        update_player: int,
        iteration: int,
        iter_regrets: dict[str, np.ndarray],
    ) -> float:
        """Recursive alternating CFR+ traversal accumulating regrets into `iter_regrets`."""
        player = get_current_player(history)

        if is_terminal(history):
            p1_payoff = get_payoff(cards, history)
            return p1_payoff if player == 0 else -p1_payoff

        infoset_key = get_infoset_key(cards, history)
        node = self.get_node(infoset_key)
        actions = get_legal_actions(history)
        strategy = self._get_current_strategy(node)

        action_utils = np.zeros(len(actions), dtype=np.float64)
        for i, action in enumerate(actions):
            next_history = history + action
            if player == 0:
                action_utils[i] = -self.cfr(
                    cards, next_history, p0 * strategy[i], p1, update_player, iteration, iter_regrets
                )
            else:
                action_utils[i] = -self.cfr(
                    cards, next_history, p0, p1 * strategy[i], update_player, iteration, iter_regrets
                )

        node_util = float(np.dot(strategy, action_utils))

        if player == update_player:
            reach_prob = p0 if player == 0 else p1
            opponent_reach = p1 if player == 0 else p0

            # Linear iteration weighting on average strategy
            node.strategy_sum += iteration * reach_prob * strategy

            # Buffer counterfactual regrets across all deals before flooring
            if infoset_key not in iter_regrets:
                iter_regrets[infoset_key] = np.zeros(len(actions), dtype=np.float64)
            for i in range(len(actions)):
                regret = action_utils[i] - node_util
                iter_regrets[infoset_key][i] += opponent_reach * regret

        return node_util

    def _apply_regret_updates(self, iter_regrets: dict[str, np.ndarray]) -> None:
        """Apply buffered regrets and floor negative values at 0 (Regret-Matching+)."""
        for infoset_key, regret_delta in iter_regrets.items():
            node = self.nodes[infoset_key]
            node.regret_sum = np.maximum(0.0, node.regret_sum + regret_delta)

    def step(self) -> float:
        """Run one alternating CFR+ iteration (update P1, then update P2)."""
        self.iteration += 1
        p1_util_sum = 0.0

        # Pass 1: Update Player 1 across all 6 deals, then floor regrets
        p1_regrets: dict[str, np.ndarray] = {}
        for cards in ALL_DEALS:
            p1_util_sum += self.cfr(
                cards, "", 1.0, 1.0, update_player=0, iteration=self.iteration, iter_regrets=p1_regrets
            )
        self._apply_regret_updates(p1_regrets)

        # Pass 2: Update Player 2 against Player 1's fresh strategy, then floor regrets
        p2_regrets: dict[str, np.ndarray] = {}
        for cards in ALL_DEALS:
            self.cfr(
                cards, "", 1.0, 1.0, update_player=1, iteration=self.iteration, iter_regrets=p2_regrets
            )
        self._apply_regret_updates(p2_regrets)

        return p1_util_sum / len(ALL_DEALS)

    def train(self, iterations: int) -> float:
        """Train for N iterations and return the average Player 1 game value."""
        util_sum = 0.0
        for _ in range(iterations):
            util_sum += self.step()
        return util_sum / iterations

    def get_average_strategy_profile(self) -> dict[str, np.ndarray]:
        """Return {infoset_key: [prob_pass, prob_bet]} for all visited infosets."""
        return {key: node.get_average_strategy() for key, node in self.nodes.items()}