# env + the Q-learning loop

# kuhn_core.py
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Tuple, Callable, List

# Type aliases
Action = int  # 0 = PASS, 1 = BET
State = Tuple[int, str]  # (card, history)
QTable = Dict[State, List[float]]
OpponentPolicy = Callable[[int, str], int]

# Constants
ACTION_PASS: Action = 0
ACTION_BET: Action = 1
ACTIONS = [ACTION_PASS, ACTION_BET]

CARD_NAMES = ["Jack", "Queen", "King"]


def deal_cards() -> Tuple[int, int]:
    """Deal two distinct cards (0,1,2) to Player 1 and Player 2."""
    cards = [0, 1, 2]
    random.shuffle(cards)
    return cards[0], cards[1]


def is_terminal(history: str) -> bool:
    """
    Terminal histories in Kuhn poker:
      'pp'   : P1 check, P2 check -> showdown
      'bp'   : P1 bet,  P2 fold   -> P1 wins
      'bb'   : P1 bet,  P2 call   -> showdown
      'pbb'  : P1 check, P2 bet, P1 call -> showdown
      'pbp'  : P1 check, P2 bet, P1 fold -> P2 wins
    """
    return history in ("pp", "bp", "bb", "pbb", "pbp")


def get_payoff(p1_card: int, p2_card: int, history: str) -> int:
    """
    Payoff for Player 1, following your CFR convention.
    Positive = good for Player 1, negative = good for Player 2.
    """
    # Showdown cases
    if history in ("pp", "bb", "pbb"):
        return 2 if p1_card > p2_card else -2
    # P2 folds to P1 bet
    if history == "bp":
        return 1
    # P1 folds to P2 bet
    if history == "pbp":
        return -1
    raise ValueError(f"Invalid terminal history: {history}")


def epsilon_greedy(state: State, Q: QTable, epsilon: float) -> Action:
    """ε-greedy action selection on a tabular Q-table."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_vals = Q[state]
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    return random.choice(best_actions)


def train_q_learning_vs_opponent(
    opponent_policy: OpponentPolicy,
    num_episodes: int = 200_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> QTable:
    """
    Train a Q-learning agent as Player 1 versus a fixed opponent (Player 2).

    The agent controls Player 1's actions at:
      - history ""   (first action)
      - history "pb" (facing a bet after checking)

    opponent_policy(p2_card, history) chooses Player 2's action at:
      - history "p" (after P1 checks)
      - history "b" (facing P1's bet)
    """
    Q: QTable = defaultdict(lambda: [0.0, 0.0])

    for ep in range(1, num_episodes + 1):
        p1_card, p2_card = deal_cards()
        history = ""

        # Linear epsilon decay
        frac = ep / num_episodes
        epsilon = max(epsilon_end, epsilon_start + (epsilon_end - epsilon_start) * frac)

        while True:
            # ----- Player 1 (learner) acts -----
            state = (p1_card, history)
            action = epsilon_greedy(state, Q, epsilon)
            action_char = "p" if action == ACTION_PASS else "b"
            history = history + action_char

            # Check terminal after P1 action
            if is_terminal(history):
                r = get_payoff(p1_card, p2_card, history)
                old = Q[state][action]
                Q[state][action] = old + alpha * (r - old)
                break

            # ----- Player 2 (opponent) acts -----
            opp_action = opponent_policy(p2_card, history)
            opp_char = "p" if opp_action == ACTION_PASS else "b"
            history = history + opp_char

            if is_terminal(history):
                r = get_payoff(p1_card, p2_card, history)
                old = Q[state][action]
                Q[state][action] = old + alpha * (r - old)
                break

            # Non-terminal: it's P1's turn again at a new state
            next_state = (p1_card, history)
            max_next_q = max(Q[next_state])
            target = gamma * max_next_q  # intermediate reward = 0
            old = Q[state][action]
            Q[state][action] = old + alpha * (target - old)
            # loop continues with updated history

        if ep % (max(1, num_episodes // 10)) == 0:
            print(f"[vs-opponent] Episode {ep}/{num_episodes}, epsilon={epsilon:.3f}")

    return Q


def train_q_learning_selfplay(
    num_episodes: int = 200_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
):
    """
    Self-play Q-learning with two independent agents:
      - Q1 controls Player 1
      - Q2 controls Player 2

    Both use ε-greedy on their own Q-tables.
    Updates are done from each player's perspective with
    +reward for themselves and -reward for the opponent.
    """
    Q1: QTable = defaultdict(lambda: [0.0, 0.0])
    Q2: QTable = defaultdict(lambda: [0.0, 0.0])

    for ep in range(1, num_episodes + 1):
        p1_card, p2_card = deal_cards()
        history = ""
        current_player = 0  # 0 = P1, 1 = P2

        # For TD updates between SAME player's decisions
        last_state = {0: None, 1: None}
        last_action = {0: None, 1: None}

        # Linear epsilon decay
        frac = ep / num_episodes
        epsilon = max(epsilon_end, epsilon_start + (epsilon_end - epsilon_start) * frac)

        while True:
            # Choose which Q-table and card to use
            if current_player == 0:
                card = p1_card
                Q = Q1
            else:
                card = p2_card
                Q = Q2

            state: State = (card, history)

            # TD update for this player's previous decision using current state
            if last_state[current_player] is not None:
                prev_s = last_state[current_player]
                prev_a = last_action[current_player]
                max_next = max(Q[state])
                Q_prev = Q1 if current_player == 0 else Q2
                old = Q_prev[prev_s][prev_a]
                Q_prev[prev_s][prev_a] = old + alpha * (0.0 + gamma * max_next - old)

            # Choose action ε-greedily
            action = epsilon_greedy(state, Q, epsilon)
            last_state[current_player] = state
            last_action[current_player] = action

            action_char = "p" if action == ACTION_PASS else "b"
            history = history + action_char

            if is_terminal(history):
                # Terminal reward from P1's perspective
                r_p1 = get_payoff(p1_card, p2_card, history)

                # Reward for current player
                r_cur = r_p1 if current_player == 0 else -r_p1
                Q_cur = Q1 if current_player == 0 else Q2
                s_last = last_state[current_player]
                a_last = last_action[current_player]
                old = Q_cur[s_last][a_last]
                Q_cur[s_last][a_last] = old + alpha * (r_cur - old)

                # Also update the opponent's last decision (if any)
                other = 1 - current_player
                if last_state[other] is not None:
                    Q_other = Q1 if other == 0 else Q2
                    r_other = -r_cur
                    s_prev_o = last_state[other]
                    a_prev_o = last_action[other]
                    old_o = Q_other[s_prev_o][a_prev_o]
                    Q_other[s_prev_o][a_prev_o] = old_o + alpha * (r_other - old_o)
                break

            # Switch player
            current_player = 1 - current_player

        if ep % (max(1, num_episodes // 10)) == 0:
            print(f"[self-play] Episode {ep}/{num_episodes}, epsilon={epsilon:.3f}")

    return Q1, Q2


def q_to_policy(Q: QTable, state: State) -> Tuple[float, float]:
    """
    Convert Q-values at a state into a stochastic policy:
    - If all Q-values are equal, returns (0.5, 0.5)
    - Otherwise puts equal probability mass over argmax actions
    """
    q_vals = Q[state]
    if q_vals[0] == q_vals[1]:
        return 0.5, 0.5
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    probs = [0.0, 0.0]
    for a in best_actions:
        probs[a] = 1.0 / len(best_actions)
    return probs[0], probs[1]


def print_learned_strategy(Q: QTable) -> None:
    """
    Pretty-print the learned policy.

    From Player 1 perspective, the real decision points are:
      - ""  : first action
      - "pb": facing a bet after checking

    But for symmetry / analysis we also show "p" and "b" if present
    (those correspond to acting as the second player).
    """
    print("\n" + "=" * 58)
    print("LEARNED STRATEGY FROM Q-TABLE")
    print("=" * 58)

    decision_states = [
        ("", "First action", "Check/Fold", "Bet"),
        ("p", "After opponent checks", "Check", "Bet"),
        ("b", "Facing a bet", "Fold", "Call"),
        ("pb", "Facing a bet after checking", "Fold", "Call"),
    ]

    for card in range(3):
        print(f"\n{CARD_NAMES[card]}:")
        for history, desc, a0_name, a1_name in decision_states:
            state = (card, history)
            if state in Q:
                p0, p1 = q_to_policy(Q, state)
                print(f"  {desc:28} -> {a0_name}: {p0:.3f}, {a1_name}: {p1:.3f}")
            else:
                print(f"  {desc:28} -> (no data, default 0.5/0.5)")


if __name__ == "__main__":
    # Quick smoke test vs random opponent
    def _rand_opp(card: int, history: str) -> int:
        return random.choice(ACTIONS)

    Q_test = train_q_learning_vs_opponent(_rand_opp, num_episodes=5000)
    print_learned_strategy(Q_test)
