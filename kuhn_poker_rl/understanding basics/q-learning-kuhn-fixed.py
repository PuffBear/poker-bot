#!/usr/bin/env python3
"""
Q-learning agent for Kuhn Poker (Player 1 vs fixed opponent).

- 3 cards: J=0, Q=1, K=2
- Actions: 0 = PASS (check/fold), 1 = BET (bet/call)
- State (for the agent): (private_card, history_string)
  where history ∈ {"", "p", "b", "pb"} when it's P1's turn.

We train a Q-table:
  Q[(card, history)][action] -> value
"""

import random
from collections import defaultdict

# ---------------------------
# Game definitions
# ---------------------------

ACTIONS = [0, 1]  # 0 = PASS, 1 = BET
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}


def deal_cards():
    """Deal two distinct cards (0,1,2) to P1 and P2."""
    cards = [0, 1, 2]
    random.shuffle(cards)
    return cards[0], cards[1]  # (p1_card, p2_card)


def is_terminal(history: str) -> bool:
    """Check if action history is terminal."""
    return history in ("pp", "bp", "bb", "pbp", "pbb")


def payoff(p1_card: int, p2_card: int, history: str) -> int:
    """
    Payoff to Player 1 (net chips, zero-sum game).

    Both players ante 1 at start (pot=2).
    - 'pp'  : showdown, pot=2  -> winner gets +1, loser -1
    - 'bp'  : P1 bet, P2 folds -> P1 wins pot=3 -> +1 (net)
    - 'bb'  : P1 bet, P2 calls -> pot=4 -> winner +2, loser -2
    - 'pbp' : P1 checked, P2 bet, P1 folds -> P1 loses 1
    - 'pbb' : P1 checked, P2 bet, P1 calls -> pot=4 -> winner +2, loser -2
    """
    if history == "pp":
        return 1 if p1_card > p2_card else -1
    elif history == "bp":   # P2 folds to P1's bet
        return 1
    elif history == "bb":   # P2 calls P1's bet
        return 2 if p1_card > p2_card else -2
    elif history == "pbp":  # P1 folds to P2's bet
        return -1
    elif history == "pbb":  # P1 calls P2's bet
        return 2 if p1_card > p2_card else -2
    else:
        raise ValueError(f"Invalid terminal history: {history}")


# ---------------------------
# Opponent policy (Player 2)
# ---------------------------

def opponent_policy(p2_card: int, history: str) -> int:
    """
    Fixed heuristic opponent for Player 2.

    Input:
      p2_card: 0=J,1=Q,2=K
      history: must be "p" (after P1 check) or "b" (facing P1 bet).

    Output:
      action: 0 = PASS (check/fold), 1 = BET (bet/call)
    """
    r = random.random()

    if history == "p":
        # P1 checked; P2 decides to check or bet.
        if p2_card == 2:      # K: usually bet
            return 1
        elif p2_card == 1:    # Q: sometimes bet
            return 1 if r < 0.2 else 0
        else:                 # J: occasionally bluff
            return 1 if r < 0.3 else 0

    elif history == "b":
        # P1 bet; P2 decides to fold or call.
        if p2_card == 2:      # K: always call
            return 1
        elif p2_card == 1:    # Q: mix call/fold
            return 1 if r < 0.5 else 0
        else:                 # J: rarely bluff-calls
            return 1 if r < 0.1 else 0

    else:
        raise ValueError(f"Opponent should not act at history='{history}'")


# ---------------------------
# Q-learning helpers
# ---------------------------

def epsilon_greedy(card: int, history: str, Q, epsilon: float) -> int:
    """
    ε-greedy action selection for the agent (Player 1).

    State is (card, history).
    """
    state = (card, history)
    if random.random() < epsilon:
        return random.choice(ACTIONS)  # explore

    q_values = Q[state]
    max_q = max(q_values)
    # break ties randomly
    best_actions = [a for a, q in enumerate(q_values) if q == max_q]
    return random.choice(best_actions)


def train_q_learning(
    num_episodes: int = 200_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
):
    """
    Train a Q-learning agent as Player 1 vs fixed Player 2 policy.

    Returns:
      Q: dict mapping (card, history) -> [Q_pass, Q_bet]
    """
    # Q[(card, history)] = [Q_pass, Q_bet]
    Q = defaultdict(lambda: [0.0, 0.0])

    epsilon = epsilon_start

    for episode in range(1, num_episodes + 1):
        # Deal cards
        p1_card, p2_card = deal_cards()
        history = ""
        current_player = 0  # 0 = P1 (agent), 1 = P2 (opponent)

        # For bootstrapping previous agent state
        prev_state = None
        prev_action = None

        while True:
            if current_player == 0:
                # -------- Agent (Player 1) turn --------
                state = (p1_card, history)
                action = epsilon_greedy(p1_card, history, Q, epsilon)
                action_char = "p" if action == 0 else "b"
                history = history + action_char

                # Check if game ended immediately after agent's action
                if is_terminal(history):
                    r = payoff(p1_card, p2_card, history)

                    # Update Q for this (state, action) with terminal reward
                    q_values = Q[state]
                    q_values[action] += alpha * (r - q_values[action])

                    # Also bootstrap previous agent state (if any)
                    if prev_state is not None:
                        # no immediate reward at prev step, only future value
                        next_v = max(Q[state])
                        q_prev = Q[prev_state]
                        q_prev[prev_action] += alpha * (
                            gamma * next_v - q_prev[prev_action]
                        )
                    break  # episode done

                else:
                    # Not terminal: pass turn to Player 2
                    prev_state = state
                    prev_action = action
                    current_player = 1

            else:
                # -------- Opponent (Player 2) turn --------
                # P2 acts using fixed policy
                action2 = opponent_policy(p2_card, history)
                action_char2 = "p" if action2 == 0 else "b"
                history = history + action_char2

                if is_terminal(history):
                    # Game ends after opponent's action
                    r = payoff(p1_card, p2_card, history)

                    # Last agent state was prev_state/prev_action
                    if prev_state is not None:
                        q_prev = Q[prev_state]
                        q_prev[prev_action] += alpha * (r - q_prev[prev_action])
                    break
                else:
                    # Not terminal: back to agent (only possible at 'pb')
                    current_player = 0

        # Linear epsilon decay
        frac = episode / num_episodes
        epsilon = max(epsilon_end, epsilon_start + (epsilon_end - epsilon_start) * frac)

        # Optional: simple progress printing
        if episode % 50_000 == 0:
            print(f"Episode {episode}/{num_episodes}, epsilon={epsilon:.3f}")

    return Q


# ---------------------------
# Inspect learned strategy
# ---------------------------

def print_learned_strategy(Q):
    """
    Print learned policy for Player 1 at its decision points:
      - history ""  : first action (check/bet)
      - history "pb": facing a bet after checking (fold/call)
    """
    print("\n==========================================")
    print(" Learned Q-learning strategy for Player 1")
    print("==========================================\n")

    for card in [0, 1, 2]:
        card_name = CARD_NAMES[card]
        print(f"{card_name}:")

        for history, desc, a0, a1 in [
            ("",  "First action",                   "Check", "Bet"),
            ("pb", "Facing bet after checking",     "Fold",  "Call"),
        ]:
            state = (card, history)
            q_pass, q_bet = Q[state]  # [Q(PASS), Q(BET)]

            # Greedy policy from Q-values
            if q_pass == q_bet:
                pi_pass = pi_bet = 0.5
            else:
                best_action = 0 if q_pass > q_bet else 1
                pi_pass = 1.0 if best_action == 0 else 0.0
                pi_bet = 1.0 - pi_pass

            print(
                f"  {desc:28} -> "
                f"{a0}: π={pi_pass:.3f}, Q={q_pass:+.3f}   "
                f"{a1}: π={pi_bet:.3f}, Q={q_bet:+.3f}"
            )
        print()


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    Q = train_q_learning(
        num_episodes=200_000,
        alpha=0.999,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )
    print_learned_strategy(Q)
