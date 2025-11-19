# train_vs_aggressive.py
from __future__ import annotations

from kuhn_core import train_q_learning_vs_opponent, print_learned_strategy
from opponents import aggressive_opponent

if __name__ == "__main__":
    Q = train_q_learning_vs_opponent(
        opponent_policy=aggressive_opponent,
        num_episodes=200_000,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    print("\n================ Q-LEARNING VS AGGRESSIVE OPPONENT ================")
    print_learned_strategy(Q)
