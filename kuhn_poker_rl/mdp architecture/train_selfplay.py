# train_selfplay_qlearning.py
from __future__ import annotations

from kuhn_core import train_q_learning_selfplay, print_learned_strategy

if __name__ == "__main__":
    Q1, Q2 = train_q_learning_selfplay(
        num_episodes=200_000,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    print("\n================ SELF-PLAY: PLAYER 1 STRATEGY ================")
    print_learned_strategy(Q1)

    print("\n================ SELF-PLAY: PLAYER 2 STRATEGY ================")
    print_learned_strategy(Q2)
