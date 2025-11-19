# train_vs_mixed.py
from __future__ import annotations

from kuhn_core import train_q_learning_vs_opponent, print_learned_strategy
from opponents import mixed_opponent_factory

if __name__ == "__main__":
    mixed_opp = mixed_opponent_factory(
        p_aggressive=0.4,
        p_passive=0.3,
        p_random=0.3,
    )

    Q = train_q_learning_vs_opponent(
        opponent_policy=mixed_opp,
        num_episodes=200_000,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    print("\n================ Q-LEARNING VS MIXED-STYLE OPPONENT ================")
    print_learned_strategy(Q)
