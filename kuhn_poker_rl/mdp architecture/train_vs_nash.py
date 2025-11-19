# train_vs_nash.py
from __future__ import annotations

from kuhn_core import train_q_learning_vs_opponent, print_learned_strategy
from opponents import nash_opponent_factory

# IMPORTANT:
# Make sure your pure CFR code (the one you pasted earlier) is in
# a file named "kuhn_poker_cfr.py" with a class `CFRTrainer`
# that has .get_average_strategy() and .load_strategy().
from kuhn_poker_cfr import CFRTrainer   # adjust filename if needed

if __name__ == "__main__":
    # Either train CFR here OR load a pre-trained strategy
    cfr = CFRTrainer()
    # If you've already trained & saved:
    cfr.load_strategy("kuhn_poker_strategy.pkl")

    nash_opp = nash_opponent_factory(cfr)

    Q = train_q_learning_vs_opponent(
        opponent_policy=nash_opp,
        num_episodes=200_000,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    print("\n================ Q-LEARNING VS CFR/NASH OPPONENT ================")
    print_learned_strategy(Q)
