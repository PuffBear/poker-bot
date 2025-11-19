# module to define the opponents

# opponents.py
from __future__ import annotations

import random
from typing import Callable

from kuhn_core import ACTION_PASS, ACTION_BET, ACTIONS

OpponentPolicy = Callable[[int, str], int]  # (card, history) -> action


# 1) Random opponent -------------------------------------------------
def random_opponent(card: int, history: str) -> int:
    """Completely random: chooses PASS / BET uniformly."""
    return random.choice(ACTIONS)


# 2) Aggressive opponent ---------------------------------------------
def aggressive_opponent(card: int, history: str) -> int:
    """
    Aggro:
      - After check ('p'): always bet.
      - Facing a bet ('b'): always call.
    """
    if history == "p":
        # P1 checked, P2 acts -> bet
        return ACTION_BET
    elif history == "b":
        # Facing P1 bet -> call
        return ACTION_BET
    else:
        # Shouldn't be called at other histories in this setup
        return ACTION_BET


# 3) Passive opponent -------------------------------------------------
def passive_opponent(card: int, history: str) -> int:
    """
    Passive:
      - After check ('p'): always check back.
      - Facing a bet ('b'): always call (never bluff-raise / fold).
    """
    if history == "p":
        # P1 checked, P2 checks back
        return ACTION_PASS
    elif history == "b":
        # Facing P1 bet -> call
        return ACTION_BET
    else:
        return ACTION_PASS


# 4) Mixed-style opponent --------------------------------------------
def mixed_opponent_factory(
    p_aggressive: float = 0.4,
    p_passive: float = 0.3,
    p_random: float = 0.3,
) -> OpponentPolicy:
    """
    Returns an opponent policy that mixes AGGRO / PASSIVE / RANDOM.
    Probabilities should sum to 1.0 (we'll normalize just in case).
    """
    total = p_aggressive + p_passive + p_random
    p_aggressive /= total
    p_passive /= total
    p_random /= total

    def policy(card: int, history: str) -> int:
        r = random.random()
        if r < p_aggressive:
            return aggressive_opponent(card, history)
        elif r < p_aggressive + p_passive:
            return passive_opponent(card, history)
        else:
            return random_opponent(card, history)

    return policy


# 5) Nash / CFR-based opponent ---------------------------------------
def nash_opponent_factory(cfr_trainer) -> OpponentPolicy:
    """
    Wraps your CFRTrainer into an opponent policy that plays
    according to its average strategy (approximate Nash).

    Assumes CFRTrainer has:
      - get_average_strategy(info_set_key: str) -> np.ndarray of shape (2,)
    where info_set_key = f"{card}:{history}" with card in {0,1,2}.
    """

    def policy(card: int, history: str) -> int:
        info_key = f"{card}:{history}"
        strat = cfr_trainer.get_average_strategy(info_key)  # [p_pass, p_bet]
        p_pass = float(strat[0])
        r = random.random()
        if r < p_pass:
            return ACTION_PASS
        else:
            return ACTION_BET

    return policy
