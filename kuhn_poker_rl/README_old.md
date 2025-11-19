# Kuhn Poker CFR (Counterfactual Regret Minimization) Bot

## Overview

This project implements a **Counterfactual Regret Minimization (CFR)** algorithm to solve Kuhn Poker and find the Nash Equilibrium strategy. It includes:

1. **CFR Training Algorithm** - Learns optimal play through self-play
2. **Interactive GUI** - Play against the trained bot
3. **Strategy Analysis** - Visualize and verify the learned Nash equilibrium

## What is Kuhn Poker?

Kuhn Poker is a simplified poker game perfect for studying game theory:
- **3 cards**: Jack (J), Queen (Q), King (K)
- **2 players**: Each gets 1 card
- **Ante**: Both players put in 1 chip
- **Actions**: Check/Bet on first action, Fold/Call when facing a bet
- **Winner**: Higher card wins at showdown

## What is CFR?

CFR is an algorithm that:
1. Tracks **regrets** (how much we regret not taking each action)
2. Updates strategy based on accumulated regrets
3. Converges to **Nash Equilibrium** (optimal play where neither player can improve)

## Files

- `kuhn_poker_cfr.py` - Core CFR implementation and training
- `kuhn_poker_gui.py` - Interactive GUI to play against the bot
- `kuhn_poker_analysis.py` - Strategy analysis and visualization
- `kuhn_poker_strategy.pkl` - Saved trained strategy (generated after training)

## Installation

```bash
# Install required packages
pip install numpy matplotlib seaborn tkinter
```

## Quick Start

### 1. Train the Bot
```bash
python kuhn_poker_cfr.py
```
This trains for 100,000 iterations and saves the strategy.

### 2. Play Against the Bot
```bash
python kuhn_poker_gui.py
```
Launch the GUI and test your skills against the Nash equilibrium bot!

### 3. Analyze the Strategy
```bash
python kuhn_poker_analysis.py
```
Generate visualizations and verify the Nash equilibrium.

## Understanding the Strategy

### Nash Equilibrium Strategy for Kuhn Poker:

**Jack (Weakest)**:
- Opening: Bluff bet ~33% of the time
- Facing bet: Always fold
- After opponent checks: Usually check back

**Queen (Middle)**:
- Opening: Always check
- Facing bet: Call ~33% of the time
- After opponent checks: Check (trap)

**King (Strongest)**:
- Opening: Bet ~67% of the time
- Facing bet: Always call
- After opponent checks: Always bet

## GUI Features

- **Visual card display** with color coding
- **Action history** showing all moves
- **Score tracking** across multiple games
- **Intuitive buttons** that change based on game state
- **Bot thinking delay** for realistic play

## How CFR Works

1. **Information Sets**: Game states from a player's perspective (card + history)
2. **Regret Calculation**: Track how much we regret not taking each action
3. **Strategy Update**: Use regret-matching to update probabilities
4. **Convergence**: After many iterations, converges to Nash equilibrium

### CFR Algorithm Steps:

```python
for each iteration:
    1. Deal random cards
    2. Traverse game tree recursively
    3. Calculate regrets for each action
    4. Update cumulative regrets
    5. Update average strategy
```

## Key Concepts

### Information Set
A player's view of the game state:
- Their card (J, Q, or K)
- Action history (e.g., "pb" = pass, then bet)

### Regret
How much we wish we had taken a different action:
- Positive regret: Should have taken this action more
- Zero/negative regret: Current strategy is fine

### Nash Equilibrium
A strategy where neither player can improve by changing their strategy unilaterally.

## Customization

### Adjust Training Iterations
In `kuhn_poker_cfr.py`:
```python
trainer.train(iterations=100000)  # Increase for better convergence
```

### Modify GUI Colors/Layout
Edit `kuhn_poker_gui.py` to customize the interface.

### Change Analysis Plots
Modify `kuhn_poker_analysis.py` for different visualizations.

## Expected Results

After training, the bot should:
- **Never call with Jack** when facing a bet
- **Sometimes bluff with Jack** when opening
- **Mix strategies with Queen** to remain unpredictable
- **Play aggressively with King** for value

The learned strategy should closely match theoretical Nash equilibrium with average deviation < 0.05.

## Troubleshooting

**Bot not training?**
- Ensure all dependencies are installed
- Check that you have write permissions for saving the strategy file

**GUI not launching?**
- Verify tkinter is installed: `python -m tkinter`
- On Linux: `sudo apt-get install python3-tk`

**Strategy seems suboptimal?**
- Increase training iterations
- Delete `kuhn_poker_strategy.pkl` and retrain

## Mathematical Background

The CFR update formula:

```
Regret(a) = Value(a) - Value(current_strategy)
Strategy(a) = max(0, Regret(a)) / Σ max(0, Regret(all_actions))
```

Expected value at Nash equilibrium:
- First player: -1/18 ≈ -0.056
- Second player: +1/18 ≈ +0.056

## Extensions

Consider extending this project:
1. **Multiplayer Kuhn Poker** (3+ players)
2. **Different deck sizes** (4+ cards)
3. **Leduc Hold'em** (more complex game)
4. **Monte Carlo CFR** (sampling-based)
5. **Neural CFR** (deep learning variant)

## References

- Zinkevich et al. "Regret Minimization in Games with Incomplete Information" (2007)
- Neller & Lanctot "An Introduction to Counterfactual Regret Minimization" (2013)
- Original Kuhn Poker: H. W. Kuhn "A Simplified Two-Person Poker" (1950)

## License

MIT License - Feel free to use and modify!
