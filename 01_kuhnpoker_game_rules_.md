# Chapter 1: KuhnPoker Game Rules

Welcome to the first chapter of our `poker-bot` tutorial! Before our bot can become a master poker player, it needs to understand the game it's playing. Just like you can't play chess without knowing how the knight moves, our bot can't play poker without knowing the fundamental rules.

This chapter introduces **Kuhn Poker**, a simplified version of poker, which is perfect for learning complex AI concepts. We'll explore its basic mechanics and how our `poker-bot` project uses an abstraction called `KuhnPoker` to define these rules.

### Why Do We Need "KuhnPoker Game Rules"?

Imagine you're teaching a robot to play a card game. What's the very first thing it needs? The rulebook!

In our `poker-bot` project, the `KuhnPoker` abstraction acts as that official rulebook. It specifies:
*   What cards are in play.
*   How many players are involved.
*   What actions players can take (like 'check' or 'bet').
*   How these actions change the game.
*   Most importantly, how to figure out who wins and how much money they get at the end of a round.

This "rulebook" is essential because:
1.  **For the Bot:** The bot uses these rules to understand what moves are allowed, what state the game is in, and ultimately, to learn how to play optimally.
2.  **For the GUI:** The graphical interface uses these rules to validate player actions, display the current game state correctly, and determine the winner of each round.

Our central use case for this chapter is simple: **We want to be able to accurately determine the winner and payoff for any given Kuhn Poker game scenario.**

---

### The Fundamental Mechanics of Kuhn Poker

Kuhn Poker is a very simple two-player game, but it captures many of the strategic elements of real poker. Let's break down its core components:

#### 1. The Cards

There are only three cards in a Kuhn Poker deck:
*   **Jack (J)**
*   **Queen (Q)**
*   **King (K)**

Their values are straightforward: King is highest, then Queen, then Jack. In our code, these are often represented by numbers: `0` for Jack, `1` for Queen, and `2` for King.

#### 2. The Players

Kuhn Poker is always played with **two players**. We'll often refer to them as Player 1 (P1) and Player 2 (P2).

#### 3. The Antes and Pot

At the start of each round, both Player 1 and Player 2 **ante** (put in) 1 chip into the **pot**. This means the pot always starts with 2 chips. If players bet or call, more chips are added to the pot.

#### 4. Player Actions

Players have two main types of actions they can take:
*   **PASS (Check/Fold)**:
    *   If no one has bet yet, "PASS" means to **Check** (stay in the game without betting).
    *   If someone *has* bet, "PASS" means to **Fold** (give up your hand, losing any money already in the pot).
*   **BET (Bet/Call)**:
    *   If no one has bet yet, "BET" means to put 1 chip into the pot.
    *   If someone *has* bet, "BET" means to **Call** (match the opponent's bet, putting 1 chip into the pot).

#### 5. The Game Flow (Action Sequence)

The game unfolds in a specific sequence of actions, which we track using a **history string**. Each action is represented by a single letter: `'p'` for PASS and `'b'` for BET.

Here's how a round typically plays out:

1.  **Cards Dealt**: Each player is secretly dealt one card.
2.  **Player 1's Turn**: Player 1 acts first.
    *   P1 can **Check** (`p`)
    *   P1 can **Bet** (`b`)
3.  **If P1 Checks (`p`)**: Now it's Player 2's turn.
    *   P2 can **Check** (`pp`): Both players checked. The game ends, cards are revealed, and the higher card wins the pot. This is a **Showdown**.
    *   P2 can **Bet** (`pb`): Player 2 bets 1 chip. Now it's P1's turn again, facing a bet.
        *   P1 can **Fold** (`pbp`): Player 1 folds. Player 1 loses the pot to Player 2. Game ends.
        *   P1 can **Call** (`pbb`): Player 1 calls Player 2's bet. Both players have matched bets. The game ends, cards are revealed, and the higher card wins the pot. This is a **Showdown**.
4.  **If P1 Bets (`b`)**: Now it's Player 2's turn, facing a bet.
    *   P2 can **Fold** (`bp`): Player 2 folds. Player 2 loses the pot to Player 1. Game ends.
    *   P2 can **Call** (`bb`): Player 2 calls Player 1's bet. Both players have matched bets. The game ends, cards are revealed, and the higher card wins the pot. This is a **Showdown**.

Let's visualize this flow with a simple sequence diagram:

```mermaid
sequenceDiagram
    participant P1 as Player 1
    participant P2 as Player 2
    participant KP as KuhnPoker Rules

    P1->>P2: (Deals cards)
    Note over P1,P2: Each player sees their own card. Pot starts at 2 chips.

    P1->>KP: What are my options?
    KP-->>P1: Check (p) or Bet (b)

    P1->>P2: Checks (action 'p')
    Note over P1,P2: History: 'p'

    P2->>KP: What are my options?
    KP-->>P2: Check (p) or Bet (b)

    P2->>P1: Bets (action 'b')
    Note over P1,P2: History: 'pb'. Pot increases to 3 chips.

    P1->>KP: What are my options?
    KP-->>P1: Fold (p) or Call (b)

    P1->>P2: Calls (action 'b')
    Note over P1,P2: History: 'pbb'. Pot increases to 4 chips.

    KP->>P1,P2: Game ends (terminal state 'pbb')
    Note over P1,P2: Cards are revealed.
    KP->>P1,P2: Determines winner and payoff.
```

#### 6. Payoffs

At the end of a round (when the game reaches a "terminal" state), the payoff is determined based on the final actions and the cards:

*   **Showdown (history `pp`, `bb`, `pbb`)**:
    *   If Player 1 has the higher card, P1 wins 2 chips from P2 (total pot 4 chips, P1 gets +2, P2 gets -2).
    *   If Player 2 has the higher card, P1 loses 2 chips to P2 (P1 gets -2, P2 gets +2).
*   **Player 2 Folds (history `bp`)**: Player 1 wins 1 chip from P2 (pot was 3 chips, P1 gets +1, P2 gets -1).
*   **Player 1 Folds (history `pbp`)**: Player 2 wins 1 chip from P1 (pot was 3 chips, P1 gets -1, P2 gets +1).

Notice that payoffs are always described from **Player 1's perspective**. A positive payoff means Player 1 wins money, a negative payoff means Player 1 loses money.

---

### How Our Project Uses These Rules (The `KuhnPoker` Class)

Our project defines these rules within a special class called `KuhnPoker`. This class provides methods to:
*   Understand the basic game setup.
*   Determine if a game has ended.
*   Find out whose turn it is.
*   Calculate the payoff.

Let's look at a simplified version of the `KuhnPoker` class from `kuhn_poker_rl/codes/kuhn_poker_rl_cfr.py`.

```python
# From kuhn_poker_rl/codes/kuhn_poker_rl_cfr.py

class KuhnPoker:
    """
    Kuhn Poker Game Rules:
    - 3 cards: Jack (0), Queen (1), King (2)
    - 2 players
    - Each player antes 1 chip
    # ... rest of rules in comments
    """
    
    def __init__(self):
        self.cards = [0, 1, 2]  # J, Q, K
        self.n_actions = 2      # PASS (0) and BET (1)
```
This `__init__` method simply sets up the basic properties of the game: the cards (represented by numbers 0, 1, 2) and the number of possible actions (2: Pass or Bet).

#### 1. Checking if the Game is Over (`is_terminal`)

The `is_terminal` method tells us if the game has reached one of its final states where cards are revealed or a player has folded.

```python
# From kuhn_poker_rl/codes/kuhn_poker_rl_cfr.py

class KuhnPoker:
    # ... __init__ method ...
    
    def is_terminal(self, history: str) -> bool:
        """Check if the game has reached a terminal state"""
        # A terminal state is when a player folds or after a showdown
        return history in ['pp', 'bp', 'bb', 'pbb', 'pbp']
```
**Explanation:** This method takes the `history` string (e.g., `'pb'`) as input. It then checks if this history matches any of the five specific sequences that signify the end of a round. If it does, the game is over, and it returns `True`. Otherwise, more actions are needed.

**Example Use:**
If `history` is `'pbb'`, `is_terminal('pbb')` would return `True` (it's a showdown).
If `history` is `'pb'`, `is_terminal('pb')` would return `False` (Player 1 still needs to act).

#### 2. Determining Whose Turn It Is (`get_current_player`)

The `get_current_player` method helps us figure out whether Player 1 or Player 2 should make the next move.

```python
# From kuhn_poker_rl/codes/kuhn_poker_rl_cfr.py

class KuhnPoker:
    # ... __init__ and is_terminal methods ...
    
    def get_current_player(self, history: str) -> int:
        """Get which player's turn it is (0 for P1, 1 for P2)"""
        # Player 1 (0) acts on even length histories (0, 2, 4...)
        # Player 2 (1) acts on odd length histories (1, 3...)
        return len(history) % 2
```
**Explanation:** The number of actions taken so far (the length of the `history` string) tells us whose turn it is.
*   If `len(history)` is `0` (start of game), `0 % 2 = 0`, so it's Player 1's turn.
*   If `len(history)` is `1` (e.g., P1 checked or bet), `1 % 2 = 1`, so it's Player 2's turn.
*   If `len(history)` is `2` (e.g., P1 checked, P2 bet), `2 % 2 = 0`, so it's Player 1's turn again.

**Example Use:**
If `history` is `""` (empty string, start of game), `get_current_player("")` returns `0` (Player 1).
If `history` is `'p'`, `get_current_player("p")` returns `1` (Player 2).

#### 3. Calculating the Payoff (`get_payoff`)

This is the core method that determines the final outcome of a round.

```python
# From kuhn_poker_rl/codes/kuhn_poker_rl_cfr.py

class KuhnPoker:
    # ... __init__, is_terminal, and get_current_player methods ...
    
    def get_payoff(self, cards: List[int], history: str) -> int:
        """
        Returns payoff for player 1 given cards and action history.
        cards: [P1_card, P2_card] where 0=J, 1=Q, 2=K
        """
        if history in ['pp', 'bb', 'pbb']:  # Showdown scenarios
            # P1 wins if their card is higher, P2 wins if their card is higher
            return 2 if cards[0] > cards[1] else -2
        elif history == 'bp':  # P1 bet, P2 folded
            return 1
        elif history == 'pbp':  # P1 checked, P2 bet, P1 folded
            return -1
        else:
            raise ValueError(f"Invalid history: {history}")
```
**Explanation:** This method takes the `cards` (e.g., `[1, 0]` means P1 has Queen, P2 has Jack) and the `history` string as input. It then uses the rules we discussed earlier to determine the payoff *from Player 1's perspective*.

**Example Use:**
*   **Showdown:** `get_payoff([2, 0], 'bb')` (P1 King, P2 Jack, P1 bet, P2 called) -> P1 has higher card (`2 > 0`), so returns `2` (P1 wins 2 chips).
*   **Fold:** `get_payoff([0, 2], 'bp')` (P1 Jack, P2 King, P1 bet, P2 folded) -> P2 folded, so P1 wins `1` chip, even with a worse card.

---

### Conclusion

In this first chapter, we've laid the groundwork for understanding Kuhn Poker, the simplified game our `poker-bot` will master. We've learned about the cards, players, actions, game flow, and most importantly, how payoffs are determined. The `KuhnPoker` class in our project acts as the "official rulebook," providing essential methods like `is_terminal`, `get_current_player`, and `get_payoff` to manage the game's mechanics.

Understanding these fundamental rules is crucial because all other components of our `poker-bot` – from its graphical interface to its learning algorithms – rely on this `KuhnPoker` abstraction to know how to play the game correctly.

Now that we understand the rules, let's move on to seeing how these rules are presented to a human player and how we can interact with the game visually.

[Next Chapter: Kuhn Poker Graphical User Interface (GUI)](02_kuhn_poker_graphical_user_interface__gui__.md)
