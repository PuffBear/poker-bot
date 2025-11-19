"""
Kuhn Poker CFR (Counterfactual Regret Minimization) Implementation
"""

import numpy as np
from enum import Enum
import random
from typing import Dict, List, Tuple, Optional
import pickle

class Actions(Enum):
    PASS = 0  # Check or Fold
    BET = 1   # Bet or Call

class KuhnPoker:
    """
    Kuhn Poker Game Rules:
    - 3 cards: Jack (0), Queen (1), King (2)
    - 2 players
    - Each player antes 1 chip
    - Each player dealt 1 card
    - Player 1 acts first: can check or bet 1
    - If P1 checks: P2 can check (showdown) or bet 1
    - If P2 bets after P1 checks: P1 can fold or call
    - If P1 bets: P2 can fold or call
    """
    
    def __init__(self):
        self.cards = [0, 1, 2]  # J, Q, K
        self.n_actions = 2
        
    def get_payoff(self, cards: List[int], history: str) -> int:
        """
        Returns payoff for player 1 given cards and action history.
        History is a string of actions: 'p' for pass/check/fold, 'b' for bet/call
        """
        # Terminal states
        if history in ['pp', 'bb', 'pbb']:  # Showdown
            return 2 if cards[0] > cards[1] else -2
        elif history == 'bp':  # P2 folds
            return 1
        elif history == 'pbp':  # P1 folds
            return -1
        else:
            raise ValueError(f"Invalid history: {history}")
    
    def is_terminal(self, history: str) -> bool:
        """Check if the game has reached a terminal state"""
        return history in ['pp', 'bp', 'bb', 'pbb', 'pbp']
    
    def get_current_player(self, history: str) -> int:
        """Get which player's turn it is (0 for P1, 1 for P2)"""
        if len(history) in [0, 2]:
            return 0
        return 1

class InformationSet:
    """Represents an information set in the game tree"""
    
    def __init__(self, card: int, history: str):
        self.card = card
        self.history = history
        self.key = f"{card}:{history}"
        
    def __str__(self):
        card_names = ['J', 'Q', 'K']
        return f"Card: {card_names[self.card]}, History: {self.history if self.history else 'start'}"

class CFRTrainer:
    """CFR algorithm implementation for Kuhn Poker"""
    
    def __init__(self):
        self.game = KuhnPoker()
        self.regret_sum = {}  # Cumulative regrets
        self.strategy_sum = {}  # Cumulative strategy for averaging
        self.n_actions = 2
        
    def get_strategy(self, info_set_key: str) -> np.ndarray:
        """
        Get current strategy for an information set using regret matching.
        """
        if info_set_key not in self.regret_sum:
            self.regret_sum[info_set_key] = np.zeros(self.n_actions)
            
        regrets = self.regret_sum[info_set_key]
        positive_regrets = np.maximum(regrets, 0)
        
        if positive_regrets.sum() > 0:
            strategy = positive_regrets / positive_regrets.sum()
        else:
            strategy = np.ones(self.n_actions) / self.n_actions
            
        return strategy
    
    def get_average_strategy(self, info_set_key: str) -> np.ndarray:
        """
        Get the average strategy (used for final strategy).
        """
        if info_set_key not in self.strategy_sum:
            return np.ones(self.n_actions) / self.n_actions
            
        avg_strategy = self.strategy_sum[info_set_key]
        if avg_strategy.sum() > 0:
            return avg_strategy / avg_strategy.sum()
        else:
            return np.ones(self.n_actions) / self.n_actions
    
    def cfr(self, cards: List[int], history: str, reach_p0: float, reach_p1: float) -> float:
        """
        Counterfactual regret minimization recursive function.
        Returns expected value for the current player.
        """
        # Check if terminal node
        if self.game.is_terminal(history):
            payoff = self.game.get_payoff(cards, history)
            current_player = self.game.get_current_player(history[:-1] if history else "")
            return payoff if current_player == 0 else -payoff
        
        # Get current player and information set
        current_player = self.game.get_current_player(history)
        info_set = InformationSet(cards[current_player], history)
        
        # Get strategy for this information set
        strategy = self.get_strategy(info_set.key)
        
        # Initialize values
        action_values = np.zeros(self.n_actions)
        node_value = 0
        
        # Recursively compute values for each action
        for action in range(self.n_actions):
            action_char = 'p' if action == Actions.PASS.value else 'b'
            next_history = history + action_char
            
            # Update reach probabilities
            if current_player == 0:
                next_reach_p0 = reach_p0 * strategy[action]
                next_reach_p1 = reach_p1
            else:
                next_reach_p0 = reach_p0
                next_reach_p1 = reach_p1 * strategy[action]
            
            # Recursive call
            action_values[action] = -self.cfr(cards, next_history, next_reach_p0, next_reach_p1)
            node_value += strategy[action] * action_values[action]
        
        # Update regrets and strategy sum
        if current_player == 0:
            counterfactual_reach = reach_p1
        else:
            counterfactual_reach = reach_p0
            
        # Update regret sum
        if info_set.key not in self.regret_sum:
            self.regret_sum[info_set.key] = np.zeros(self.n_actions)
        
        for action in range(self.n_actions):
            regret = action_values[action] - node_value
            self.regret_sum[info_set.key][action] += counterfactual_reach * regret
        
        # Update strategy sum for averaging
        if info_set.key not in self.strategy_sum:
            self.strategy_sum[info_set.key] = np.zeros(self.n_actions)
            
        reach_probability = reach_p0 if current_player == 0 else reach_p1
        self.strategy_sum[info_set.key] += reach_probability * strategy
        
        return node_value
    
    def train(self, iterations: int = 10000) -> None:
        """Train using CFR for specified iterations"""
        cards = [0, 1, 2]
        
        for i in range(iterations):
            # Shuffle and deal cards
            random.shuffle(cards)
            dealt_cards = cards[:2]
            
            # Run CFR
            self.cfr(dealt_cards, "", 1.0, 1.0)
            
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{iterations}")
        
        print("\nTraining complete!")
        self.print_strategy()
    
    def print_strategy(self) -> None:
        """Print the computed strategy"""
        print("\n" + "="*50)
        print("FINAL STRATEGY (Nash Equilibrium):")
        print("="*50)
        
        card_names = ['Jack', 'Queen', 'King']
        
        for card in range(3):
            print(f"\n{card_names[card]}:")
            
            # Different game states
            states = [
                ("", "First action", "Check", "Bet"),
                ("p", "After opponent checks", "Check", "Bet"),
                ("b", "Facing a bet", "Fold", "Call"),
                ("pb", "Facing a bet after checking", "Fold", "Call")
            ]
            
            for history, description, action0, action1 in states:
                info_set_key = f"{card}:{history}"
                if info_set_key in self.strategy_sum:
                    strategy = self.get_average_strategy(info_set_key)
                    print(f"  {description:25} -> {action0}: {strategy[0]:.3f}, {action1}: {strategy[1]:.3f}")
    
    def get_action(self, card: int, history: str) -> int:
        """Get action from trained strategy"""
        info_set_key = f"{card}:{history}"
        strategy = self.get_average_strategy(info_set_key)
        return np.random.choice(self.n_actions, p=strategy)
    
    def save_strategy(self, filename: str = "kuhn_poker_strategy.pkl") -> None:
        """Save the trained strategy to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'regret_sum': self.regret_sum,
                'strategy_sum': self.strategy_sum
            }, f)
        print(f"Strategy saved to {filename}")
    
    def load_strategy(self, filename: str = "kuhn_poker_strategy.pkl") -> None:
        """Load a trained strategy from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.regret_sum = data['regret_sum']
            self.strategy_sum = data['strategy_sum']
        print(f"Strategy loaded from {filename}")

def main():
    """Train the CFR agent"""
    print("Training Kuhn Poker CFR Agent...")
    print("-" * 50)
    
    trainer = CFRTrainer()
    trainer.train(iterations=100000)  # More iterations for better convergence
    trainer.save_strategy()
    
    print("\n" + "="*50)
    print("Training complete! Strategy saved.")
    print("Run kuhn_poker_gui.py to play against the bot!")

if __name__ == "__main__":
    main()
