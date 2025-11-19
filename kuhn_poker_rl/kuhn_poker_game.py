import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class KuhnPokerGame:
    """Kuhn Poker game implementation with CFR solver"""
    
    CARDS = ['J', 'Q', 'K']
    ACTIONS = ['check', 'bet']
    
    def __init__(self):
        self.reset_game()
        
    def reset_game(self):
        """Reset game state for a new hand"""
        self.deck = self.CARDS.copy()
        random.shuffle(self.deck)
        self.player_card = self.deck[0]
        self.bot_card = self.deck[1]
        self.pot = 2  # Both players ante 1 chip
        self.history = []
        self.terminal = False
        self.winner = None
        self.payoff = 0
        
    def get_info_set(self, card: str, history: List[str]) -> str:
        """Get information set string for CFR"""
        return f"{card}:{''.join(history)}"
    
    def is_terminal(self, history: List[str]) -> bool:
        """Check if game state is terminal"""
        if len(history) >= 2:
            # Both players checked
            if history[-1] == 'check' and history[-2] == 'check':
                return True
            # Bet was called or folded
            if len(history) >= 2 and history[-2] == 'bet':
                return True
        return False
    
    def get_payoff(self, history: List[str], player_card: str, bot_card: str) -> int:
        """Calculate payoff for player (positive = player wins, negative = bot wins)"""
        card_values = {'J': 0, 'Q': 1, 'K': 2}
        
        # Someone folded to a bet
        if len(history) >= 2 and history[-2] == 'bet' and history[-1] == 'check':
            # Player bet, bot folded OR bot bet, player folded
            if len(history) % 2 == 0:  # Even length means bot made last move
                return 1  # Bot folded, player wins
            else:
                return -1  # Player folded, bot wins
        
        # Showdown scenarios
        player_value = card_values[player_card]
        bot_value = card_values[bot_card]
        
        # Both checked (pot = 2)
        if history[-1] == 'check' and history[-2] == 'check':
            return 1 if player_value > bot_value else -1
        
        # Bet was called (pot = 4)
        if history[-1] == 'bet' or (len(history) >= 2 and history[-2] == 'bet'):
            payoff = 2 if player_value > bot_value else -2
            return payoff
        
        return 0
    
    def make_move(self, action: str, is_player: bool) -> bool:
        """Make a move in the game"""
        if self.terminal:
            return False
        
        self.history.append(action)
        
        if self.is_terminal(self.history):
            self.terminal = True
            self.payoff = self.get_payoff(self.history, self.player_card, self.bot_card)
            
            # Determine winner
            if self.payoff > 0:
                self.winner = "Player"
            else:
                self.winner = "Bot"
                
            # Update pot size
            if action == 'bet' or (len(self.history) >= 2 and self.history[-2] == 'bet'):
                self.pot = 4
                
            return True
        
        return False


class CFRStrategy:
    """Counterfactual Regret Minimization strategy"""
    
    def __init__(self):
        self.regret_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
        self.strategy_sum: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
        self.strategy: Dict[str, np.ndarray] = {}
        
    def get_strategy(self, info_set: str) -> np.ndarray:
        """Get current strategy for an information set"""
        regrets = self.regret_sum[info_set]
        
        # Use regret matching
        strategy = np.maximum(regrets, 0)
        normalizing_sum = np.sum(strategy)
        
        if normalizing_sum > 0:
            strategy = strategy / normalizing_sum
        else:
            strategy = np.ones(2) / 2  # Uniform random if no positive regrets
            
        return strategy
    
    def get_average_strategy(self, info_set: str) -> np.ndarray:
        """Get average strategy over all iterations"""
        avg_strategy = self.strategy_sum[info_set]
        normalizing_sum = np.sum(avg_strategy)
        
        if normalizing_sum > 0:
            return avg_strategy / normalizing_sum
        else:
            return np.ones(2) / 2
    
    def update_strategy(self, info_set: str, realized_strategy: np.ndarray):
        """Update strategy sum for averaging"""
        self.strategy_sum[info_set] += realized_strategy
    
    def update_regret(self, info_set: str, regrets: np.ndarray):
        """Update regret sum"""
        self.regret_sum[info_set] += regrets
    
    def train(self, iterations: int = 10000):
        """Train the CFR strategy"""
        game = KuhnPokerGame()
        
        for _ in range(iterations):
            # Shuffle deck
            deck = game.CARDS.copy()
            random.shuffle(deck)
            player_card = deck[0]
            bot_card = deck[1]
            
            # Train from both perspectives
            self._cfr([], player_card, bot_card, 1.0, 1.0)
            self._cfr([], player_card, bot_card, 1.0, 1.0)
    
    def _cfr(self, history: List[str], player_card: str, bot_card: str, 
             p0: float, p1: float) -> float:
        """CFR recursive algorithm"""
        game = KuhnPokerGame()
        
        # Check if terminal
        if game.is_terminal(history):
            return game.get_payoff(history, player_card, bot_card)
        
        # Determine current player (0 = player, 1 = bot)
        current_player = len(history) % 2
        card = player_card if current_player == 0 else bot_card
        info_set = game.get_info_set(card, history)
        
        # Get strategy
        strategy = self.get_strategy(info_set)
        
        # Compute action utilities
        action_utils = np.zeros(2)
        for i, action in enumerate(game.ACTIONS):
            new_history = history + [action]
            if current_player == 0:
                action_utils[i] = -self._cfr(new_history, player_card, bot_card, p0 * strategy[i], p1)
            else:
                action_utils[i] = -self._cfr(new_history, player_card, bot_card, p0, p1 * strategy[i])
        
        # Expected utility
        util = np.sum(action_utils * strategy)
        
        # Compute regrets
        regrets = action_utils - util
        
        # Update regrets and strategy
        if current_player == 0:
            self.update_regret(info_set, regrets * p1)
            self.update_strategy(info_set, strategy * p0)
        else:
            self.update_regret(info_set, regrets * p0)
            self.update_strategy(info_set, strategy * p1)
        
        return util


class BotPlayer:
    """Bot player with different strategy options"""
    
    def __init__(self, strategy_type: str = "CFR"):
        self.strategy_type = strategy_type
        self.cfr = CFRStrategy()
        self.last_probabilities = None
        self.last_info_set = None
        self.thought_process = ""
        
        # Train CFR if using that strategy
        if strategy_type == "CFR":
            print("Training CFR strategy...")
            self.cfr.train(10000)
            print("CFR training complete!")
    
    def get_action(self, game: KuhnPokerGame) -> Tuple[str, dict]:
        """Get bot action based on current strategy"""
        info_set = game.get_info_set(game.bot_card, game.history)
        self.last_info_set = info_set
        
        if self.strategy_type == "CFR":
            return self._cfr_action(game, info_set)
        elif self.strategy_type == "Random":
            return self._random_action(game, info_set)
        elif self.strategy_type == "Aggressive":
            return self._aggressive_action(game, info_set)
        elif self.strategy_type == "Conservative":
            return self._conservative_action(game, info_set)
        else:
            return self._cfr_action(game, info_set)
    
    def _cfr_action(self, game: KuhnPokerGame, info_set: str) -> Tuple[str, dict]:
        """CFR-based action"""
        strategy = self.cfr.get_average_strategy(info_set)
        self.last_probabilities = strategy.copy()
        
        action = np.random.choice(game.ACTIONS, p=strategy)
        
        # Build thought process
        card_strength = {"J": "weak", "Q": "medium", "K": "strong"}[game.bot_card]
        self.thought_process = f"Card: {game.bot_card} ({card_strength})\n"
        self.thought_process += f"Check: {strategy[0]:.1%}, Bet: {strategy[1]:.1%}\n"
        self.thought_process += f"Decision: {action.upper()}"
        
        info = {
            "probabilities": strategy,
            "info_set": info_set,
            "using_strategy": True,
            "thought": self.thought_process
        }
        
        return action, info
    
    def _random_action(self, game: KuhnPokerGame, info_set: str) -> Tuple[str, dict]:
        """Random action"""
        strategy = np.array([0.5, 0.5])
        self.last_probabilities = strategy
        action = random.choice(game.ACTIONS)
        
        self.thought_process = f"Card: {game.bot_card}\n"
        self.thought_process += "Playing randomly\n"
        self.thought_process += f"Decision: {action.upper()}"
        
        info = {
            "probabilities": strategy,
            "info_set": info_set,
            "using_strategy": False,
            "thought": self.thought_process
        }
        
        return action, info
    
    def _aggressive_action(self, game: KuhnPokerGame, info_set: str) -> Tuple[str, dict]:
        """Aggressive betting strategy"""
        card_values = {'J': 0, 'Q': 1, 'K': 2}
        card_value = card_values[game.bot_card]
        
        # Bet more often, even with weak cards
        if card_value == 2:  # King
            bet_prob = 0.9
        elif card_value == 1:  # Queen
            bet_prob = 0.7
        else:  # Jack
            bet_prob = 0.5
        
        strategy = np.array([1 - bet_prob, bet_prob])
        self.last_probabilities = strategy
        action = np.random.choice(game.ACTIONS, p=strategy)
        
        self.thought_process = f"Card: {game.bot_card}\n"
        self.thought_process += "Aggressive strategy\n"
        self.thought_process += f"Check: {strategy[0]:.1%}, Bet: {strategy[1]:.1%}\n"
        self.thought_process += f"Decision: {action.upper()}"
        
        info = {
            "probabilities": strategy,
            "info_set": info_set,
            "using_strategy": False,
            "thought": self.thought_process
        }
        
        return action, info
    
    def _conservative_action(self, game: KuhnPokerGame, info_set: str) -> Tuple[str, dict]:
        """Conservative strategy - only bet with strong cards"""
        card_values = {'J': 0, 'Q': 1, 'K': 2}
        card_value = card_values[game.bot_card]
        
        # Only bet with strong cards
        if card_value == 2:  # King
            bet_prob = 0.8
        elif card_value == 1:  # Queen
            bet_prob = 0.3
        else:  # Jack
            bet_prob = 0.1
        
        strategy = np.array([1 - bet_prob, bet_prob])
        self.last_probabilities = strategy
        action = np.random.choice(game.ACTIONS, p=strategy)
        
        self.thought_process = f"Card: {game.bot_card}\n"
        self.thought_process += "Conservative strategy\n"
        self.thought_process += f"Check: {strategy[0]:.1%}, Bet: {strategy[1]:.1%}\n"
        self.thought_process += f"Decision: {action.upper()}"
        
        info = {
            "probabilities": strategy,
            "info_set": info_set,
            "using_strategy": False,
            "thought": self.thought_process
        }
        
        return action, info