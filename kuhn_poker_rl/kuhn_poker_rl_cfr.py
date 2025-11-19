"""
Kuhn Poker RL-CFR (Reinforcement Learning CFR) Implementation
Uses Monte Carlo sampling, experience replay, and importance sampling
"""

import numpy as np
from enum import Enum
import random
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque, defaultdict
import pickle
from dataclasses import dataclass
import time

class Actions(Enum):
    PASS = 0  # Check or Fold
    BET = 1   # Bet or Call

@dataclass
class Experience:
    """Store experience for replay buffer"""
    info_set_key: str
    action: int
    regret: float
    strategy: np.ndarray
    importance_weight: float
    timestamp: int

class KuhnPoker:
    """
    Kuhn Poker Game Rules:
    - 3 cards: Jack (0), Queen (1), King (2)
    - 2 players
    - Each player antes 1 chip
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
        """
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
        return len(history) % 2

class ReplayBuffer:
    """Experience replay buffer for RL-CFR"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.priorities = defaultdict(float)  # Prioritized replay
        
    def add(self, experience: Experience):
        """Add experience to buffer with priority"""
        self.buffer.append(experience)
        # Priority based on regret magnitude
        priority = abs(experience.regret) + 0.01  # Small epsilon to ensure non-zero
        self.priorities[experience.info_set_key] = max(
            self.priorities[experience.info_set_key], 
            priority
        )
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch with prioritized replay"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Calculate sampling probabilities based on priorities
        weights = []
        experiences = list(self.buffer)
        
        for exp in experiences:
            weights.append(self.priorities.get(exp.info_set_key, 0.01))
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            probs = [w / total_weight for w in weights]
        else:
            probs = [1.0 / len(experiences)] * len(experiences)
        
        # Sample with replacement based on priorities
        indices = np.random.choice(
            len(experiences), 
            size=batch_size, 
            p=probs,
            replace=True
        )
        
        return [experiences[i] for i in indices]
    
    def update_priorities(self, info_set_key: str, new_priority: float):
        """Update priority for an information set"""
        self.priorities[info_set_key] = new_priority

class RLCFRTrainer:
    """RL-CFR algorithm implementation for Kuhn Poker"""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 discount_factor: float = 1.0,
                 epsilon: float = 0.1,
                 buffer_capacity: int = 100000,
                 batch_size: int = 32):
        
        self.game = KuhnPoker()
        self.regret_sum = defaultdict(lambda: np.zeros(2))
        self.strategy_sum = defaultdict(lambda: np.zeros(2))
        self.n_actions = 2
        
        # RL components
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # For epsilon-greedy exploration
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        
        # Value estimates for bootstrapping
        self.value_estimates = defaultdict(float)
        
        # Tracking
        self.iteration_count = 0
        self.sampling_player = 0  # Player we're updating this iteration
        
        # Learning rate scheduling
        self.initial_lr = learning_rate
        self.lr_decay = 0.99999
        
    def get_strategy(self, info_set_key: str, use_epsilon_greedy: bool = False) -> np.ndarray:
        """
        Get current strategy using regret matching with optional epsilon-greedy exploration.
        """
        regrets = self.regret_sum[info_set_key]
        positive_regrets = np.maximum(regrets, 0)
        
        if positive_regrets.sum() > 0:
            strategy = positive_regrets / positive_regrets.sum()
        else:
            strategy = np.ones(self.n_actions) / self.n_actions
        
        # Epsilon-greedy exploration during training
        if use_epsilon_greedy and random.random() < self.epsilon:
            # Exploration: uniform random action
            strategy = np.ones(self.n_actions) / self.n_actions
        
        return strategy
    
    def get_average_strategy(self, info_set_key: str) -> np.ndarray:
        """
        Get the average strategy (used for final strategy).
        """
        avg_strategy = self.strategy_sum[info_set_key]
        if avg_strategy.sum() > 0:
            return avg_strategy / avg_strategy.sum()
        else:
            return np.ones(self.n_actions) / self.n_actions
    
    def external_sampling_cfr(self, cards: List[int], history: str, 
                            reach_p0: float, reach_p1: float,
                            sampling_player: int) -> float:
        """
        External sampling MCCFR - only samples actions for one player per iteration.
        This is more efficient than vanilla CFR and allows for RL-style updates.
        """
        # Check if terminal node
        if self.game.is_terminal(history):
            payoff = self.game.get_payoff(cards, history)
            return payoff if sampling_player == 0 else -payoff
        
        # Get current player and information set
        current_player = self.game.get_current_player(history)
        info_set_key = f"{cards[current_player]}:{history}"
        
        # Get strategy
        use_exploration = (current_player == sampling_player)
        strategy = self.get_strategy(info_set_key, use_epsilon_greedy=use_exploration)
        
        # Initialize values
        action_values = np.zeros(self.n_actions)
        
        if current_player == sampling_player:
            # Sample single action for the sampling player (external sampling)
            if use_exploration and random.random() < self.epsilon:
                # Exploration: random action
                action = random.randint(0, self.n_actions - 1)
            else:
                # Exploitation: sample from strategy
                action = np.random.choice(self.n_actions, p=strategy)
            
            action_char = 'p' if action == Actions.PASS.value else 'b'
            next_history = history + action_char
            
            # Update reach probabilities
            if current_player == 0:
                next_reach_p0 = reach_p0 * strategy[action]
                next_reach_p1 = reach_p1
            else:
                next_reach_p0 = reach_p0
                next_reach_p1 = reach_p1 * strategy[action]
            
            # Recursive call for sampled action
            action_values[action] = self.external_sampling_cfr(
                cards, next_history, next_reach_p0, next_reach_p1, sampling_player
            )
            
            # Calculate regrets for all actions (counterfactual)
            for a in range(self.n_actions):
                if a != action:
                    # Estimate value for unsampled actions using value function
                    cf_history = history + ('p' if a == Actions.PASS.value else 'b')
                    cf_info_set = f"{cards[current_player]}:{cf_history}"
                    action_values[a] = self.value_estimates.get(cf_info_set, 0)
            
            node_value = np.dot(strategy, action_values)
            
            # Calculate importance sampling weight
            counterfactual_reach = reach_p1 if current_player == 0 else reach_p0
            importance_weight = counterfactual_reach
            
            # Update regrets with learning rate (RL-style update)
            for a in range(self.n_actions):
                regret = action_values[a] - node_value
                # RL-style update with learning rate
                self.regret_sum[info_set_key][a] += (
                    self.learning_rate * importance_weight * regret
                )
            
            # Store experience in replay buffer
            experience = Experience(
                info_set_key=info_set_key,
                action=action,
                regret=action_values[action] - node_value,
                strategy=strategy.copy(),
                importance_weight=importance_weight,
                timestamp=self.iteration_count
            )
            self.replay_buffer.add(experience)
            
            # Update value estimate (bootstrapping)
            self.value_estimates[info_set_key] = (
                (1 - self.learning_rate) * self.value_estimates[info_set_key] +
                self.learning_rate * node_value
            )
            
            return action_values[action]
            
        else:
            # For opponent, traverse all actions
            node_value = 0
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
                
                action_values[action] = self.external_sampling_cfr(
                    cards, next_history, next_reach_p0, next_reach_p1, sampling_player
                )
                node_value += strategy[action] * action_values[action]
            
            # Update strategy sum for averaging
            reach_probability = reach_p0 if current_player == 0 else reach_p1
            self.strategy_sum[info_set_key] += reach_probability * strategy
            
            return node_value
    
    def replay_update(self):
        """
        Update strategies using experience replay (RL component).
        """
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        for experience in batch:
            # Decay old experiences
            age = self.iteration_count - experience.timestamp
            decay_factor = self.discount_factor ** (age / 1000)  # Slow decay
            
            # Update regret with decayed importance
            current_regrets = self.regret_sum[experience.info_set_key]
            
            # RL-style TD update
            td_error = experience.regret * decay_factor
            current_regrets[experience.action] += (
                self.learning_rate * td_error * experience.importance_weight
            )
            
            # Update priorities in replay buffer
            new_priority = abs(td_error) + 0.01
            self.replay_buffer.update_priorities(experience.info_set_key, new_priority)
    
    def train(self, iterations: int = 10000) -> None:
        """Train using RL-CFR with external sampling and experience replay"""
        cards = [0, 1, 2]
        
        print("Training with RL-CFR (External Sampling + Experience Replay)...")
        print("-" * 60)
        
        start_time = time.time()
        
        for i in range(iterations):
            self.iteration_count = i
            
            # Decay learning rate
            self.learning_rate = self.initial_lr * (self.lr_decay ** i)
            
            # Decay epsilon for exploration
            self.epsilon = max(0.01, self.epsilon * 0.9999)
            
            # Shuffle and deal cards
            random.shuffle(cards)
            dealt_cards = cards[:2]
            
            # Alternate which player we're sampling for
            self.sampling_player = i % 2
            
            # Run external sampling CFR
            self.external_sampling_cfr(dealt_cards, "", 1.0, 1.0, self.sampling_player)
            
            # Periodic replay updates (every 10 iterations)
            if i % 10 == 0 and i > 0:
                self.replay_update()
            
            # Progress updates
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"Iteration {i + 1}/{iterations} | "
                      f"LR: {self.learning_rate:.6f} | "
                      f"ε: {self.epsilon:.4f} | "
                      f"Buffer: {len(self.replay_buffer.buffer)} | "
                      f"Rate: {rate:.1f} it/s")
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.2f}s")
        print(f"Final learning rate: {self.learning_rate:.6f}")
        print(f"Final epsilon: {self.epsilon:.4f}")
        print(f"Replay buffer size: {len(self.replay_buffer.buffer)}")
        self.print_strategy()
    
    def print_strategy(self) -> None:
        """Print the computed strategy"""
        print("\n" + "="*50)
        print("FINAL STRATEGY (RL-CFR Nash Equilibrium):")
        print("="*50)
        
        card_names = ['Jack', 'Queen', 'King']
        
        for card in range(3):
            print(f"\n{card_names[card]}:")
            
            states = [
                ("", "First action", "Check", "Bet"),
                ("p", "After opponent checks", "Check", "Bet"),
                ("b", "Facing a bet", "Fold", "Call"),
                ("pb", "Facing a bet after checking", "Fold", "Call")
            ]
            
            for history, description, action0, action1 in states:
                info_set_key = f"{card}:{history}"
                strategy = self.get_average_strategy(info_set_key)
                print(f"  {description:25} -> {action0}: {strategy[0]:.3f}, {action1}: {strategy[1]:.3f}")
    
    def get_action(self, card: int, history: str) -> int:
        """Get action from trained strategy (for playing)"""
        info_set_key = f"{card}:{history}"
        strategy = self.get_average_strategy(info_set_key)
        return np.random.choice(self.n_actions, p=strategy)
    
    def save_strategy(self, filename: str = "kuhn_poker_rl_strategy.pkl") -> None:
        """Save the trained strategy to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'regret_sum': dict(self.regret_sum),
                'strategy_sum': dict(self.strategy_sum),
                'value_estimates': dict(self.value_estimates),
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'iteration_count': self.iteration_count
            }, f)
        print(f"Strategy saved to {filename}")
    
    def load_strategy(self, filename: str = "kuhn_poker_rl_strategy.pkl") -> None:
        """Load a trained strategy from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.regret_sum = defaultdict(lambda: np.zeros(2), data['regret_sum'])
            self.strategy_sum = defaultdict(lambda: np.zeros(2), data['strategy_sum'])
            self.value_estimates = defaultdict(float, data.get('value_estimates', {}))
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.iteration_count = data.get('iteration_count', 0)
        print(f"Strategy loaded from {filename}")
    
    def compare_with_nash(self) -> None:
        """Compare learned strategy with known Nash equilibrium"""
        print("\n" + "="*50)
        print("COMPARISON WITH THEORETICAL NASH:")
        print("="*50)
        
        # Known Nash equilibrium probabilities for betting/calling
        nash_values = {
            "0:": (1/3, "Jack should bluff bet 1/3"),  # Jack opening
            "1:": (0, "Queen should never open bet"),    # Queen opening
            "2:": (1, "King should always value bet"),   # King opening
            "0:b": (0, "Jack should never call"),       # Jack facing bet
            "1:b": (1/3, "Queen should call 1/3"),      # Queen facing bet
            "2:b": (1, "King should always call"),      # King facing bet
            "0:p": (1/3, "Jack should bluff 1/3 after check"),  # Jack after check
            "2:p": (1, "King should always bet after check"),   # King after check
        }
        
        total_error = 0
        count = 0
        
        for info_set, (nash_prob, description) in nash_values.items():
            learned = self.get_average_strategy(info_set)[1]  # Bet/Call probability
            error = abs(learned - nash_prob)
            total_error += error
            count += 1
            
            symbol = "✓" if error < 0.1 else "⚠"
            print(f"{symbol} {info_set}: Learned={learned:.3f}, Nash={nash_prob:.3f} ({description})")
        
        avg_error = total_error / count
        print(f"\nAverage deviation: {avg_error:.4f}")
        
        if avg_error < 0.05:
            print("Excellent convergence! RL-CFR has learned near-optimal play.")
        elif avg_error < 0.1:
            print("Good convergence. Strategy is close to Nash equilibrium.")
        else:
            print("More training may be needed for optimal convergence.")

def main():
    """Train the RL-CFR agent"""
    print("="*60)
    print("   Kuhn Poker RL-CFR Training")
    print("   (Reinforcement Learning CFR with Experience Replay)")
    print("="*60)
    
    trainer = RLCFRTrainer(
        learning_rate=0.1,      # Higher initial LR for faster learning
        discount_factor=0.999,   # Slight discounting
        epsilon=0.2,            # Initial exploration rate
        buffer_capacity=50000,   # Experience replay buffer size
        batch_size=64           # Batch size for replay updates
    )
    
    # Train with more iterations for RL-CFR (converges differently than vanilla CFR)
    trainer.train(iterations=10000000)
    trainer.save_strategy()
    
    # Compare with theoretical Nash
    trainer.compare_with_nash()
    
    print("\n" + "="*60)
    print("Training complete! RL-CFR strategy saved.")
    print("The agent uses:")
    print("  • External sampling (Monte Carlo)")
    print("  • Experience replay buffer")
    print("  • Epsilon-greedy exploration")
    print("  • Learning rate decay")
    print("  • Prioritized experience replay")
    print("\nRun kuhn_poker_rl_gui.py to play against the RL-CFR bot!")
    print("="*60)

if __name__ == "__main__":
    main()
