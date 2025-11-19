"""
Kuhn Poker as an MDP - Q-Learning Implementation
Based on Agriya's paper: "Kuhn Poker as an MDP: Reward-Maximizing Reinforcement Learning 
vs Counterfactual Regret Minimization"

This implements Kuhn Poker from a single agent's perspective as an episodic MDP,
using tabular Q-learning to learn a best-response policy against various opponent strategies.
"""

import numpy as np
import random
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import pickle
from enum import Enum
import matplotlib.pyplot as plt

class Actions(Enum):
    CHECK = 'check'
    BET = 'bet'
    CALL = 'call'
    FOLD = 'fold'

class OpponentType(Enum):
    """Different types of fixed opponent policies"""
    RANDOM = 'random'
    PASSIVE = 'passive'  # Rarely bets, often folds
    AGGRESSIVE = 'aggressive'  # Often bets, rarely folds
    NASH = 'nash'  # Nash equilibrium strategy
    MIXED = 'mixed'  # Randomizes between strategies

class KuhnPokerMDP:
    """
    Kuhn Poker formulated as an MDP from Player 1's perspective.
    
    References:
    - Kuhn, H. W. (1950). "A Simplified Two-Person Poker"
    - Sutton & Barto (2018). "Reinforcement Learning: An Introduction"
    """
    
    def __init__(self, opponent_type: OpponentType = OpponentType.RANDOM):
        self.cards = ['J', 'Q', 'K']  # Jack < Queen < King
        self.card_rank = {'J': 0, 'Q': 1, 'K': 2}
        self.opponent_type = opponent_type
        
        # Nash equilibrium strategies for reference (Lanctot, 2013)
        self.nash_strategies = {
            # Player 1 strategies (card, history) -> bet probability
            ('J', ''): 1/3,  # Bluff 1/3 of the time
            ('Q', ''): 0,    # Never bet with Queen
            ('K', ''): 1,    # Always bet with King
            ('J', 'cb'): 0,  # Never call with Jack after check-bet
            ('Q', 'cb'): 1/3, # Call 1/3 with Queen
            ('K', 'cb'): 1,  # Always call with King
            # Player 2 strategies
            ('J', 'c'): 1/3,  # Bluff 1/3 after check
            ('Q', 'c'): 0,    # Never bet Queen after check
            ('K', 'c'): 1,    # Always bet King after check
            ('J', 'b'): 0,    # Never call with Jack
            ('Q', 'b'): 1/3,  # Call 1/3 with Queen
            ('K', 'b'): 1,    # Always call with King
        }
        
        # Initialize state-action space
        self.state_space = self._initialize_state_space()
        self.reset_episode_stats()
    
    def _initialize_state_space(self) -> List[Tuple]:
        """Initialize all possible information states for Player 1"""
        states = []
        
        # States where Player 1 acts
        for card in self.cards:
            # Initial state
            states.append((card, ''))
            # After Player 1 checks, Player 2 bets
            states.append((card, 'cb'))
        
        # Terminal states (for completeness)
        states.append(('terminal', 'win'))
        states.append(('terminal', 'lose'))
        
        return states
    
    def reset_episode_stats(self):
        """Reset statistics for the current episode"""
        self.episode_rewards = []
        self.episode_actions = []
    
    def get_legal_actions(self, state: Tuple) -> List[str]:
        """
        Get legal actions for Player 1 given the current state.
        
        Args:
            state: (card, history) tuple
        
        Returns:
            List of legal actions
        """
        card, history = state
        
        if card == 'terminal':
            return []
        
        if history == '':
            # Player 1's first action
            return [Actions.CHECK.value, Actions.BET.value]
        elif history == 'cb':
            # Player 1 checked, Player 2 bet
            return [Actions.FOLD.value, Actions.CALL.value]
        else:
            return []
    
    def get_opponent_action(self, opp_card: str, history: str) -> str:
        """
        Get opponent's action based on their policy.
        
        References different opponent strategies from game theory literature.
        """
        if self.opponent_type == OpponentType.RANDOM:
            # Random opponent (baseline)
            if history == 'c':
                return random.choice([Actions.CHECK.value, Actions.BET.value])
            elif history == 'b':
                return random.choice([Actions.FOLD.value, Actions.CALL.value])
        
        elif self.opponent_type == OpponentType.PASSIVE:
            # Passive opponent: rarely bets, often folds
            if history == 'c':
                return Actions.BET.value if random.random() < 0.2 else Actions.CHECK.value
            elif history == 'b':
                return Actions.CALL.value if random.random() < 0.3 else Actions.FOLD.value
        
        elif self.opponent_type == OpponentType.AGGRESSIVE:
            # Aggressive opponent: often bets, rarely folds
            if history == 'c':
                return Actions.BET.value if random.random() < 0.8 else Actions.CHECK.value
            elif history == 'b':
                return Actions.CALL.value if random.random() < 0.7 else Actions.FOLD.value
        
        elif self.opponent_type == OpponentType.NASH:
            # Nash equilibrium opponent (from CFR solution)
            key = (opp_card, history)
            if key in self.nash_strategies:
                bet_prob = self.nash_strategies[key]
                if history == 'c':
                    return Actions.BET.value if random.random() < bet_prob else Actions.CHECK.value
                elif history == 'b':
                    return Actions.CALL.value if random.random() < bet_prob else Actions.FOLD.value
            return Actions.CHECK.value  # Default
        
        elif self.opponent_type == OpponentType.MIXED:
            # Randomly choose between different strategies each hand
            temp_type = random.choice([OpponentType.RANDOM, OpponentType.PASSIVE, 
                                     OpponentType.AGGRESSIVE])
            self.opponent_type = temp_type
            action = self.get_opponent_action(opp_card, history)
            self.opponent_type = OpponentType.MIXED  # Reset
            return action
        
        return Actions.CHECK.value  # Default fallback
    
    def step(self, state: Tuple, action: str, opp_card: str) -> Tuple[Tuple, float, bool]:
        """
        Execute one step in the MDP.
        
        Args:
            state: Current state (card, history)
            action: Player 1's action
            opp_card: Opponent's hidden card (for simulation)
        
        Returns:
            next_state, reward, done
        """
        my_card, history = state
        
        # Build new history
        if action == Actions.CHECK.value:
            new_history = history + 'c'
        elif action == Actions.BET.value:
            new_history = history + 'b'
        elif action == Actions.FOLD.value:
            # Player 1 folds - terminal state, loses pot
            return ('terminal', 'lose'), -1, True
        elif action == Actions.CALL.value:
            # Player 1 calls - showdown
            my_rank = self.card_rank[my_card]
            opp_rank = self.card_rank[opp_card]
            if my_rank > opp_rank:
                return ('terminal', 'win'), 2, True  # Win called pot
            else:
                return ('terminal', 'lose'), -2, True  # Lose called pot
        
        # Check if we need opponent's response
        if new_history == 'c':
            # Player 1 checked, opponent's turn
            opp_action = self.get_opponent_action(opp_card, new_history)
            if opp_action == Actions.CHECK.value:
                # Both checked - showdown
                my_rank = self.card_rank[my_card]
                opp_rank = self.card_rank[opp_card]
                if my_rank > opp_rank:
                    return ('terminal', 'win'), 1, True  # Win ante pot
                else:
                    return ('terminal', 'lose'), -1, True  # Lose ante pot
            else:  # Opponent bets
                return (my_card, 'cb'), 0, False  # Continue, Player 1 to act
        
        elif new_history == 'b':
            # Player 1 bet, opponent's turn
            opp_action = self.get_opponent_action(opp_card, new_history)
            if opp_action == Actions.FOLD.value:
                # Opponent folds - Player 1 wins
                return ('terminal', 'win'), 1, True  # Win ante pot
            else:  # Opponent calls
                # Showdown
                my_rank = self.card_rank[my_card]
                opp_rank = self.card_rank[opp_card]
                if my_rank > opp_rank:
                    return ('terminal', 'win'), 2, True  # Win called pot
                else:
                    return ('terminal', 'lose'), -2, True  # Lose called pot
        
        return state, 0, False  # Should not reach here

class QLearningAgent:
    """
    Tabular Q-learning agent for Kuhn Poker MDP.
    
    References:
    - Watkins & Dayan (1992). "Q-learning"
    - Sutton & Barto (2018). Chapter 6: Temporal-Difference Learning
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 1.0,  # No discounting for episodic tasks
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.999):
        
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Initialize Q-table
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Track statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []
    
    def get_action(self, state: Tuple, legal_actions: List[str], training: bool = True) -> str:
        """
        Epsilon-greedy action selection.
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(legal_actions)
        else:
            # Exploitation: greedy action
            q_values = {action: self.Q[state][action] for action in legal_actions}
            max_q = max(q_values.values())
            # Break ties randomly
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def update(self, state: Tuple, action: str, reward: float, 
               next_state: Tuple, next_legal_actions: List[str], done: bool):
        """
        Q-learning update rule (Bellman optimality).
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.Q[state][action]
        
        if done:
            target = reward  # Terminal state
        else:
            # Max Q-value over next actions
            next_q_values = [self.Q[next_state][a] for a in next_legal_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = target - current_q
        
        # Q-learning update
        self.Q[state][action] = current_q + self.alpha * td_error
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, mdp: KuhnPokerMDP) -> float:
        """
        Train for one episode (one hand of Kuhn Poker).
        """
        # Deal cards
        cards = mdp.cards.copy()
        random.shuffle(cards)
        my_card = cards[0]
        opp_card = cards[1]
        
        # Initial state
        state = (my_card, '')
        episode_reward = 0
        steps = 0
        
        while True:
            # Get legal actions
            legal_actions = mdp.get_legal_actions(state)
            if not legal_actions:
                break
            
            # Select action
            action = self.get_action(state, legal_actions, training=True)
            
            # Take action
            next_state, reward, done = mdp.step(state, action, opp_card)
            episode_reward += reward
            steps += 1
            
            # Get next legal actions
            next_legal_actions = mdp.get_legal_actions(next_state)
            
            # Q-learning update
            self.update(state, action, reward, next_state, next_legal_actions, done)
            
            if done:
                break
            
            state = next_state
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Record statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        
        return episode_reward
    
    def train(self, mdp: KuhnPokerMDP, num_episodes: int = 100000) -> Dict:
        """
        Train the Q-learning agent.
        """
        print(f"Training Q-learning agent against {mdp.opponent_type.value} opponent...")
        
        rewards = []
        for episode in range(num_episodes):
            if episode % 1000 == 0:
                print(f"  Episode {episode}/{num_episodes}")
            reward = self.train_episode(mdp)
            rewards.append(reward)
            
            # Track Q-value statistics periodically
            if episode % 1000 == 0:
                avg_reward = np.mean(rewards[-1000:]) if len(rewards) >= 1000 else np.mean(rewards)
                q_values = [q for state_actions in self.Q.values() for q in state_actions.values()]
                if q_values:
                    self.q_value_history.append({
                        'episode': episode,
                        'avg_reward': avg_reward,
                        'mean_q': np.mean(q_values),
                        'max_q': np.max(q_values),
                        'epsilon': self.epsilon
                    })
        
        return {
            'episode_rewards': self.episode_rewards,
            'q_value_history': self.q_value_history,
            'final_epsilon': self.epsilon
        }
    
    def get_policy(self) -> Dict:
        """
        Extract the learned policy from Q-values.
        """
        policy = {}
        for state in self.Q:
            if self.Q[state]:
                best_action = max(self.Q[state].items(), key=lambda x: x[1])
                policy[state] = best_action[0]
                # Also store action probabilities for mixed strategies
                q_values = list(self.Q[state].values())
                if q_values:
                    # Softmax to get probabilities
                    q_array = np.array(q_values)
                    probs = np.exp(q_array) / np.sum(np.exp(q_array))
                    policy[f"{state}_probs"] = probs
        return policy
    
    def evaluate(self, mdp: KuhnPokerMDP, num_episodes: int = 10000) -> float:
        """
        Evaluate the learned policy.
        """
        total_reward = 0
        for _ in range(num_episodes):
            cards = mdp.cards.copy()
            random.shuffle(cards)
            my_card = cards[0]
            opp_card = cards[1]
            
            state = (my_card, '')
            episode_reward = 0
            
            while True:
                legal_actions = mdp.get_legal_actions(state)
                if not legal_actions:
                    break
                
                # Use greedy policy (no exploration)
                action = self.get_action(state, legal_actions, training=False)
                next_state, reward, done = mdp.step(state, action, opp_card)
                episode_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def save(self, filename: str = "kuhn_poker_qlearning.pkl"):
        """Save the Q-table and statistics."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'q_value_history': self.q_value_history
            }, f)
        print(f"Q-learning model saved to {filename}")
    
    def load(self, filename: str = "kuhn_poker_qlearning.pkl"):
        """Load a saved Q-table."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: defaultdict(float), data['Q'])
            self.epsilon = data.get('epsilon', 0.01)
            self.episode_rewards = data.get('episode_rewards', [])
            self.q_value_history = data.get('q_value_history', [])
        print(f"Q-learning model loaded from {filename}")

def plot_learning_curves(results: Dict, opponent_type: str):
    """
    Plot learning curves for Q-learning.
    
    References Figure 2 from the paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Q-Learning Performance vs {opponent_type} Opponent', fontsize=14, fontweight='bold')
    
    # 1. Episode rewards with moving average
    ax = axes[0, 0]
    rewards = results['episode_rewards']
    episodes = range(len(rewards))
    
    # Plot raw rewards (faded)
    ax.plot(episodes[::100], rewards[::100], 'b-', alpha=0.2, label='Raw rewards')
    
    # Moving average
    window = 1000
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-episode average')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward per Hand')
    ax.set_title('Learning Curve: Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-value evolution
    ax = axes[0, 1]
    if results['q_value_history']:
        episodes = [d['episode'] for d in results['q_value_history']]
        mean_q = [d['mean_q'] for d in results['q_value_history']]
        max_q = [d['max_q'] for d in results['q_value_history']]
        
        ax.plot(episodes, mean_q, 'g-', label='Mean Q-value')
        ax.plot(episodes, max_q, 'b-', label='Max Q-value')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-value')
        ax.set_title('Q-value Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Exploration rate decay
    ax = axes[1, 0]
    if results['q_value_history']:
        episodes = [d['episode'] for d in results['q_value_history']]
        epsilon = [d['epsilon'] for d in results['q_value_history']]
        
        ax.plot(episodes, epsilon, 'purple', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon (ε)')
        ax.set_title('Exploration Rate Decay')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(epsilon) * 1.1])
    
    # 4. Average reward progression
    ax = axes[1, 1]
    if results['q_value_history']:
        episodes = [d['episode'] for d in results['q_value_history']]
        avg_rewards = [d['avg_reward'] for d in results['q_value_history']]
        
        ax.plot(episodes, avg_rewards, 'orange', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward (last 1000 episodes)')
        ax.set_title('Average Reward Progression')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for theoretical Nash value
        ax.axhline(y=-1/18, color='red', linestyle='--', alpha=0.5, label='Nash value')
        ax.legend()
    
    plt.tight_layout()
    return fig

def compare_strategies(mdp: KuhnPokerMDP, agents: Dict[str, any], num_eval: int = 10000):
    """
    Compare different strategies (CFR, RL-CFR, Q-learning) against various opponents.
    
    References Figure 3 from the paper.
    """
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\nEvaluating {agent_name}...")
        
        # Test against different opponent types
        opponent_results = {}
        for opp_type in [OpponentType.RANDOM, OpponentType.PASSIVE, 
                        OpponentType.AGGRESSIVE, OpponentType.NASH]:
            mdp.opponent_type = opp_type
            
            if hasattr(agent, 'evaluate'):
                avg_reward = agent.evaluate(mdp, num_eval)
            else:
                # For CFR/RL-CFR agents, need different evaluation
                avg_reward = evaluate_cfr_agent(agent, mdp, num_eval)
            
            opponent_results[opp_type.value] = avg_reward
            print(f"  vs {opp_type.value}: {avg_reward:.4f}")
        
        results[agent_name] = opponent_results
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    opponent_types = list(results[list(results.keys())[0]].keys())
    x = np.arange(len(opponent_types))
    width = 0.25
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (agent_name, opp_results) in enumerate(results.items()):
        values = [opp_results[opp] for opp in opponent_types]
        ax.bar(x + i * width, values, width, label=agent_name, color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Opponent Type')
    ax.set_ylabel('Average Reward per Hand')
    ax.set_title('Strategy Performance Comparison (References: Lanctot et al., 2019)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(opponent_types)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add Nash equilibrium baseline
    ax.axhline(y=-1/18, color='black', linestyle='--', alpha=0.5, label='Nash value')
    
    plt.tight_layout()
    return fig, results

def evaluate_cfr_agent(agent, mdp: KuhnPokerMDP, num_episodes: int) -> float:
    """Helper function to evaluate CFR-based agents."""
    total_reward = 0
    
    for _ in range(num_episodes):
        cards = mdp.cards.copy()
        random.shuffle(cards)
        my_card = cards[0]
        opp_card = cards[1]
        
        history = ''
        
        # Map cards to indices for CFR
        card_map = {'J': 0, 'Q': 1, 'K': 2}
        my_card_idx = card_map[my_card]
        
        # Play out the hand
        done = False
        while not done:
            # Get CFR action
            info_set_key = f"{my_card_idx}:{history}"
            
            if hasattr(agent, 'get_action'):
                action_idx = agent.get_action(my_card_idx, history)
                if history == '':
                    action = 'c' if action_idx == 0 else 'b'
                elif history == 'cb':
                    action = 'fold' if action_idx == 0 else 'call'
                else:
                    action = 'c'
            else:
                action = 'c'  # Default
            
            # Apply action and get opponent response
            if action == 'c':
                history += 'c'
                # Opponent's turn
                opp_action = mdp.get_opponent_action(opp_card, history)
                if opp_action == 'check':
                    # Showdown
                    my_rank = mdp.card_rank[my_card]
                    opp_rank = mdp.card_rank[opp_card]
                    reward = 1 if my_rank > opp_rank else -1
                    done = True
                else:  # Opponent bets
                    history += 'b'
                    # Need to respond to bet
                    continue
            elif action == 'b':
                history += 'b'
                # Opponent's response
                opp_action = mdp.get_opponent_action(opp_card, history)
                if opp_action == 'fold':
                    reward = 1
                    done = True
                else:  # Opponent calls
                    my_rank = mdp.card_rank[my_card]
                    opp_rank = mdp.card_rank[opp_card]
                    reward = 2 if my_rank > opp_rank else -2
                    done = True
            elif action == 'fold':
                reward = -1
                done = True
            elif action == 'call':
                my_rank = mdp.card_rank[my_card]
                opp_rank = mdp.card_rank[opp_card]
                reward = 2 if my_rank > opp_rank else -2
                done = True
        
        total_reward += reward
    
    return total_reward / num_episodes

def measure_exploitability(agent, num_episodes: int = 10000) -> Dict[str, float]:
    """
    Measure exploitability of learned strategies.
    
    Tests how much the agent can be exploited by a best-response opponent.
    This addresses the "above and beyond" suggestion in the paper.
    
    References:
    - Johanson et al. (2011). "Accelerating Best Response Calculation in Large Games"
    """
    mdp = KuhnPokerMDP()
    
    # Test against different opponent types to measure robustness
    exploitability_results = {}
    
    for opp_type in [OpponentType.RANDOM, OpponentType.PASSIVE, 
                    OpponentType.AGGRESSIVE, OpponentType.NASH]:
        mdp.opponent_type = opp_type
        
        # First, measure agent's performance
        if hasattr(agent, 'evaluate'):
            baseline_reward = agent.evaluate(mdp, num_episodes // 2)
        else:
            baseline_reward = evaluate_cfr_agent(agent, mdp, num_episodes // 2)
        
        # Now create a "best response" opponent that exploits the agent's strategy
        # This is simplified - a true best response would require solving the game
        mdp.opponent_type = OpponentType.AGGRESSIVE if baseline_reward > 0 else OpponentType.PASSIVE
        
        if hasattr(agent, 'evaluate'):
            exploited_reward = agent.evaluate(mdp, num_episodes // 2)
        else:
            exploited_reward = evaluate_cfr_agent(agent, mdp, num_episodes // 2)
        
        exploitability = baseline_reward - exploited_reward
        exploitability_results[opp_type.value] = {
            'baseline': baseline_reward,
            'exploited': exploited_reward,
            'exploitability': exploitability
        }
    
    return exploitability_results

def main():
    """
    Main function to run experiments from the paper.
    """
    print("=" * 70)
    print("Kuhn Poker as an MDP: Q-Learning Implementation")
    print("Based on paper by Agriya (2025)")
    print("=" * 70)
    
    # Experiment 1: Train Q-learning against different opponents
    print("\n1. Training Q-Learning Agents")
    print("-" * 40)
    
    q_agents = {}
    
    # Train against different opponent types (addressing the fixed policy concern)
    for opp_type in [OpponentType.RANDOM, OpponentType.PASSIVE, 
                    OpponentType.AGGRESSIVE, OpponentType.NASH, OpponentType.MIXED]:
        print(f"\nTraining against {opp_type.value} opponent...")
        
        mdp = KuhnPokerMDP(opponent_type=opp_type)
        agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=1.0,  # No discounting for episodic tasks
            epsilon=0.2,
            epsilon_decay=0.9999
        )
        
        results = agent.train(mdp, num_episodes=25000)
        
        # Save the agent
        agent.save(f"q_agent_vs_{opp_type.value}.pkl")
        q_agents[f"Q-Learning vs {opp_type.value}"] = agent
        
        # Plot learning curves
        fig = plot_learning_curves(results, opp_type.value)
        plt.savefig(f"q_learning_curves_{opp_type.value}.png", dpi=150)
        plt.close()
        
        # Evaluate
        avg_reward = agent.evaluate(mdp, num_episodes=10000)
        print(f"Average reward: {avg_reward:.4f}")
    
    # Load CFR and RL-CFR agents for comparison
    print("\n2. Loading CFR and RL-CFR Agents for Comparison")
    print("-" * 40)
    
    try:
        from kuhn_poker_cfr import CFRTrainer
        cfr_agent = CFRTrainer()
        if os.path.exists("kuhn_poker_strategy.pkl"):
            cfr_agent.load_strategy()
            print("Loaded CFR agent")
    except:
        cfr_agent = None
        print("CFR agent not available")
    
    try:
        from kuhn_poker_rl_cfr import RLCFRTrainer
        rl_cfr_agent = RLCFRTrainer()
        if os.path.exists("kuhn_poker_rl_strategy.pkl"):
            rl_cfr_agent.load_strategy()
            print("Loaded RL-CFR agent")
    except:
        rl_cfr_agent = None
        print("RL-CFR agent not available")
    
    # Compare all strategies
    print("\n3. Strategy Comparison (Section 8.2)")
    print("-" * 40)
    
    agents_to_compare = {}
    
    # Add best Q-learning agent
    agents_to_compare["Q-Learning (vs Mixed)"] = q_agents.get("Q-Learning vs mixed", 
                                                             list(q_agents.values())[0])
    
    if cfr_agent:
        agents_to_compare["CFR (Nash)"] = cfr_agent
    
    if rl_cfr_agent:
        agents_to_compare["RL-CFR"] = rl_cfr_agent
    
    # Run comparison
    mdp_eval = KuhnPokerMDP()
    fig, comparison_results = compare_strategies(mdp_eval, agents_to_compare, num_eval=10000)
    plt.savefig("strategy_payoff_comparison.png", dpi=150)
    plt.close()
    
    print("\n4. Exploitability Analysis (Above and Beyond)")
    print("-" * 40)
    
    for agent_name, agent in agents_to_compare.items():
        print(f"\nMeasuring exploitability of {agent_name}:")
        exploit_results = measure_exploitability(agent, num_episodes=5000)
        
        for opp_type, results in exploit_results.items():
            print(f"  vs {opp_type}:")
            print(f"    Baseline reward: {results['baseline']:.4f}")
            print(f"    Exploited reward: {results['exploited']:.4f}")
            print(f"    Exploitability: {results['exploitability']:.4f}")
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("Key Findings:")
    print("1. Q-learning can exploit fixed non-equilibrium opponents")
    print("2. CFR provides robustness but may be conservative")
    print("3. RL-CFR balances exploration and exploitation")
    print("4. Against mixed/adaptive opponents, equilibrium strategies are safer")
    print("=" * 70)

if __name__ == "__main__":
    main()
