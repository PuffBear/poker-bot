"""
Comparison of Plain CFR vs RL-CFR for Kuhn Poker
Shows convergence rates, performance, and strategic differences
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import seaborn as sns
from kuhn_poker_cfr import CFRTrainer
from kuhn_poker_rl_cfr import RLCFRTrainer
import random

class CFRComparison:
    def __init__(self):
        self.plain_cfr = CFRTrainer()
        self.rl_cfr = RLCFRTrainer(
            learning_rate=0.1,
            discount_factor=0.999,
            epsilon=0.2,
            buffer_capacity=50000,
            batch_size=64
        )
        
        # Known Nash equilibrium values for comparison
        self.nash_equilibrium = {
            "0:": 1/3,      # Jack opening bet
            "1:": 0,        # Queen opening bet  
            "2:": 1,        # King opening bet
            "0:b": 0,       # Jack facing bet
            "1:b": 1/3,     # Queen facing bet
            "2:b": 1,       # King facing bet
            "0:p": 1/3,     # Jack after check
            "2:p": 1,       # King after check
        }
        
        # Tracking convergence
        self.plain_cfr_convergence = []
        self.rl_cfr_convergence = []
        self.plain_cfr_times = []
        self.rl_cfr_times = []
    
    def measure_convergence(self, trainer, is_rl=False) -> float:
        """Measure how close the strategy is to Nash equilibrium"""
        total_error = 0
        count = 0
        
        for info_set, nash_value in self.nash_equilibrium.items():
            if is_rl:
                strategy = trainer.get_average_strategy(info_set)
            else:
                if info_set in trainer.strategy_sum:
                    strategy = trainer.get_average_strategy(info_set)
                else:
                    continue
            
            learned_value = strategy[1]  # Bet/Call probability
            error = abs(learned_value - nash_value)
            total_error += error
            count += 1
        
        return total_error / count if count > 0 else 1.0
    
    def train_and_compare(self, iterations: int = 10000, checkpoint_interval: int = 100):
        """Train both algorithms and track convergence"""
        print("=" * 70)
        print("         COMPARING PLAIN CFR vs RL-CFR")
        print("=" * 70)
        
        # Reset trainers
        self.plain_cfr = CFRTrainer()
        self.rl_cfr = RLCFRTrainer(
            learning_rate=0.1,
            discount_factor=0.999,
            epsilon=0.2,
            buffer_capacity=50000,
            batch_size=64
        )
        
        # Train Plain CFR with checkpoints
        print("\n1. Training Plain CFR...")
        print("-" * 40)
        start_time = time.time()
        cards = [0, 1, 2]
        
        for i in range(iterations):
            random.shuffle(cards)
            dealt_cards = cards[:2]
            self.plain_cfr.cfr(dealt_cards, "", 1.0, 1.0)
            
            if (i + 1) % checkpoint_interval == 0:
                convergence = self.measure_convergence(self.plain_cfr, is_rl=False)
                elapsed = time.time() - start_time
                self.plain_cfr_convergence.append(convergence)
                self.plain_cfr_times.append(elapsed)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Iteration {i+1}: Error = {convergence:.4f}, Time = {elapsed:.2f}s")
        
        plain_cfr_time = time.time() - start_time
        
        # Train RL-CFR with checkpoints
        print("\n2. Training RL-CFR...")
        print("-" * 40)
        start_time = time.time()
        
        for i in range(iterations):
            self.rl_cfr.iteration_count = i
            
            # Decay learning rate and epsilon
            self.rl_cfr.learning_rate = self.rl_cfr.initial_lr * (self.rl_cfr.lr_decay ** i)
            self.rl_cfr.epsilon = max(0.01, self.rl_cfr.epsilon * 0.9999)
            
            # Sample and train
            random.shuffle(cards)
            dealt_cards = cards[:2]
            sampling_player = i % 2
            self.rl_cfr.external_sampling_cfr(dealt_cards, "", 1.0, 1.0, sampling_player)
            
            # Replay updates
            if i % 10 == 0 and i > 0:
                self.rl_cfr.replay_update()
            
            if (i + 1) % checkpoint_interval == 0:
                convergence = self.measure_convergence(self.rl_cfr, is_rl=True)
                elapsed = time.time() - start_time
                self.rl_cfr_convergence.append(convergence)
                self.rl_cfr_times.append(elapsed)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Iteration {i+1}: Error = {convergence:.4f}, Time = {elapsed:.2f}s, "
                          f"LR = {self.rl_cfr.learning_rate:.6f}, ε = {self.rl_cfr.epsilon:.4f}")
        
        rl_cfr_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 70)
        print("                    TRAINING SUMMARY")
        print("=" * 70)
        print(f"\nPlain CFR:")
        print(f"  Total time: {plain_cfr_time:.2f}s")
        print(f"  Final error: {self.plain_cfr_convergence[-1]:.4f}")
        print(f"  Iterations/second: {iterations/plain_cfr_time:.1f}")
        
        print(f"\nRL-CFR:")
        print(f"  Total time: {rl_cfr_time:.2f}s")
        print(f"  Final error: {self.rl_cfr_convergence[-1]:.4f}")
        print(f"  Iterations/second: {iterations/rl_cfr_time:.1f}")
        print(f"  Replay buffer size: {len(self.rl_cfr.replay_buffer.buffer)}")
        
        speedup = plain_cfr_time / rl_cfr_time
        print(f"\nSpeedup: RL-CFR is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    def visualize_comparison(self):
        """Create comparison visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Plain CFR vs RL-CFR Comparison', fontsize=16, fontweight='bold')
        
        # 1. Convergence over iterations
        ax = axes[0, 0]
        iterations = np.arange(len(self.plain_cfr_convergence)) * 100
        ax.plot(iterations, self.plain_cfr_convergence, 'b-', label='Plain CFR', linewidth=2)
        ax.plot(iterations, self.rl_cfr_convergence, 'r-', label='RL-CFR', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Average Error from Nash')
        ax.set_title('Convergence Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. Convergence over time
        ax = axes[0, 1]
        ax.plot(self.plain_cfr_times, self.plain_cfr_convergence, 'b-', label='Plain CFR', linewidth=2)
        ax.plot(self.rl_cfr_times, self.rl_cfr_convergence, 'r-', label='RL-CFR', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Average Error from Nash')
        ax.set_title('Convergence over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. Strategy comparison heatmap
        ax = axes[0, 2]
        strategies_plain = []
        strategies_rl = []
        labels = []
        
        for info_set in ["0:", "1:", "2:", "0:b", "1:b", "2:b"]:
            # Plain CFR strategy
            if info_set in self.plain_cfr.strategy_sum:
                plain_strategy = self.plain_cfr.get_average_strategy(info_set)[1]
            else:
                plain_strategy = 0.5
            strategies_plain.append(plain_strategy)
            
            # RL-CFR strategy
            rl_strategy = self.rl_cfr.get_average_strategy(info_set)[1]
            strategies_rl.append(rl_strategy)
            
            # Nash equilibrium
            nash = self.nash_equilibrium.get(info_set, 0.5)
            
            # Labels
            card = ['J', 'Q', 'K'][int(info_set[0])]
            action = info_set[2:] if len(info_set) > 2 else "open"
            labels.append(f"{card}:{action}")
        
        # Create comparison matrix
        comparison_matrix = np.array([strategies_plain, strategies_rl])
        im = ax.imshow(comparison_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Plain CFR', 'RL-CFR'])
        ax.set_title('Strategy Comparison (Bet/Call Prob)')
        
        # Add values to heatmap
        for i in range(2):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{comparison_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
        
        # 4. Error distribution
        ax = axes[1, 0]
        plain_errors = []
        rl_errors = []
        
        for info_set, nash_value in self.nash_equilibrium.items():
            if info_set in self.plain_cfr.strategy_sum:
                plain_strategy = self.plain_cfr.get_average_strategy(info_set)[1]
                plain_errors.append(abs(plain_strategy - nash_value))
            
            rl_strategy = self.rl_cfr.get_average_strategy(info_set)[1]
            rl_errors.append(abs(rl_strategy - nash_value))
        
        x = np.arange(len(self.nash_equilibrium))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, plain_errors, width, label='Plain CFR', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, rl_errors, width, label='RL-CFR', color='red', alpha=0.7)
        
        ax.set_xlabel('Information Sets')
        ax.set_ylabel('Error from Nash')
        ax.set_title('Error Distribution by Information Set')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{['J','Q','K'][int(k[0])]}:{k[2:]}" for k in self.nash_equilibrium.keys()], 
                           rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 5. Learning metrics (RL-CFR specific)
        ax = axes[1, 1]
        iterations = np.arange(len(self.rl_cfr_convergence)) * 100
        
        # Create twin axis for learning rate
        ax2 = ax.twinx()
        
        # Plot convergence on left axis
        line1 = ax.plot(iterations, self.rl_cfr_convergence, 'b-', label='Convergence', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error from Nash', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot learning rate on right axis
        lr_values = [self.rl_cfr.initial_lr * (self.rl_cfr.lr_decay ** i) 
                    for i in range(0, len(iterations) * 100, 100)]
        line2 = ax2.plot(iterations, lr_values, 'r--', label='Learning Rate', linewidth=1)
        ax2.set_ylabel('Learning Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('RL-CFR Learning Dynamics')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        # 6. Performance comparison table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        # Create comparison data
        data = [
            ['Metric', 'Plain CFR', 'RL-CFR'],
            ['Final Error', f'{self.plain_cfr_convergence[-1]:.4f}', f'{self.rl_cfr_convergence[-1]:.4f}'],
            ['Training Time', f'{self.plain_cfr_times[-1]:.2f}s', f'{self.rl_cfr_times[-1]:.2f}s'],
            ['Memory Usage', 'Full tree', f'Buffer: {len(self.rl_cfr.replay_buffer.buffer)}'],
            ['Sampling', 'Full traversal', 'Monte Carlo'],
            ['Updates', 'Immediate', 'Batch replay'],
            ['Exploration', 'None', 'ε-greedy'],
        ]
        
        table = ax.table(cellText=data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Algorithm Comparison', pad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cfr_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_strategy_comparison(self):
        """Print detailed strategy comparison"""
        print("\n" + "=" * 70)
        print("              DETAILED STRATEGY COMPARISON")
        print("=" * 70)
        
        card_names = ['Jack', 'Queen', 'King']
        scenarios = [
            ("", "Opening"),
            ("p", "After check"),
            ("b", "Facing bet"),
            ("pb", "After check, facing bet")
        ]
        
        for card in range(3):
            print(f"\n{card_names[card]}:")
            print("-" * 50)
            print(f"{'Scenario':<20} {'Plain CFR':<15} {'RL-CFR':<15} {'Nash':<15}")
            print("-" * 50)
            
            for history, desc in scenarios:
                info_set = f"{card}:{history}"
                
                # Get strategies
                if info_set in self.plain_cfr.strategy_sum:
                    plain_strategy = self.plain_cfr.get_average_strategy(info_set)[1]
                else:
                    plain_strategy = 0.5
                
                rl_strategy = self.rl_cfr.get_average_strategy(info_set)[1]
                
                # Get Nash value if known
                nash_value = self.nash_equilibrium.get(info_set, -1)
                nash_str = f"{nash_value:.3f}" if nash_value >= 0 else "N/A"
                
                print(f"{desc:<20} {plain_strategy:<15.3f} {rl_strategy:<15.3f} {nash_str:<15}")

def main():
    print("Kuhn Poker: Comparing Plain CFR vs RL-CFR")
    print("This will train both algorithms and compare their performance.\n")
    
    comparison = CFRComparison()
    
    # Train and compare
    comparison.train_and_compare(iterations=10000, checkpoint_interval=100)
    
    # Print strategy comparison
    comparison.print_strategy_comparison()
    
    # Visualize results
    print("\nGenerating comparison visualizations...")
    comparison.visualize_comparison()
    print("Visualization saved as 'cfr_comparison.png'")

if __name__ == "__main__":
    main()
