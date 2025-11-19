"""
Kuhn Poker Strategy Analysis and Visualization
Analyzes and visualizes the Nash Equilibrium strategy learned by CFR
"""

import matplotlib.pyplot as plt
import numpy as np
from kuhn_poker_cfr import CFRTrainer, KuhnPoker
import seaborn as sns

class StrategyAnalyzer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.game = KuhnPoker()
        
    def visualize_strategy(self):
        """Create visualizations of the learned strategy"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Kuhn Poker Nash Equilibrium Strategy (CFR)', fontsize=16, fontweight='bold')
        
        card_names = ['Jack', 'Queen', 'King']
        
        # Define all possible information sets
        scenarios = [
            ("", "Opening Action"),
            ("p", "After Check"),
            ("b", "Facing Opening Bet"),
            ("pb", "Facing Bet After Check")
        ]
        
        # Collect strategy data
        strategy_data = []
        
        for card in range(3):
            card_strategies = []
            for history, _ in scenarios:
                info_set_key = f"{card}:{history}"
                if info_set_key in self.trainer.strategy_sum:
                    strategy = self.trainer.get_average_strategy(info_set_key)
                    card_strategies.append(strategy[1])  # Probability of betting/calling
                else:
                    card_strategies.append(0.0)
            strategy_data.append(card_strategies)
        
        # Create heatmap in the first subplot
        ax = axes[0, 0]
        sns.heatmap(strategy_data, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   xticklabels=[s[1] for s in scenarios[:4]],
                   yticklabels=card_names,
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'P(Bet/Call)'},
                   ax=ax)
        ax.set_title('Strategy Heatmap')
        ax.set_xlabel('Game State')
        ax.set_ylabel('Card')
        
        # Individual card strategies
        for i, card in enumerate(range(3)):
            row = (i + 1) // 3
            col = (i + 1) % 3
            ax = axes[row, col + (1 if row == 0 else 0)]
            
            # Collect data for this card
            bet_probs = []
            pass_probs = []
            labels = []
            
            for history, description in scenarios:
                info_set_key = f"{card}:{history}"
                if info_set_key in self.trainer.strategy_sum:
                    strategy = self.trainer.get_average_strategy(info_set_key)
                    bet_probs.append(strategy[1])
                    pass_probs.append(strategy[0])
                    labels.append(description.split()[0] if len(description.split()) > 0 else "Start")
            
            if bet_probs:
                x = np.arange(len(labels))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, pass_probs, width, label='Pass/Check/Fold', color='#e74c3c', alpha=0.8)
                bars2 = ax.bar(x + width/2, bet_probs, width, label='Bet/Call', color='#3498db', alpha=0.8)
                
                ax.set_ylabel('Probability')
                ax.set_title(f'{card_names[card]} Strategy')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_ylim([0, 1])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0.05:
                        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                               f'{height:.2f}', ha='center', va='center', fontsize=8)
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0.05:
                        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                               f'{height:.2f}', ha='center', va='center', fontsize=8)
        
        # Expected value analysis
        ax = axes[1, 2]
        self.plot_expected_values(ax)
        
        plt.tight_layout()
        plt.savefig('kuhn_poker_strategy.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_expected_values(self, ax):
        """Plot expected values for different starting hands"""
        card_names = ['Jack', 'Queen', 'King']
        expected_values = []
        
        # Simulate many games to estimate expected value
        n_simulations = 10000
        
        for player_card in range(3):
            total_value = 0
            for _ in range(n_simulations):
                # Random opponent card
                possible_cards = [c for c in range(3) if c != player_card]
                bot_card = np.random.choice(possible_cards)
                
                # Simulate a game using the strategy
                history = ""
                cards = [player_card, bot_card]
                
                while not self.game.is_terminal(history):
                    current_player = self.game.get_current_player(history)
                    info_set_key = f"{cards[current_player]}:{history}"
                    strategy = self.trainer.get_average_strategy(info_set_key)
                    action = np.random.choice(2, p=strategy)
                    history += 'p' if action == 0 else 'b'
                
                # Get payoff
                payoff = self.game.get_payoff(cards, history)
                total_value += payoff
            
            expected_values.append(total_value / n_simulations)
        
        # Plot
        bars = ax.bar(card_names, expected_values, color=['#e74c3c', '#f39c12', '#27ae60'], alpha=0.8)
        ax.set_ylabel('Expected Value')
        ax.set_title('Expected Value by Card')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim([-0.2, 0.2])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, expected_values):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01 if val > 0 else val - 0.01,
                   f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    def print_detailed_analysis(self):
        """Print detailed analysis of the strategy"""
        print("\n" + "="*60)
        print("DETAILED STRATEGY ANALYSIS")
        print("="*60)
        
        print("\nOptimal Strategy Summary:")
        print("-" * 40)
        
        # Known Nash equilibrium for Kuhn Poker
        print("\n1. JACK (Weakest card):")
        print("   - Opening: Mostly check, occasional bluff bet")
        print("   - Facing bet: Usually fold")
        print("   - After opponent checks: Mixed strategy")
        
        print("\n2. QUEEN (Middle card):")
        print("   - Opening: Check")
        print("   - Facing bet: Mixed call/fold")
        print("   - After opponent checks: Check (trap)")
        
        print("\n3. KING (Strongest card):")
        print("   - Opening: Mixed bet/check")
        print("   - Facing bet: Always call")
        print("   - After opponent checks: Always bet")
        
        print("\n" + "-"*40)
        print("Learned CFR Strategy:")
        print("-" * 40)
        
        card_names = ['Jack', 'Queen', 'King']
        scenarios = [
            ("", "Opening"),
            ("p", "After check"),
            ("b", "Facing bet"),
            ("pb", "After check, facing bet")
        ]
        
        for card in range(3):
            print(f"\n{card_names[card]}:")
            for history, description in scenarios:
                info_set_key = f"{card}:{history}"
                if info_set_key in self.trainer.strategy_sum:
                    strategy = self.trainer.get_average_strategy(info_set_key)
                    print(f"  {description:25} -> Bet/Call: {strategy[1]:.1%}, Pass/Fold: {strategy[0]:.1%}")
    
    def verify_nash_equilibrium(self):
        """Verify that the learned strategy approximates Nash equilibrium"""
        print("\n" + "="*60)
        print("NASH EQUILIBRIUM VERIFICATION")
        print("="*60)
        
        # Known Nash equilibrium frequencies for Kuhn Poker
        nash_benchmarks = {
            "0:": 0.33,      # Jack opening bet (bluff)
            "1:": 0.0,       # Queen opening bet (never)
            "2:": 0.67,      # King opening bet (value)
            "0:b": 0.0,      # Jack facing bet (never call)
            "1:b": 0.33,     # Queen facing bet (sometimes call)
            "2:b": 1.0,      # King facing bet (always call)
            "0:pb": 0.0,     # Jack after check facing bet (never call)
            "1:pb": 0.33,    # Queen after check facing bet (sometimes call)
            "2:pb": 1.0,     # King after check facing bet (always call)
        }
        
        print("\nComparing learned strategy to theoretical Nash equilibrium:")
        print("-" * 50)
        print(f"{'Info Set':<15} {'Learned':<12} {'Nash':<12} {'Difference':<12}")
        print("-" * 50)
        
        total_error = 0
        for info_set, nash_value in nash_benchmarks.items():
            if info_set in self.trainer.strategy_sum:
                learned = self.trainer.get_average_strategy(info_set)[1]  # Bet/Call probability
                diff = abs(learned - nash_value)
                total_error += diff
                
                card = ['J', 'Q', 'K'][int(info_set[0])]
                history = info_set[2:] if len(info_set) > 2 else "start"
                
                print(f"{card + ':' + history:<15} {learned:.3f}        {nash_value:.3f}        {diff:.3f}")
        
        avg_error = total_error / len(nash_benchmarks)
        print("-" * 50)
        print(f"Average deviation from Nash: {avg_error:.4f}")
        
        if avg_error < 0.05:
            print("✓ Excellent convergence to Nash equilibrium!")
        elif avg_error < 0.1:
            print("✓ Good convergence to Nash equilibrium.")
        else:
            print("⚠ Strategy may need more training iterations.")

def main():
    """Run the analysis"""
    print("Loading trained strategy...")
    trainer = CFRTrainer()
    
    # Train if no saved strategy exists
    import os
    if not os.path.exists("kuhn_poker_strategy.pkl"):
        print("No saved strategy found. Training...")
        trainer.train(iterations=100000)
        trainer.save_strategy()
    else:
        trainer.load_strategy()
    
    # Analyze
    analyzer = StrategyAnalyzer(trainer)
    analyzer.print_detailed_analysis()
    analyzer.verify_nash_equilibrium()
    
    print("\nGenerating strategy visualization...")
    analyzer.visualize_strategy()
    print("Visualization saved as 'kuhn_poker_strategy.png'")

if __name__ == "__main__":
    main()
