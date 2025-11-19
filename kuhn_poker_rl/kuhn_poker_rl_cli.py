"""
Kuhn Poker RL-CFR CLI - Play against the RL-CFR trained bot in terminal
"""

import random
import os
from kuhn_poker_rl_cfr import RLCFRTrainer, KuhnPoker
import time

class KuhnPokerRLCLI:
    def __init__(self):
        self.game = KuhnPoker()
        self.trainer = RLCFRTrainer()
        self.load_or_train_bot()
        
        # Stats
        self.player_score = 0
        self.bot_score = 0
        self.games_played = 0
        
        # Card display
        self.card_symbols = {
            0: '🃏 Jack',
            1: '👸 Queen', 
            2: '👑 King'
        }
    
    def load_or_train_bot(self):
        """Load existing strategy or train new one"""
        if os.path.exists("kuhn_poker_rl_strategy.pkl"):
            print("Loading existing RL-CFR strategy...")
            self.trainer.load_strategy()
            print(f"Loaded strategy trained for {self.trainer.iteration_count} iterations")
        else:
            print("No saved RL-CFR strategy found. Training bot with RL-CFR...")
            print("This uses Monte Carlo sampling and experience replay...")
            self.trainer.train(iterations=30000)
            self.trainer.save_strategy()
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_game_state(self, player_card, bot_card_hidden, history, pot, current_player, bot_card=None):
        """Display the current game state"""
        self.clear_screen()
        print("=" * 60)
        print("         KUHN POKER vs RL-CFR Bot")
        print("    (Reinforcement Learning CFR with Experience Replay)")
        print("=" * 60)
        print(f"Score -> You: {self.player_score} | Bot: {self.bot_score} | Games: {self.games_played}")
        print("-" * 60)
        
        # Bot's card (hidden or revealed)
        if bot_card_hidden:
            print(f"RL-CFR Bot's card: [Hidden]")
        else:
            print(f"RL-CFR Bot's card: {self.card_symbols[bot_card]}")
            # Show bot's strategy for this position
            if bot_card is not None and history:
                info_set_key = f"{bot_card}:{history}"
                strategy = self.trainer.get_average_strategy(info_set_key)
                print(f"Bot's strategy: [Pass/Fold: {strategy[0]:.2%}, Bet/Call: {strategy[1]:.2%}]")
        
        print()
        print(f"         POT: {pot} chips")
        print()
        
        # Your card
        print(f"Your card: {self.card_symbols[player_card]}")
        print()
        
        # History
        if history:
            self.display_history(history, current_player)
        print("-" * 60)
    
    def display_history(self, history, first_player):
        """Display action history in readable format"""
        print("Actions: ", end="")
        for i, action in enumerate(history):
            if i > 0:
                print(" → ", end="")
            
            # Determine who made this action
            player_idx = (first_player + i) % 2
            player = "You" if player_idx == 0 else "Bot"
            
            # Determine action name based on context
            if action == 'p':
                if i > 0 and history[i-1] == 'b':
                    action_name = "Fold"
                else:
                    action_name = "Check"
            else:  # action == 'b'
                if i > 0 and 'b' in history[:i]:
                    action_name = "Call"
                else:
                    action_name = "Bet"
            
            print(f"{player}: {action_name}", end="")
        print()
    
    def get_player_action(self, history):
        """Get player's action choice"""
        print("\nYour turn!")
        
        # Determine available actions based on history
        if 'b' in history:
            # Facing a bet
            print("1. Fold (give up the pot)")
            print("2. Call (match the bet)")
            while True:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice == '1':
                    return 'p'  # Fold
                elif choice == '2':
                    return 'b'  # Call
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            # Can check or bet
            print("1. Check (pass the action)")
            print("2. Bet (bet 1 chip)")
            while True:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice == '1':
                    return 'p'  # Check
                elif choice == '2':
                    return 'b'  # Bet
                else:
                    print("Invalid choice. Please enter 1 or 2.")
    
    def play_game(self):
        """Play a single game of Kuhn Poker"""
        # Reset game state
        cards = [0, 1, 2]
        random.shuffle(cards)
        player_card = cards[0]
        bot_card = cards[1]
        
        history = ""
        pot = 2  # Both players ante 1
        
        # Randomly determine who goes first
        current_player = random.randint(0, 1)
        first_player = current_player
        
        if current_player == 0:
            print("\nYou act first!")
        else:
            print("\nRL-CFR Bot acts first!")
        
        input("\nPress Enter to start the game...")
        
        # Game loop
        while not self.game.is_terminal(history):
            # Display current state
            self.display_game_state(player_card, True, history, pot, first_player)
            
            if current_player == 0:
                # Player's turn
                action = self.get_player_action(history)
                history += action
                if action == 'b':
                    pot += 1
            else:
                # Bot's turn
                print("\nRL-CFR Bot is computing (using learned strategy)...")
                
                # Show bot's decision process
                info_set_key = f"{bot_card}:{history}"
                strategy = self.trainer.get_average_strategy(info_set_key)
                print(f"Bot's probabilities: Pass/Fold={strategy[0]:.2%}, Bet/Call={strategy[1]:.2%}")
                
                action_idx = self.trainer.get_action(bot_card, history)
                action = 'p' if action_idx == 0 else 'b'
                history += action
                if action == 'b':
                    pot += 1
                
                # Show bot's action
                if action == 'p' and 'b' in history[:-1]:
                    print("Bot folds!")
                elif action == 'p':
                    print("Bot checks!")
                elif action == 'b' and 'b' in history[:-1]:
                    print("Bot calls!")
                else:
                    print("Bot bets!")
                
                time.sleep(1)  # Brief pause for readability
                input("\nPress Enter to continue...")
            
            # Switch players
            current_player = 1 - current_player
        
        # Game over - determine winner
        self.display_game_state(player_card, False, history, pot, first_player, bot_card)
        
        # Calculate payoff (from player 1's perspective)
        if first_player == 0:
            payoff = self.game.get_payoff([player_card, bot_card], history)
        else:
            payoff = -self.game.get_payoff([bot_card, player_card], history)
        
        # Update scores and display result
        print("\n" + "=" * 60)
        if payoff > 0:
            print(f"YOU WIN! You gain {payoff} chips!")
            self.player_score += payoff
        else:
            print(f"RL-CFR Bot wins! You lose {abs(payoff)} chips.")
            self.bot_score += abs(payoff)
        print("=" * 60)
        
        self.games_played += 1
    
    def view_statistics(self):
        """Display game statistics"""
        self.clear_screen()
        print("=" * 60)
        print("                  STATISTICS")
        print("=" * 60)
        print(f"Games played: {self.games_played}")
        print(f"Your total score: {self.player_score}")
        print(f"Bot's total score: {self.bot_score}")
        
        if self.games_played > 0:
            player_avg = self.player_score / self.games_played
            bot_avg = self.bot_score / self.games_played
            print(f"\nYour average: {player_avg:.3f} chips per game")
            print(f"Bot's average: {bot_avg:.3f} chips per game")
            
            if player_avg > 0:
                print("\nYou're beating the Nash equilibrium!")
            elif player_avg < -0.1:
                print("\nThe RL-CFR bot is exploiting your play style.")
            else:
                print("\nYou're playing close to optimal!")
        
        print("\n" + "-" * 60)
        print("RL-CFR Training Info:")
        print(f"  Iterations trained: {self.trainer.iteration_count}")
        print(f"  Final learning rate: {self.trainer.learning_rate:.6f}")
        print(f"  Final epsilon: {self.trainer.epsilon:.4f}")
        print(f"  Replay buffer size: {len(self.trainer.replay_buffer.buffer)}")
        
        input("\nPress Enter to continue...")
    
    def view_strategy(self):
        """Display the bot's strategy"""
        self.clear_screen()
        print("=" * 60)
        print("        RL-CFR BOT'S STRATEGY (Learned Nash Equilibrium)")
        print("=" * 60)
        
        card_names = ['Jack', 'Queen', 'King']
        
        for card in range(3):
            print(f"\n{card_names[card]}:")
            
            states = [
                ("", "Opening", "Check", "Bet"),
                ("p", "After check", "Check", "Bet"),
                ("b", "Facing bet", "Fold", "Call"),
                ("pb", "After check, facing bet", "Fold", "Call")
            ]
            
            for history, desc, action0, action1 in states:
                info_set_key = f"{card}:{history}"
                strategy = self.trainer.get_average_strategy(info_set_key)
                print(f"  {desc:20} -> {action0}: {strategy[0]:.1%}, {action1}: {strategy[1]:.1%}")
        
        print("\n" + "-" * 60)
        print("Key insights (Nash Equilibrium):")
        print("- Jack: Bluffs ~33% opening, always folds to bets")
        print("- Queen: Never opens with bet, calls ~33% when facing bets")
        print("- King: Usually bets for value, always calls when facing bets")
        
        print("\n" + "-" * 60)
        print("RL-CFR Features Used:")
        print("- External sampling (Monte Carlo)")
        print("- Experience replay buffer")
        print("- Epsilon-greedy exploration")
        print("- Learning rate decay")
        print("- Value function bootstrapping")
        
        input("\nPress Enter to continue...")
    
    def compare_strategies(self):
        """Compare RL-CFR with theoretical Nash"""
        self.clear_screen()
        self.trainer.compare_with_nash()
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main game loop"""
        self.clear_screen()
        print("=" * 60)
        print("    Welcome to Kuhn Poker vs RL-CFR Bot!")
        print("=" * 60)
        print("\nThis bot uses Reinforcement Learning CFR:")
        print("• Monte Carlo sampling (not full tree traversal)")
        print("• Experience replay from past games")
        print("• Exploration-exploitation balance")
        print("• Learning rate adaptation")
        print("\nRules:")
        print("- 3 cards: Jack < Queen < King")
        print("- Each player antes 1 chip")
        print("- Players can check/bet, then fold/call")
        print("- Higher card wins at showdown")
        
        while True:
            print("\n" + "-" * 60)
            print("1. Play a game")
            print("2. View statistics")
            print("3. View bot's strategy")
            print("4. Compare with Nash equilibrium")
            print("5. Quit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.play_game()
            elif choice == '2':
                self.view_statistics()
            elif choice == '3':
                self.view_strategy()
            elif choice == '4':
                self.compare_strategies()
            elif choice == '5':
                print("\nThanks for playing against the RL-CFR bot!")
                break
            else:
                print("Invalid choice. Please enter 1-5.")

def main():
    game = KuhnPokerRLCLI()
    game.run()

if __name__ == "__main__":
    main()
