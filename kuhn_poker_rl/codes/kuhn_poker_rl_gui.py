"""
Kuhn Poker GUI - Play against the RL-CFR trained bot
Updated to use Reinforcement Learning CFR
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import os
from kuhn_poker_rl_cfr import RLCFRTrainer, KuhnPoker
import threading
import time

class KuhnPokerRLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kuhn Poker - Play vs RL-CFR Bot")
        self.root.geometry("900x750")
        self.root.configure(bg='#2c3e50')
        
        # Game components
        self.game = KuhnPoker()
        self.trainer = RLCFRTrainer()
        self.load_or_train_bot()
        
        # Game state
        self.reset_game_state()
        
        # Stats
        self.player_score = 0
        self.bot_score = 0
        self.games_played = 0
        
        # Create GUI
        self.create_widgets()
        self.new_game()
    
    def reset_game_state(self):
        """Reset the game state for a new round"""
        self.cards = [0, 1, 2]  # J, Q, K
        self.player_card = None
        self.bot_card = None
        self.history = ""
        self.pot = 2  # Both players ante 1
        self.player_chips = 0
        self.bot_chips = 0
        self.game_over = False
        self.current_player = 0  # 0 for human, 1 for bot
    
    def load_or_train_bot(self):
        """Load existing strategy or train new one"""
        if os.path.exists("kuhn_poker_rl_strategy.pkl"):
            print("Loading existing RL-CFR strategy...")
            self.trainer.load_strategy()
        else:
            print("No saved RL-CFR strategy found. Training bot with RL-CFR...")
            messagebox.showinfo("Training", "Training the RL-CFR bot... This may take a moment.")
            self.trainer.train(iterations=30000)
            self.trainer.save_strategy()
    
    def create_widgets(self):
        """Create all GUI elements"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=15)
        
        title = tk.Label(title_frame, text="♠ KUHN POKER ♥", 
                        font=('Arial Bold', 28), bg='#2c3e50', fg='white')
        title.pack()
        
        subtitle = tk.Label(title_frame, text="vs RL-CFR Bot (Reinforcement Learning CFR)", 
                           font=('Arial', 12), bg='#2c3e50', fg='#95a5a6')
        subtitle.pack()
        
        # Score Frame
        score_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        score_frame.pack(pady=10)
        
        self.score_label = tk.Label(score_frame, 
                                   text=f"Player: {self.player_score}  |  Bot: {self.bot_score}  |  Games: {self.games_played}",
                                   font=('Arial', 14), bg='#34495e', fg='white', padx=20, pady=10)
        self.score_label.pack()
        
        # Game Area
        game_frame = tk.Frame(self.root, bg='#27ae60', relief=tk.SUNKEN, bd=3)
        game_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Bot Area
        bot_frame = tk.Frame(game_frame, bg='#27ae60')
        bot_frame.pack(pady=20)
        
        tk.Label(bot_frame, text="RL-CFR Bot", font=('Arial Bold', 16), 
                bg='#27ae60', fg='white').pack()
        
        self.bot_card_label = tk.Label(bot_frame, text="🎴", font=('Arial', 60), 
                                      bg='#27ae60', fg='white')
        self.bot_card_label.pack(pady=10)
        
        self.bot_status = tk.Label(bot_frame, text="Waiting...", 
                                  font=('Arial', 12), bg='#27ae60', fg='yellow')
        self.bot_status.pack()
        
        # Strategy display for bot
        self.bot_strategy_label = tk.Label(bot_frame, text="", 
                                          font=('Courier', 10), bg='#27ae60', fg='#ecf0f1')
        self.bot_strategy_label.pack()
        
        # Pot Display
        self.pot_label = tk.Label(game_frame, text=f"POT: {self.pot} chips", 
                                 font=('Arial Bold', 20), bg='#27ae60', fg='gold')
        self.pot_label.pack(pady=20)
        
        # History Display
        self.history_label = tk.Label(game_frame, text="Actions: ", 
                                     font=('Arial', 12), bg='#27ae60', fg='white')
        self.history_label.pack()
        
        # Player Area
        player_frame = tk.Frame(game_frame, bg='#27ae60')
        player_frame.pack(pady=20)
        
        tk.Label(player_frame, text="You", font=('Arial Bold', 16), 
                bg='#27ae60', fg='white').pack()
        
        self.player_card_label = tk.Label(player_frame, text="🎴", font=('Arial', 60), 
                                         bg='#27ae60', fg='white')
        self.player_card_label.pack(pady=10)
        
        self.player_status = tk.Label(player_frame, text="", 
                                     font=('Arial', 12), bg='#27ae60', fg='yellow')
        self.player_status.pack()
        
        # Action Buttons
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        self.check_button = tk.Button(button_frame, text="CHECK/FOLD", 
                                     font=('Arial Bold', 14), width=12, height=2,
                                     bg='#e74c3c', fg='white', 
                                     command=lambda: self.player_action('p'))
        self.check_button.pack(side=tk.LEFT, padx=10)
        
        self.bet_button = tk.Button(button_frame, text="BET/CALL", 
                                   font=('Arial Bold', 14), width=12, height=2,
                                   bg='#3498db', fg='white',
                                   command=lambda: self.player_action('b'))
        self.bet_button.pack(side=tk.LEFT, padx=10)
        
        self.new_game_button = tk.Button(button_frame, text="NEW GAME", 
                                        font=('Arial Bold', 14), width=12, height=2,
                                        bg='#95a5a6', fg='white',
                                        command=self.new_game)
        self.new_game_button.pack(side=tk.LEFT, padx=10)
        
        # Info Panel
        info_frame = tk.Frame(self.root, bg='#2c3e50')
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = """RL-CFR: Uses Monte Carlo sampling, experience replay, and exploration.
        Rules: Jack < Queen < King. Check/Bet first, then Fold/Call facing bets."""
        
        tk.Label(info_frame, text=info_text, font=('Arial', 10), 
                bg='#2c3e50', fg='#ecf0f1', wraplength=850).pack()
        
        # Training info
        self.training_info = tk.Label(info_frame, text="", font=('Courier', 9), 
                                     bg='#2c3e50', fg='#95a5a6')
        self.training_info.pack()
        self.update_training_info()
    
    def update_training_info(self):
        """Display training information"""
        info = (f"Bot trained with {self.trainer.iteration_count} iterations | "
                f"Learning rate: {self.trainer.learning_rate:.6f} | "
                f"Buffer size: {len(self.trainer.replay_buffer.buffer)}")
        self.training_info.config(text=info)
    
    def show_bot_strategy(self):
        """Display the bot's strategy for its current card"""
        if self.bot_card is not None:
            info_set_key = f"{self.bot_card}:{self.history}"
            strategy = self.trainer.get_average_strategy(info_set_key)
            
            # Determine action names based on context
            if 'b' in self.history:
                actions = "Fold/Call"
            else:
                actions = "Check/Bet"
            
            strategy_text = f"Strategy: {actions} = [{strategy[0]:.2f}, {strategy[1]:.2f}]"
            self.bot_strategy_label.config(text=strategy_text)
    
    def new_game(self):
        """Start a new game"""
        self.reset_game_state()
        
        # Shuffle and deal cards
        random.shuffle(self.cards)
        self.player_card = self.cards[0]
        self.bot_card = self.cards[1]
        
        # Update display
        card_names = ['J', 'Q', 'K']
        card_symbols = ['J♠', 'Q♦', 'K♣']
        
        self.player_card_label.config(text=card_symbols[self.player_card], 
                                     fg=['#3498db', '#e74c3c', '#f39c12'][self.player_card])
        self.bot_card_label.config(text="🎴")
        self.bot_strategy_label.config(text="")
        
        self.pot_label.config(text=f"POT: {self.pot} chips")
        self.history_label.config(text="Actions: (game start)")
        
        # Randomly determine who goes first
        self.current_player = random.randint(0, 1)
        
        if self.current_player == 0:
            self.player_status.config(text="Your turn! Check or Bet?")
            self.bot_status.config(text="Waiting...")
            self.enable_buttons()
        else:
            self.player_status.config(text="Bot's turn...")
            self.bot_status.config(text="Computing with RL-CFR...")
            self.disable_buttons()
            # Bot moves first
            self.root.after(1000, self.bot_action)
    
    def enable_buttons(self):
        """Enable action buttons"""
        if not self.game_over:
            # Determine which buttons to show based on context
            if 'b' in self.history:
                # Facing a bet
                self.check_button.config(text="FOLD", state=tk.NORMAL)
                self.bet_button.config(text="CALL", state=tk.NORMAL)
            else:
                # Can check or bet
                self.check_button.config(text="CHECK", state=tk.NORMAL)
                self.bet_button.config(text="BET", state=tk.NORMAL)
    
    def disable_buttons(self):
        """Disable action buttons"""
        self.check_button.config(state=tk.DISABLED)
        self.bet_button.config(state=tk.DISABLED)
    
    def player_action(self, action):
        """Handle player action"""
        if self.game_over:
            return
        
        self.disable_buttons()
        self.history += action
        
        # Update display
        action_name = "Check" if action == 'p' and 'b' not in self.history[:-1] else "Fold" if action == 'p' else "Bet" if 'b' not in self.history[:-1] else "Call"
        self.player_status.config(text=f"You: {action_name}")
        self.update_history_display()
        
        # Update pot if betting/calling
        if action == 'b':
            self.pot += 1
            self.pot_label.config(text=f"POT: {self.pot} chips")
        
        # Check if game is over
        if self.game.is_terminal(self.history):
            self.end_game()
        else:
            # Bot's turn
            self.bot_status.config(text="Computing with RL-CFR...")
            self.show_bot_strategy()  # Show bot's strategy
            self.root.after(1000, self.bot_action)
    
    def bot_action(self):
        """Handle bot action"""
        if self.game_over:
            return
        
        # Get bot's action from RL-CFR strategy
        action_idx = self.trainer.get_action(self.bot_card, self.history)
        action = 'p' if action_idx == 0 else 'b'
        
        self.history += action
        
        # Update display
        action_name = "Check" if action == 'p' and 'b' not in self.history[:-1] else "Fold" if action == 'p' else "Bet" if 'b' not in self.history[:-1] else "Call"
        self.bot_status.config(text=f"Bot: {action_name}")
        self.update_history_display()
        
        # Update pot if betting/calling
        if action == 'b':
            self.pot += 1
            self.pot_label.config(text=f"POT: {self.pot} chips")
        
        # Check if game is over
        if self.game.is_terminal(self.history):
            self.end_game()
        else:
            # Player's turn
            self.player_status.config(text="Your turn!")
            self.enable_buttons()
    
    def update_history_display(self):
        """Update the action history display"""
        display_history = ""
        for i, action in enumerate(self.history):
            if i > 0:
                display_history += " → "
            
            player = "You" if (i % 2 == 0 and self.current_player == 0) or (i % 2 == 1 and self.current_player == 1) else "Bot"
            
            if action == 'p':
                if i > 0 and self.history[i-1] == 'b':
                    display_history += f"{player}: Fold"
                else:
                    display_history += f"{player}: Check"
            else:
                if i > 0 and 'b' in self.history[:i]:
                    display_history += f"{player}: Call"
                else:
                    display_history += f"{player}: Bet"
        
        self.history_label.config(text=f"Actions: {display_history}")
    
    def end_game(self):
        """End the current game and determine winner"""
        self.game_over = True
        self.disable_buttons()
        
        # Reveal bot's card
        card_symbols = ['J♠', 'Q♦', 'K♣']
        self.bot_card_label.config(text=card_symbols[self.bot_card],
                                  fg=['#3498db', '#e74c3c', '#f39c12'][self.bot_card])
        
        # Show final strategy
        self.show_bot_strategy()
        
        # Determine winner based on who was player 1
        if self.current_player == 0:
            # Human was player 1
            payoff = self.game.get_payoff([self.player_card, self.bot_card], self.history)
        else:
            # Bot was player 1
            payoff = -self.game.get_payoff([self.bot_card, self.player_card], self.history)
        
        # Update scores
        if payoff > 0:
            self.player_score += payoff
            self.player_status.config(text=f"You WIN! +{payoff} chips", fg='gold')
            self.bot_status.config(text="Lost", fg='red')
        else:
            self.bot_score += abs(payoff)
            self.player_status.config(text=f"You lose! {payoff} chips", fg='red')
            self.bot_status.config(text=f"Won! +{abs(payoff)} chips", fg='gold')
        
        self.games_played += 1
        
        # Update score display
        self.score_label.config(text=f"Player: {self.player_score}  |  Bot: {self.bot_score}  |  Games: {self.games_played}")
        
        # Show winner message
        winner_text = "You win!" if payoff > 0 else "RL-CFR Bot wins!"
        self.pot_label.config(text=f"{winner_text} POT: {self.pot} chips")

def main():
    root = tk.Tk()
    app = KuhnPokerRLGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
