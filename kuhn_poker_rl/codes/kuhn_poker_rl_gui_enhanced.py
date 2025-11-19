"""
Kuhn Poker GUI - Play against the RL-CFR trained bot
Updated to use Reinforcement Learning CFR
Enhanced with: win tracking by card, bot thought process, and strategy adherence indicator
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import os
from kuhn_poker_rl_cfr import RLCFRTrainer, KuhnPoker
import threading
import time
import numpy as np

class KuhnPokerRLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kuhn Poker - Play vs RL-CFR Bot (Enhanced)")
        self.root.geometry("1200x850")
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
        
        # NEW: Card-based win tracking
        self.card_wins = {
            'player': {'J': 0, 'Q': 0, 'K': 0},
            'bot': {'J': 0, 'Q': 0, 'K': 0}
        }
        
        # NEW: Bot thinking variables
        self.bot_last_strategy = None
        self.bot_last_action_idx = None
        self.bot_followed_strategy = None
        
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
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (game)
        left_panel = tk.Frame(main_container, bg='#2c3e50')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Right panel (NEW: statistics and bot thinking)
        right_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, bd=2, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        right_panel.pack_propagate(False)
        
        # Title (in left panel)
        title_frame = tk.Frame(left_panel, bg='#2c3e50')
        title_frame.pack(pady=15)
        
        title = tk.Label(title_frame, text="♠ KUHN POKER ♥", 
                        font=('Arial Bold', 28), bg='#2c3e50', fg='white')
        title.pack()
        
        subtitle = tk.Label(title_frame, text="vs RL-CFR Bot (Reinforcement Learning CFR)", 
                           font=('Arial', 12), bg='#2c3e50', fg='#95a5a6')
        subtitle.pack()
        
        # Score Frame
        score_frame = tk.Frame(left_panel, bg='#34495e', relief=tk.RAISED, bd=2)
        score_frame.pack(pady=10)
        
        self.score_label = tk.Label(score_frame, 
                                   text=f"Player: {self.player_score}  |  Bot: {self.bot_score}  |  Games: {self.games_played}",
                                   font=('Arial', 14), bg='#34495e', fg='white', padx=20, pady=10)
        self.score_label.pack()
        
        # Game Area
        game_frame = tk.Frame(left_panel, bg='#27ae60', relief=tk.SUNKEN, bd=3)
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
        button_frame = tk.Frame(left_panel, bg='#2c3e50')
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
        info_frame = tk.Frame(left_panel, bg='#2c3e50')
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = """RL-CFR: Uses Monte Carlo sampling, experience replay, and exploration.
        Rules: Jack < Queen < King. Check/Bet first, then Fold/Call facing bets."""
        
        tk.Label(info_frame, text=info_text, font=('Arial', 10), 
                bg='#2c3e50', fg='#ecf0f1', wraplength=600).pack()
        
        # Training info
        self.training_info = tk.Label(info_frame, text="", font=('Courier', 9), 
                                     bg='#2c3e50', fg='#95a5a6')
        self.training_info.pack()
        self.update_training_info()
        
        # NEW: Right Panel Contents
        self.create_right_panel(right_panel)
    
    def create_right_panel(self, parent):
        """NEW: Create the right panel with statistics and bot thinking"""
        
        # Title for right panel
        tk.Label(parent, text="📊 Statistics & Analysis", 
                font=('Arial Bold', 14), bg='#34495e', fg='white').pack(pady=10)
        
        # Card Win Statistics
        stats_frame = tk.LabelFrame(parent, text="Win Statistics by Card", 
                                   font=('Arial Bold', 11), bg='#34495e', fg='white',
                                   relief=tk.RAISED, bd=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Header
        header_frame = tk.Frame(stats_frame, bg='#34495e')
        header_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(header_frame, text="Card", font=('Arial Bold', 10), 
                bg='#34495e', fg='white', width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(header_frame, text="You", font=('Arial Bold', 10), 
                bg='#34495e', fg='#2ecc71', width=8).pack(side=tk.LEFT, padx=5)
        tk.Label(header_frame, text="Bot", font=('Arial Bold', 10), 
                bg='#34495e', fg='#e74c3c', width=8).pack(side=tk.LEFT, padx=5)
        
        # Card rows
        self.card_stat_labels = {}
        for card, symbol in [('J', 'J♠'), ('Q', 'Q♦'), ('K', 'K♣')]:
            row_frame = tk.Frame(stats_frame, bg='#2c3e50', relief=tk.SUNKEN, bd=1)
            row_frame.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(row_frame, text=symbol, font=('Arial', 10), 
                    bg='#2c3e50', fg='white', width=8).pack(side=tk.LEFT, padx=5)
            
            player_label = tk.Label(row_frame, text="0", font=('Arial', 10), 
                                   bg='#2c3e50', fg='#2ecc71', width=8)
            player_label.pack(side=tk.LEFT, padx=5)
            
            bot_label = tk.Label(row_frame, text="0", font=('Arial', 10), 
                                bg='#2c3e50', fg='#e74c3c', width=8)
            bot_label.pack(side=tk.LEFT, padx=5)
            
            self.card_stat_labels[card] = {'player': player_label, 'bot': bot_label}
        
        # Bot Thinking Panel
        thinking_frame = tk.LabelFrame(parent, text="🤖 Bot's Thought Process", 
                                      font=('Arial Bold', 11), bg='#34495e', fg='white',
                                      relief=tk.RAISED, bd=2)
        thinking_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current situation
        tk.Label(thinking_frame, text="Current Situation:", 
                font=('Arial Bold', 10), bg='#34495e', fg='#f39c12',
                anchor='w').pack(fill=tk.X, padx=5, pady=(5,0))
        
        self.bot_situation_label = tk.Label(thinking_frame, text="Waiting for game...", 
                                           font=('Courier', 9), bg='#2c3e50', fg='white',
                                           justify=tk.LEFT, anchor='w', wraplength=260)
        self.bot_situation_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Strategy probabilities
        tk.Label(thinking_frame, text="Strategy Probabilities:", 
                font=('Arial Bold', 10), bg='#34495e', fg='#f39c12',
                anchor='w').pack(fill=tk.X, padx=5, pady=(10,0))
        
        self.bot_probabilities_label = tk.Label(thinking_frame, text="---", 
                                               font=('Courier', 9), bg='#2c3e50', fg='white',
                                               justify=tk.LEFT, anchor='w', wraplength=260)
        self.bot_probabilities_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Action taken
        tk.Label(thinking_frame, text="Action Taken:", 
                font=('Arial Bold', 10), bg='#34495e', fg='#f39c12',
                anchor='w').pack(fill=tk.X, padx=5, pady=(10,0))
        
        self.bot_action_label = tk.Label(thinking_frame, text="---", 
                                        font=('Courier', 10), bg='#2c3e50', fg='white',
                                        justify=tk.LEFT, anchor='w', wraplength=260)
        self.bot_action_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Strategy adherence indicator
        tk.Label(thinking_frame, text="Strategy Adherence:", 
                font=('Arial Bold', 10), bg='#34495e', fg='#f39c12',
                anchor='w').pack(fill=tk.X, padx=5, pady=(10,0))
        
        self.bot_adherence_label = tk.Label(thinking_frame, text="---", 
                                           font=('Courier', 9), bg='#2c3e50', fg='white',
                                           justify=tk.LEFT, anchor='w', wraplength=260)
        self.bot_adherence_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Explanation
        tk.Label(thinking_frame, text="Explanation:", 
                font=('Arial Bold', 10), bg='#34495e', fg='#f39c12',
                anchor='w').pack(fill=tk.X, padx=5, pady=(10,0))
        
        self.bot_explanation_label = tk.Label(thinking_frame, text="---", 
                                             font=('Courier', 9), bg='#2c3e50', fg='#95a5a6',
                                             justify=tk.LEFT, anchor='w', wraplength=260)
        self.bot_explanation_label.pack(fill=tk.X, padx=5, pady=2)
    
    def update_training_info(self):
        """Display training information"""
        info = (f"Bot trained with {self.trainer.iteration_count} iterations | "
                f"Learning rate: {self.trainer.learning_rate:.6f} | "
                f"Buffer size: {len(self.trainer.replay_buffer.buffer)}")
        self.training_info.config(text=info)
    
    def update_card_statistics(self):
        """NEW: Update the card win statistics display"""
        for card in ['J', 'Q', 'K']:
            self.card_stat_labels[card]['player'].config(
                text=str(self.card_wins['player'][card])
            )
            self.card_stat_labels[card]['bot'].config(
                text=str(self.card_wins['bot'][card])
            )
    
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
    
    def update_bot_thinking_display(self, phase="waiting"):
        """NEW: Update the bot's thinking process display"""
        if phase == "waiting":
            self.bot_situation_label.config(text="Waiting for game...")
            self.bot_probabilities_label.config(text="---")
            self.bot_action_label.config(text="---")
            self.bot_adherence_label.config(text="---")
            self.bot_explanation_label.config(text="---")
            
        elif phase == "computing":
            if self.bot_card is not None:
                card_names = ['Jack (J♠)', 'Queen (Q♦)', 'King (K♣)']
                
                # Situation
                situation = f"Card: {card_names[self.bot_card]}\nHistory: '{self.history}'"
                if 'b' in self.history:
                    situation += "\nFacing: BET (must Fold/Call)"
                else:
                    situation += "\nFacing: No bet (can Check/Bet)"
                self.bot_situation_label.config(text=situation)
                
                # Get strategy
                info_set_key = f"{self.bot_card}:{self.history}"
                strategy = self.trainer.get_average_strategy(info_set_key)
                self.bot_last_strategy = strategy
                
                # Probabilities
                if 'b' in self.history:
                    prob_text = f"Fold: {strategy[0]:.1%}\nCall: {strategy[1]:.1%}"
                else:
                    prob_text = f"Check: {strategy[0]:.1%}\nBet: {strategy[1]:.1%}"
                
                self.bot_probabilities_label.config(text=prob_text)
                
                self.bot_action_label.config(text="Computing...")
                self.bot_adherence_label.config(text="Analyzing...")
                self.bot_explanation_label.config(text="Bot is sampling from strategy...")
        
        elif phase == "decided":
            if self.bot_last_strategy is not None and self.bot_last_action_idx is not None:
                # Action taken
                if 'b' in self.history[:-1]:  # The history before bot's last action
                    action_names = ["FOLD", "CALL"]
                else:
                    action_names = ["CHECK", "BET"]
                
                action_text = f"✓ {action_names[self.bot_last_action_idx]}"
                self.bot_action_label.config(text=action_text, fg='#2ecc71')
                
                # Strategy adherence
                optimal_action = 1 if self.bot_last_strategy[1] > self.bot_last_strategy[0] else 0
                
                if self.bot_last_action_idx == optimal_action:
                    if self.bot_last_strategy[optimal_action] > 0.8:
                        adherence = "✓ STRONG FOLLOW"
                        color = '#2ecc71'
                        explanation = f"Took the dominant action ({self.bot_last_strategy[optimal_action]:.1%} probability)."
                    else:
                        adherence = "✓ FOLLOWING STRATEGY"
                        color = '#3498db'
                        explanation = f"Took the slightly favored action ({self.bot_last_strategy[optimal_action]:.1%} vs {self.bot_last_strategy[1-optimal_action]:.1%})."
                else:
                    if self.bot_last_strategy[self.bot_last_action_idx] > 0.3:
                        adherence = "↔ MIXED STRATEGY"
                        color = '#f39c12'
                        explanation = f"Exploring: took less favored action ({self.bot_last_strategy[self.bot_last_action_idx]:.1%} probability)."
                    else:
                        adherence = "⚠ EXPLORATION"
                        color = '#e67e22'
                        explanation = f"Heavy exploration: took low-probability action ({self.bot_last_strategy[self.bot_last_action_idx]:.1%})."
                
                self.bot_adherence_label.config(text=adherence, fg=color)
                self.bot_explanation_label.config(text=explanation)
    
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
        
        # Reset bot thinking display
        self.update_bot_thinking_display("waiting")
        
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
            # NEW: Show bot is computing
            self.update_bot_thinking_display("computing")
            self.root.after(1500, self.bot_action)
    
    def bot_action(self):
        """Handle bot action"""
        if self.game_over:
            return
        
        # Get bot's action from RL-CFR strategy
        action_idx = self.trainer.get_action(self.bot_card, self.history)
        action = 'p' if action_idx == 0 else 'b'
        
        # NEW: Store bot's action for analysis
        self.bot_last_action_idx = action_idx
        
        self.history += action
        
        # Update display
        action_name = "Check" if action == 'p' and 'b' not in self.history[:-1] else "Fold" if action == 'p' else "Bet" if 'b' not in self.history[:-1] else "Call"
        self.bot_status.config(text=f"Bot: {action_name}")
        self.update_history_display()
        
        # NEW: Update bot thinking display with decision
        self.update_bot_thinking_display("decided")
        
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
        
        # NEW: Track card-based wins
        card_names = ['J', 'Q', 'K']
        if payoff > 0:
            self.card_wins['player'][card_names[self.player_card]] += 1
        else:
            self.card_wins['bot'][card_names[self.bot_card]] += 1
        
        # Update card statistics display
        self.update_card_statistics()
        
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