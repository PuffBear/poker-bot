import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional
from kuhn_poker_game import KuhnPokerGame, BotPlayer
import time

class KuhnPokerGUI:
    """Enhanced Kuhn Poker GUI with hand tracking and strategy visualization"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kuhn Poker - Play vs CFR Bot")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Game state
        self.game = KuhnPokerGame()
        self.bot = BotPlayer("CFR")
        self.hand_history = []
        self.player_wins = 0
        self.bot_wins = 0
        self.total_hands = 0
        
        # Track current hand details
        self.current_hand_details = {}
        
        # Colors
        self.bg_color = '#2c3e50'
        self.table_color = '#27ae60'
        self.card_color = '#ecf0f1'
        self.button_color = '#3498db'
        self.button_hover = '#2980b9'
        
        # Create UI
        self._create_ui()
        self._start_new_hand()
        
    def _create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Game area
        left_panel = tk.Frame(main_frame, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right panel - Info area
        right_panel = tk.Frame(main_frame, bg=self.bg_color, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # === LEFT PANEL ===
        
        # Title
        title_label = tk.Label(
            left_panel,
            text="♠ KUHN POKER ♥",
            font=('Arial', 32, 'bold'),
            bg=self.bg_color,
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Game stats
        self.stats_label = tk.Label(
            left_panel,
            text="Games: 0 | You: 0 | Bot: 0",
            font=('Arial', 14),
            bg=self.bg_color,
            fg='white'
        )
        self.stats_label.pack()
        
        # Strategy selector
        strategy_frame = tk.Frame(left_panel, bg=self.bg_color)
        strategy_frame.pack(pady=10)
        
        tk.Label(
            strategy_frame,
            text="Bot Strategy:",
            font=('Arial', 12),
            bg=self.bg_color,
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        self.strategy_var = tk.StringVar(value="CFR")
        self.strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            values=["CFR", "Random", "Aggressive", "Conservative"],
            state='readonly',
            width=15,
            font=('Arial', 12)
        )
        self.strategy_combo.pack(side=tk.LEFT, padx=5)
        self.strategy_combo.bind('<<ComboboxSelected>>', self._change_strategy)
        
        # Poker table
        self.table_canvas = tk.Canvas(
            left_panel,
            width=700,
            height=450,
            bg=self.table_color,
            highlightthickness=2,
            highlightbackground='#1e8449'
        )
        self.table_canvas.pack(pady=20)
        
        # Draw table elements
        self._draw_table()
        
        # Action buttons
        button_frame = tk.Frame(left_panel, bg=self.bg_color)
        button_frame.pack(pady=20)
        
        self.check_btn = tk.Button(
            button_frame,
            text="CHECK",
            font=('Arial', 16, 'bold'),
            bg=self.button_color,
            fg='white',
            width=12,
            height=2,
            command=lambda: self._player_action('check')
        )
        self.check_btn.pack(side=tk.LEFT, padx=10)
        
        self.bet_btn = tk.Button(
            button_frame,
            text="BET",
            font=('Arial', 16, 'bold'),
            bg='#e74c3c',
            fg='white',
            width=12,
            height=2,
            command=lambda: self._player_action('bet')
        )
        self.bet_btn.pack(side=tk.LEFT, padx=10)
        
        self.new_game_btn = tk.Button(
            button_frame,
            text="NEW GAME",
            font=('Arial', 16, 'bold'),
            bg='#95a5a6',
            fg='white',
            width=12,
            height=2,
            command=self._start_new_hand
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=10)
        
        # Game rules
        rules_text = "Rules: Each player antes 1 chip. Cards: Jack < Queen < King.\n" \
                    "First to act: Check or Bet. Facing a bet: Fold or Call. Pot goes to winner."
        tk.Label(
            left_panel,
            text=rules_text,
            font=('Arial', 10),
            bg=self.bg_color,
            fg='#bdc3c7',
            justify=tk.CENTER
        ).pack(pady=10)
        
        # === RIGHT PANEL ===
        
        # Bot thoughts section
        thoughts_label = tk.Label(
            right_panel,
            text="🤖 Bot Thought Process",
            font=('Arial', 14, 'bold'),
            bg=self.bg_color,
            fg='white'
        )
        thoughts_label.pack(pady=(0, 10))
        
        self.thoughts_text = scrolledtext.ScrolledText(
            right_panel,
            width=45,
            height=8,
            font=('Courier', 11),
            bg='#34495e',
            fg='#ecf0f1',
            wrap=tk.WORD
        )
        self.thoughts_text.pack(pady=(0, 20))
        
        # Strategy indicator
        indicator_label = tk.Label(
            right_panel,
            text="📊 Strategy Indicator",
            font=('Arial', 14, 'bold'),
            bg=self.bg_color,
            fg='white'
        )
        indicator_label.pack(pady=(0, 10))
        
        self.indicator_frame = tk.Frame(right_panel, bg='#34495e', relief=tk.RIDGE, bd=2)
        self.indicator_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.indicator_text = tk.Label(
            self.indicator_frame,
            text="Waiting for bot action...",
            font=('Arial', 11),
            bg='#34495e',
            fg='#ecf0f1',
            wraplength=350,
            justify=tk.LEFT,
            padx=10,
            pady=10
        )
        self.indicator_text.pack()
        
        # Hand history section
        history_label = tk.Label(
            right_panel,
            text="📋 Hand History",
            font=('Arial', 14, 'bold'),
            bg=self.bg_color,
            fg='white'
        )
        history_label.pack(pady=(0, 10))
        
        # History table with headers
        history_frame = tk.Frame(right_panel, bg='#34495e')
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        # Column headers
        headers_frame = tk.Frame(history_frame, bg='#1a252f')
        headers_frame.pack(fill=tk.X)
        
        tk.Label(
            headers_frame,
            text="Hand",
            font=('Arial', 10, 'bold'),
            bg='#1a252f',
            fg='white',
            width=6
        ).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Label(
            headers_frame,
            text="Winner",
            font=('Arial', 10, 'bold'),
            bg='#1a252f',
            fg='white',
            width=8
        ).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Label(
            headers_frame,
            text="Cards",
            font=('Arial', 10, 'bold'),
            bg='#1a252f',
            fg='white',
            width=10
        ).pack(side=tk.LEFT, padx=2, pady=5)
        
        tk.Label(
            headers_frame,
            text="Pot",
            font=('Arial', 10, 'bold'),
            bg='#1a252f',
            fg='white',
            width=6
        ).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Scrollable history list
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            width=45,
            height=15,
            font=('Courier', 10),
            bg='#34495e',
            fg='#ecf0f1',
            wrap=tk.WORD
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
    def _draw_table(self):
        """Draw the poker table elements"""
        canvas = self.table_canvas
        
        # Bot area
        canvas.create_text(350, 60, text="CFR Bot", font=('Arial', 18, 'bold'), fill='white')
        
        # Bot card placeholder
        self.bot_card_rect = canvas.create_rectangle(310, 90, 390, 160, fill='#c0392b', outline='black', width=3)
        self.bot_card_text = canvas.create_text(350, 125, text="?", font=('Arial', 48, 'bold'), fill='white')
        
        # Bot action text
        self.bot_action_text = canvas.create_text(350, 185, text="", font=('Arial', 14, 'bold'), fill='#f39c12')
        
        # Pot
        self.pot_text = canvas.create_text(350, 225, text="POT: 2 chips", font=('Arial', 20, 'bold'), fill='#f1c40f')
        
        # Player area
        canvas.create_text(350, 280, text="You", font=('Arial', 18, 'bold'), fill='white')
        
        # Player card
        self.player_card_rect = canvas.create_rectangle(310, 310, 390, 380, fill=self.card_color, outline='black', width=3)
        self.player_card_text = canvas.create_text(350, 345, text="K♣", font=('Arial', 48, 'bold'), fill='#e74c3c')
        
        # Player action text
        self.player_action_text = canvas.create_text(350, 405, text="Your turn", font=('Arial', 14, 'bold'), fill='#e74c3c')
        
    def _update_table(self):
        """Update table display"""
        canvas = self.table_canvas
        
        # Update player card
        suit_symbol = self._get_suit_symbol(self.game.player_card)
        suit_color = '#e74c3c' if suit_symbol in ['♥', '♦'] else 'black'
        canvas.itemconfig(self.player_card_text, text=f"{self.game.player_card}{suit_symbol}", fill=suit_color)
        
        # Update bot card (hidden or revealed)
        if self.game.terminal:
            suit_symbol = self._get_suit_symbol(self.game.bot_card)
            suit_color = '#e74c3c' if suit_symbol in ['♥', '♦'] else 'black'
            canvas.itemconfig(self.bot_card_text, text=f"{self.game.bot_card}{suit_symbol}", fill=suit_color)
            canvas.itemconfig(self.bot_card_rect, fill=self.card_color)
        else:
            canvas.itemconfig(self.bot_card_text, text="?", fill='white')
            canvas.itemconfig(self.bot_card_rect, fill='#c0392b')
        
        # Update pot
        canvas.itemconfig(self.pot_text, text=f"POT: {self.game.pot} chips")
        
        # Update action texts
        if len(self.game.history) > 0:
            last_action = self.game.history[-1].upper()
            if len(self.game.history) % 2 == 1:  # Player acted
                canvas.itemconfig(self.player_action_text, text=f"You: {last_action}")
            else:  # Bot acted
                canvas.itemconfig(self.bot_action_text, text=f"Bot: {last_action}")
        
        # Show winner
        if self.game.terminal:
            winner_text = f"🏆 {self.game.winner} WINS!"
            if self.game.winner == "Player":
                canvas.itemconfig(self.player_action_text, text=winner_text, fill='#2ecc71')
            else:
                canvas.itemconfig(self.bot_action_text, text=winner_text, fill='#2ecc71')
    
    def _get_suit_symbol(self, card: str) -> str:
        """Get suit symbol for card"""
        suits = {'J': '♣', 'Q': '♦', 'K': '♠'}
        return suits.get(card, '♠')
    
    def _start_new_hand(self):
        """Start a new hand"""
        self.game.reset_game()
        self._update_table()
        
        # Clear action texts
        self.table_canvas.itemconfig(self.bot_action_text, text="", fill='#f39c12')
        self.table_canvas.itemconfig(self.player_action_text, text="Your turn", fill='#e74c3c')
        
        # Enable buttons
        self.check_btn.config(state=tk.NORMAL)
        self.bet_btn.config(state=tk.NORMAL)
        
        # Clear thoughts
        self.thoughts_text.delete('1.0', tk.END)
        self.thoughts_text.insert('1.0', "Waiting for your action...")
        
        # Update indicator
        self.indicator_text.config(text="Waiting for your action...")
        
        # Store current hand details
        self.current_hand_details = {
            'player_card': self.game.player_card,
            'bot_card': self.game.bot_card,
            'actions': []
        }
    
    def _player_action(self, action: str):
        """Handle player action"""
        if self.game.terminal:
            return
        
        # Record action
        self.current_hand_details['actions'].append(('Player', action))
        
        # Make player move
        terminal = self.game.make_move(action, is_player=True)
        self._update_table()
        
        if terminal:
            self._handle_terminal()
            return
        
        # Disable buttons during bot thinking
        self.check_btn.config(state=tk.DISABLED)
        self.bet_btn.config(state=tk.DISABLED)
        
        # Bot turn
        self.root.after(500, self._bot_turn)
    
    def _bot_turn(self):
        """Handle bot turn"""
        # Get bot action
        bot_action, info = self.bot.get_action(self.game)
        
        # Record action
        self.current_hand_details['actions'].append(('Bot', bot_action))
        
        # Update thoughts
        self.thoughts_text.delete('1.0', tk.END)
        self.thoughts_text.insert('1.0', info['thought'])
        
        # Update strategy indicator
        self._update_strategy_indicator(info)
        
        # Make bot move
        terminal = self.game.make_move(bot_action, is_player=False)
        self._update_table()
        
        if terminal:
            self._handle_terminal()
        else:
            # Re-enable buttons for player
            self.check_btn.config(state=tk.NORMAL)
            self.bet_btn.config(state=tk.NORMAL)
    
    def _update_strategy_indicator(self, info: dict):
        """Update the strategy indicator"""
        probs = info['probabilities']
        using_strategy = info['using_strategy']
        
        if using_strategy:
            status = "✓ Following CFR Strategy"
            color = '#2ecc71'
        else:
            status = f"Playing {self.strategy_var.get()} Mode"
            color = '#f39c12'
        
        indicator_text = f"{status}\n\n"
        indicator_text += f"Action Probabilities:\n"
        indicator_text += f"  Check: {probs[0]:.1%}\n"
        indicator_text += f"  Bet:   {probs[1]:.1%}\n\n"
        indicator_text += f"Info Set: {info['info_set']}"
        
        self.indicator_text.config(text=indicator_text, fg=color)
    
    def _handle_terminal(self):
        """Handle terminal state (hand ended)"""
        # Disable buttons
        self.check_btn.config(state=tk.DISABLED)
        self.bet_btn.config(state=tk.DISABLED)
        
        # Update stats
        self.total_hands += 1
        if self.game.winner == "Player":
            self.player_wins += 1
        else:
            self.bot_wins += 1
        
        self.stats_label.config(
            text=f"Games: {self.total_hands} | You: {self.player_wins} | Bot: {self.bot_wins}"
        )
        
        # Add to hand history
        self._add_hand_to_history()
        
    def _add_hand_to_history(self):
        """Add current hand to history"""
        hand_num = self.total_hands
        winner = self.game.winner
        player_card = self.current_hand_details['player_card']
        bot_card = self.current_hand_details['bot_card']
        pot = self.game.pot
        
        # Format card display
        player_suit = self._get_suit_symbol(player_card)
        bot_suit = self._get_suit_symbol(bot_card)
        cards = f"{player_card}{player_suit} vs {bot_card}{bot_suit}"
        
        # Create history entry
        history_line = f"#{hand_num:<4} {winner:<8} {cards:<11} {pot} chips\n"
        
        # Insert at top of history
        self.history_text.insert('1.0', history_line)
        
        # Add action sequence
        actions_str = " → ".join([f"{actor}: {action}" for actor, action in self.current_hand_details['actions']])
        self.history_text.insert('2.0', f"     Actions: {actions_str}\n\n")
        
    def _change_strategy(self, event=None):
        """Change bot strategy"""
        new_strategy = self.strategy_var.get()
        self.bot = BotPlayer(new_strategy)
        
        # Show message
        self.thoughts_text.delete('1.0', tk.END)
        self.thoughts_text.insert('1.0', f"Strategy changed to: {new_strategy}\n\nStart a new game to see the difference!")
        
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


if __name__ == "__main__":
    app = KuhnPokerGUI()
    app.run()