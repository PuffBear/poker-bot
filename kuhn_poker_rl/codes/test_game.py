#!/usr/bin/env python3
"""Test script for Kuhn Poker game logic"""

from kuhn_poker_game import KuhnPokerGame, BotPlayer, CFRStrategy
import numpy as np

def test_game_basics():
    """Test basic game functionality"""
    print("Testing basic game functionality...")
    
    game = KuhnPokerGame()
    assert game.pot == 2, "Initial pot should be 2"
    assert len(game.deck) == 3, "Deck should have 3 cards"
    assert game.player_card in ['J', 'Q', 'K'], "Player card should be J, Q, or K"
    assert game.bot_card in ['J', 'Q', 'K'], "Bot card should be J, Q, or K"
    assert game.player_card != game.bot_card, "Player and bot should have different cards"
    
    print("✓ Basic game initialization works")

def test_game_flow():
    """Test game flow scenarios"""
    print("\nTesting game flow scenarios...")
    
    # Test both players check
    game = KuhnPokerGame()
    game.make_move('check', is_player=True)
    assert not game.terminal, "Game shouldn't be terminal after first check"
    game.make_move('check', is_player=False)
    assert game.terminal, "Game should be terminal after both check"
    assert game.pot == 2, "Pot should remain 2 when both check"
    print("✓ Both players check scenario works")
    
    # Test player bets, bot folds
    game = KuhnPokerGame()
    game.make_move('bet', is_player=True)
    assert not game.terminal, "Game shouldn't be terminal after player bets"
    game.make_move('check', is_player=False)  # In this context, check means fold
    assert game.terminal, "Game should be terminal after fold"
    assert game.winner == "Player", "Player should win when bot folds"
    print("✓ Bet and fold scenario works")
    
    # Test player bets, bot calls
    game = KuhnPokerGame()
    game.player_card = 'K'
    game.bot_card = 'Q'
    game.make_move('bet', is_player=True)
    game.make_move('bet', is_player=False)  # Call
    assert game.terminal, "Game should be terminal after call"
    assert game.pot == 4, "Pot should be 4 after bet and call"
    assert game.winner == "Player", "Player with K should beat bot with Q"
    print("✓ Bet and call scenario works")

def test_cfr_strategy():
    """Test CFR strategy"""
    print("\nTesting CFR strategy...")
    
    cfr = CFRStrategy()
    print("Training CFR (this may take a moment)...")
    cfr.train(iterations=1000)
    
    # Check that strategies are learned
    info_set = "K:"
    strategy = cfr.get_average_strategy(info_set)
    assert len(strategy) == 2, "Strategy should have 2 actions"
    assert np.isclose(np.sum(strategy), 1.0), "Strategy probabilities should sum to 1"
    
    # King should prefer betting
    assert strategy[1] > 0.5, "With King, betting probability should be > 50%"
    print(f"✓ CFR strategy learned (K bets {strategy[1]:.1%} of the time)")

def test_bot_strategies():
    """Test different bot strategies"""
    print("\nTesting bot strategies...")
    
    strategies = ["CFR", "Random", "Aggressive", "Conservative"]
    
    for strategy_name in strategies:
        bot = BotPlayer(strategy_name)
        game = KuhnPokerGame()
        
        action, info = bot.get_action(game)
        assert action in ['check', 'bet'], f"Action should be check or bet, got {action}"
        assert 'probabilities' in info, "Info should contain probabilities"
        assert 'thought' in info, "Info should contain thought process"
        
        probs = info['probabilities']
        assert len(probs) == 2, "Probabilities should have 2 values"
        assert np.isclose(np.sum(probs), 1.0), f"Probabilities should sum to 1 for {strategy_name}"
        
        print(f"✓ {strategy_name} strategy works")

def test_information_sets():
    """Test information set generation"""
    print("\nTesting information sets...")
    
    game = KuhnPokerGame()
    
    # Test various info sets
    info_set_1 = game.get_info_set('K', [])
    assert info_set_1 == "K:", "Empty history info set"
    
    info_set_2 = game.get_info_set('Q', ['check'])
    assert info_set_2 == "Q:check", "Single action info set"
    
    info_set_3 = game.get_info_set('J', ['check', 'bet'])
    assert info_set_3 == "J:checkbet", "Multiple actions info set"
    
    print("✓ Information set generation works")

def test_payoffs():
    """Test payoff calculations"""
    print("\nTesting payoffs...")
    
    game = KuhnPokerGame()
    
    # Both check, K beats Q
    payoff = game.get_payoff(['check', 'check'], 'K', 'Q')
    assert payoff == 1, "Player with K should win +1 vs Q when both check"
    
    # Both check, Q loses to K
    payoff = game.get_payoff(['check', 'check'], 'Q', 'K')
    assert payoff == -1, "Player with Q should lose -1 vs K when both check"
    
    # Player bets, bot calls, K beats Q
    payoff = game.get_payoff(['bet', 'bet'], 'K', 'Q')
    assert payoff == 2, "Player with K should win +2 vs Q when betting"
    
    # Player checks, bot bets, player folds
    payoff = game.get_payoff(['check', 'bet', 'check'], 'K', 'Q')
    assert payoff == -1, "Player folding should lose -1"
    
    print("✓ Payoff calculations work")

def main():
    """Run all tests"""
    print("=" * 50)
    print("KUHN POKER GAME TESTS")
    print("=" * 50)
    
    try:
        test_game_basics()
        test_game_flow()
        test_information_sets()
        test_payoffs()
        test_cfr_strategy()
        test_bot_strategies()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("\nThe game is ready to play!")
        print("Run: python3 kuhn_poker_gui.py")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())