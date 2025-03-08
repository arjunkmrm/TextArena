from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import asyncio

import random
import itertools
from collections import Counter

# Initialize FastMCP server
mcp = FastMCP("poker")


# Card representations
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def __repr__(self):
        return self.__str__()

class PokerGame:
    def __init__(self):
        self.deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        
    def deal_cards(self, my_hand=None, opponent_hand=None, community_cards=None):
        """Set up the game state with known cards"""
        self.my_hand = my_hand if my_hand else []
        self.opponent_hand = opponent_hand if opponent_hand else []
        self.community_cards = community_cards if community_cards else []
        
        # Remove known cards from deck
        all_known_cards = self.my_hand + self.opponent_hand + self.community_cards
        self.deck = [card for card in self.deck if not any(
            card.rank == known.rank and card.suit == known.suit for known in all_known_cards)]
            
    def evaluate_hand(self, hand):
        """Evaluate a 5-card poker hand and return a score"""
        if len(hand) != 5:
            return 0
            
        ranks = [card.rank for card in hand]
        suits = [card.suit for card in hand]
        rank_values = [RANK_VALUES[rank] for rank in ranks]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        sorted_values = sorted(rank_values)
        is_straight = (sorted_values == list(range(min(sorted_values), max(sorted_values) + 1)))
        # Check for A-5 straight
        if sorted_values == [0, 1, 2, 3, 12]:
            is_straight = True
            sorted_values = [0, 1, 2, 3, 4]
        
        # Count occurrences
        value_counts = Counter(rank_values)
        count_values = sorted(value_counts.values(), reverse=True)
        
        # Score hand
        if is_straight and is_flush:
            if sorted_values == [8, 9, 10, 11, 12]:  # Royal flush
                return 9
            return 8  # Straight flush
        elif count_values == [4, 1]:
            return 7  # Four of a kind
        elif count_values == [3, 2]:
            return 6  # Full house
        elif is_flush:
            return 5  # Flush
        elif is_straight:
            return 4  # Straight
        elif count_values[0] == 3:
            return 3  # Three of a kind
        elif count_values == [2, 2, 1]:
            return 2  # Two pair
        elif count_values[0] == 2:
            return 1  # Pair
        else:
            return 0  # High card
    
    def find_best_hand(self, player_cards, community_cards):
        """Find the best 5-card hand from player's cards and community cards"""
        all_cards = player_cards + community_cards
        best_score = -1
        best_hand = None
        
        for combo in itertools.combinations(all_cards, 5):
            score = self.evaluate_hand(list(combo))
            if score > best_score:
                best_score = score
                best_hand = list(combo)
        
        return best_score, best_hand
        
    def calculate_win_probability(self, simulations=1000):
        """Calculate the probability of winning through Monte Carlo simulation"""
        wins = 0
        unknown_community_count = 5 - len(self.community_cards)
        
        # Determine if opponent's hand is fully known
        opponent_known = len(self.opponent_hand) == 2
        
        for _ in range(simulations):
            # Create a copy of the deck for this simulation
            sim_deck = self.deck.copy()
            random.shuffle(sim_deck)
            
            # Complete community cards
            sim_community = self.community_cards.copy()
            for _ in range(unknown_community_count):
                sim_community.append(sim_deck.pop())
            
            # Complete opponent's hand if not fully known
            sim_opponent = self.opponent_hand.copy()
            while len(sim_opponent) < 2:
                sim_opponent.append(sim_deck.pop())
            
            # Evaluate hands
            my_score, _ = self.find_best_hand(self.my_hand, sim_community)
            opponent_score, _ = self.find_best_hand(sim_opponent, sim_community)
            
            if my_score > opponent_score:
                wins += 1
            elif my_score == opponent_score:
                # Implement tiebreaker if needed
                wins += 0.5
        
        return wins / simulations
    
    def suggest_action(self, win_probability):
        """Suggest an action based on win probability"""
        if win_probability > 0.8:
            return "Strong hand. Consider raising significantly."
        elif win_probability > 0.6:
            return "Good hand. Consider raising moderately."
        elif win_probability > 0.4:
            return "Decent hand. Consider calling or small raise."
        elif win_probability > 0.25:
            return "Weak hand. Consider checking or calling if the bet is small."
        else:
            return "Very weak hand. Consider folding unless the bet is minimal."

def parse_card(card_str):
    """Parse a card string like 'As' (Ace of spades) or '10h' (10 of hearts)"""
    suit_map = {'h': 'hearts', 'd': 'diamonds', 'c': 'clubs', 's': 'spades'}
    rank_map = {'A': 'A', 'K': 'K', 'Q': 'Q', 'J': 'J', 'T': '10'}
    
    if len(card_str) == 2:
        rank_char, suit_char = card_str[0], card_str[1]
        rank = rank_map.get(rank_char, rank_char)
    else:  # Handle 10
        rank_char, suit_char = card_str[0:2], card_str[2]
        rank = rank_map.get(rank_char, rank_char)
    
    suit = suit_map.get(suit_char.lower())
    return Card(rank, suit)


@mcp.tool()
async def suggest_actions(my_cards_input: str, community_input: str, opponent_input: str ) -> str:    
    # Get your hand
    # my_cards_input = input("Enter your two cards (space-separated, e.g., 'As Kh'): ")
    my_cards = [parse_card(card.strip()) for card in my_cards_input.split()]
    
    # Get community cards
    # community_input = input("Enter community cards (space-separated, leave blank if none): ")
    community_cards = []
    if community_input.strip():
        community_cards = [parse_card(card.strip()) for card in community_input.split()]
    
    # Get opponent's cards if known
    opponent_cards = []
    if opponent_input.strip():
        opponent_cards = [parse_card(card.strip()) for card in opponent_input.split()]
    
    # Set up game and calculate probabilities
    game = PokerGame()
    game.deal_cards(my_cards, opponent_cards, community_cards)
    
    win_prob = game.calculate_win_probability(simulations=5000)
    print(f"\nYour win probability: {win_prob:.2f} ({win_prob*100:.1f}%)")
    print(f"Suggested action: {game.suggest_action(win_prob)}")

    my_score = None
    my_best = None
    
    if community_cards:
        my_score, my_best = game.find_best_hand(my_cards, community_cards)
        print(f"\nYour score currently: {my_score}")
        print(f"\nYour best hand currently: {my_best}")
    
    return {
        "win_probability": win_prob,
        "best_hand": my_best
    }

if __name__ == "__main__":
    print("Poker Win Probability Calculator")
    print("--------------------------------")
    print("Enter cards in format: rank+suit (e.g., Ah for Ace of hearts, 10s for 10 of spades)")
    print("Rank: 2-10, J, Q, K, A")
    print("Suit: h (hearts), d (diamonds), c (clubs), s (spades)")

    my_cards_input = input("Your cards:")
    community_input = input("Community cards:")
    opponents_input = input("Opponent cards:")

    asyncio.run(suggest_actions(my_cards_input=my_cards_input, community_input=community_input, opponent_input=opponents_input))