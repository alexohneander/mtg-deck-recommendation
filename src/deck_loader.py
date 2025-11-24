"""
Deck Data Loader
Helper functions for loading and processing real MTG deck data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
import re


class DeckLoader:
    """Class for loading and processing MTG deck data"""
    
    def __init__(self, card_metadata: pd.DataFrame):
        """
        Args:
            card_metadata: DataFrame with card information (uuid, name, setCode)
        """
        self.metadata = card_metadata
        self.card_name_to_idx = {name.lower(): idx for idx, name in enumerate(card_metadata['name'])}
        self.n_cards = len(card_metadata)
    
    def parse_decklist_text(self, decklist_text: str) -> List[Tuple[int, str]]:
        """
        Parse a decklist in the standard format

        Format:
            4 Lightning Bolt
            2 Counterspell

        Returns:
            List of (count, card_name) tuples
        """
        cards = []
        lines = decklist_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Match "4 Card Name" or "4x Card Name"
            match = re.match(r'^(\d+)x?\s+(.+)$', line)
            if match:
                count = int(match.group(1))
                card_name = match.group(2).strip()
                cards.append((count, card_name))
        
        return cards
    
    def decklist_to_vector(self, cards: List[Tuple[int, str]], 
                          use_counts: bool = False) -> np.ndarray:
        """
        Convert a decklist to a binary/count vector

        Args:
            cards: List of (count, card_name) tuples
            use_counts: If True, use counts; if False, binary presence (0/1)

        Returns:
            Vector of size n_cards
        """
        vector = np.zeros(self.n_cards, dtype=np.float32)
        
        for count, card_name in cards:
            card_name_lower = card_name.lower()
            
            if card_name_lower in self.card_name_to_idx:
                idx = self.card_name_to_idx[card_name_lower]
                if use_counts:
                    vector[idx] = count
                else:
                    vector[idx] = 1
            else:
                print(f"Warning: Card not found: {card_name}")
        
        return vector
    
    def load_deck_from_file(self, filepath: str, use_counts: bool = False) -> np.ndarray:
        """Load a deck from a text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            decklist_text = f.read()
        
        cards = self.parse_decklist_text(decklist_text)
        return self.decklist_to_vector(cards, use_counts=use_counts)
    
    def load_multiple_decks(self, filepaths: List[str], 
                           use_counts: bool = False) -> np.ndarray:
        """
        Load multiple decks and return a matrix

        Returns:
            Matrix (n_decks, n_cards)
        """
        deck_vectors = []
        
        for filepath in filepaths:
            try:
                deck_vector = self.load_deck_from_file(filepath, use_counts=use_counts)
                deck_vectors.append(deck_vector)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return np.array(deck_vectors, dtype=np.float32)
    
    def load_from_json(self, json_path: str, use_counts: bool = False) -> np.ndarray:
        """
        Load decks from a JSON file

        Expected format:
        [
            {
                "name": "Deck Name",
                "cards": [
                    {"name": "Lightning Bolt", "count": 4},
                    {"name": "Counterspell", "count": 2}
                ]
            }
        ]
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            decks_data = json.load(f)
        
        deck_vectors = []
        
        for deck in decks_data:
            cards = [(card['count'], card['name']) for card in deck['cards']]
            vector = self.decklist_to_vector(cards, use_counts=use_counts)
            deck_vectors.append(vector)
        
        return np.array(deck_vectors, dtype=np.float32)
    
    def vector_to_decklist(self, vector: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
        """
        Convert a vector back to a decklist

        Args:
            vector: Card vector (n_cards,)
            threshold: Minimum value for a card to be included in the deck

        Returns:
            DataFrame with card overview
        """
        card_indices = np.where(vector >= threshold)[0]
        
        decklist = []
        for idx in card_indices:
            decklist.append({
                'name': self.metadata.iloc[idx]['name'],
                'probability': vector[idx],
                'manaCost': self.metadata.iloc[idx]['manaCost'],
                'type': self.metadata.iloc[idx]['type']
            })
        
        return pd.DataFrame(decklist).sort_values('probability', ascending=False)


def create_example_decklists():
    """Create example deck files for testing"""
    import os
    
    os.makedirs('data/sample_decks', exist_ok=True)
    
    # Example Deck 1: Mono Red Aggro
    mono_red = """
# Mono Red Aggro
4 Lightning Bolt
4 Goblin Guide
4 Monastery Swiftspear
4 Eidolon of the Great Revel
4 Lava Spike
4 Rift Bolt
3 Skullcrack
2 Searing Blaze
4 Light Up the Stage
17 Mountain
4 Sunbaked Canyon
2 Den of the Bugbear
4 Ramunap Ruins
"""
    
    # Example Deck 2: Blue Control
    blue_control = """
# Blue Control
4 Counterspell
3 Cryptic Command
2 Jace, the Mind Sculptor
4 Snapcaster Mage
3 Supreme Verdict
2 Teferi, Hero of Dominaria
4 Opt
3 Archmage's Charm
2 Mystical Dispute
24 Island
4 Celestial Colonnade
3 Hallowed Fountain
2 Field of Ruin
"""
    
    with open('data/sample_decks/mono_red.txt', 'w') as f:
        f.write(mono_red)
    
    with open('data/sample_decks/blue_control.txt', 'w') as f:
        f.write(blue_control)
    
    print("Example decks created in data/sample_decks/")


if __name__ == "__main__":
    print("=== Deck Loader Demo ===\n")

    # Load card metadata
    print("Loading card metadata...")
    metadata = pd.read_csv('data/embeddings/card_metadata.csv')

    # Create DeckLoader
    loader = DeckLoader(metadata)

    # Create example decks
    print("\nCreating example decklists...")
    create_example_decklists()

    # Load example deck
    print("\nLoading example deck...")
    try:
        deck_vector = loader.load_deck_from_file('data/sample_decks/mono_red.txt')
        print(f"Deck Vector Shape: {deck_vector.shape}")
        print(f"Number of cards in deck: {int(deck_vector.sum())}")

        # Show deck contents
        decklist = loader.vector_to_decklist(deck_vector, threshold=0.5)
        print("\nDeck contents:")
        print(decklist.to_string(index=False))
    except FileNotFoundError:
        print("Example deck could not be loaded.")

    print("\n=== Demo finished ===")
    print("\nUsage:")
    print("1. Create deck files in the standard format (see sample_decks/)")
    print("2. Load decks with DeckLoader")
    print("3. Train the autoencoder with real deck data")
