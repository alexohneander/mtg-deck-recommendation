# MTG Card Embedding Indexer

This system creates vector representations (embeddings) of Magic The Gathering cards from the AllPrintings.sqlite database to make them usable for machine learning models like Denoising Autoencoders.

## Overview

The system extracts relevant features from each card and creates numerical vectors that can be used for:
- Denoising Autoencoder training
- Deck recommendation systems
- Card search and similarity analysis

## Extracted Features

Each card is converted to a **33-dimensional** feature vector:

### 1. Mana Cost Features (7)
- Generic Mana
- White (W), Blue (U), Black (B), Red (R), Green (G)
- Colorless (C)

### 2. Mana Value (1)
- Converted Mana Cost (CMC)

### 3. Color Identity (6)
- One-hot Encoding for W, U, B, R, G, Colorless

### 4. Card Types (8)
- Binary Features: Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle

### 5. Combat Stats (3)
- Power
- Toughness
- Loyalty (for Planeswalkers)

### 6. Rarity (4)
- Common, Uncommon, Rare, Mythic

### 7. Additional Features (4)
- Keyword Count
- Is Reserved
- Has Alternative Deck Limit

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start - Test with 1000 cards

```bash
python src/index_cards.py --limit 1000
```

### Index all cards (~107,000 cards)

```bash
python src/index_cards.py
```

### Additional Options

```bash
python src/index_cards.py --help

# Examples:
python src/index_cards.py --limit 5000 --batch-size 1000
python src/index_cards.py --output-dir data/my_embeddings
```

## Output

The system creates the following files in the `data/embeddings/` directory:

1. **card_embeddings.npy** - Numpy array with all embedding vectors
2. **card_metadata.csv** - Metadata (UUID, Name, Set, etc.)
3. **feature_names.txt** - Names of all features

## Loading and Using Embeddings

```python
from card_embeddings import MTGCardEmbedder
import numpy as np
import pandas as pd

# Load embeddings
embeddings = np.load('data/embeddings/card_embeddings.npy')
metadata = pd.read_csv('data/embeddings/card_metadata.csv')

print(f"Shape: {embeddings.shape}")  # (n_cards, 33)
print(f"First card: {metadata.iloc[0]['name']}")
print(f"Embedding: {embeddings[0]}")
```

## Next Steps: Denoising Autoencoder

The created embeddings can be used for a Denoising Autoencoder:

1. **Collect deck data**: Load ready-made pro-decks (e.g., from MTGO, Arena, or tournament data)
2. **Create deck matrix**: Create binary vectors representing which cards are in a deck
3. **Train Autoencoder**: 
   - Input: Deck vector with "noise"
   - Output: Original deck
   - Latent space learns deck archetypes
4. **Deck recommendations**: Use the decoder to suggest missing cards

### Example Autoencoder Architecture

```python
import torch
import torch.nn as nn

class DeckAutoencoder(nn.Module):
    def __init__(self, n_cards=107558, embedding_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_cards, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, n_cards),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

## Performance

- **Processing speed**: ~1000-2000 cards/second
- **Memory requirement**: ~14MB for 107,000 cards (float32)
- **Complete indexing**: ~60-120 seconds

## Database Schema

Data comes from the MTG JSON AllPrintings.sqlite database:
- Download: https://mtgjson.com/downloads/all-files/
- Version: Current AllPrintings.sqlite

## License

See LICENSE file.

## Contributors

Created for MTG Deck Recommendation System.
