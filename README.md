# MTG Deck Recommendation System

A machine learning-based recommendation system for Magic: The Gathering decks that uses Denoising Autoencoders to generate deck suggestions.

## Overview

This system:
1. **Indexes** Magic cards from the AllPrintings.sqlite database as numerical vectors
2. **Trains** a Denoising Autoencoder with Pro Deck data
3. **Recommends** cards based on incomplete decks

## Features

- 🎴 **Card Embedding System**: Extracts 32 features per card (Mana Cost, Colors, Types, Power/Toughness, etc.)
- 🧠 **Denoising Autoencoder**: Learns deck archetypes and synergies
- 📊 **Deck Analysis**: Finds similar decks based on Latent Space Distance
- 🔍 **Card Recommendations**: Suggests suitable cards for incomplete decks

## Quick Start

### 1. Installation

```bash
# Create virtual environment (if not already created)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Index cards

```bash
# Test with 1000 cards
python src/index_cards.py --limit 1000

# Index all cards (~107,000 cards, takes ~2 minutes)
python src/index_cards.py
```

### 3. Prepare deck data

```bash
# Create example decks (for testing)
python src/deck_loader.py

# Create your own decks in this format:
# data/decks/my_deck.txt:
# 4 Lightning Bolt
# 4 Monastery Swiftspear
# ...
```

### 4. Train Autoencoder

```bash
# Demo with sample data
python src/deck_autoencoder.py
```

## Project Structure

```
mtg-deck-recommendation/
├── data/
│   ├── AllPrintings.sqlite          # MTG JSON Database
│   ├── embeddings/                   # Generated Card Embeddings
│   │   ├── card_embeddings.npy
│   │   ├── card_metadata.csv
│   │   └── feature_names.txt
│   └── sample_decks/                 # Example Deck Lists
├── src/
│   ├── main.py                       # Original Script
│   ├── card_embeddings.py            # Card Embedding Generator
│   ├── index_cards.py                # Indexing Script
│   ├── deck_autoencoder.py           # Autoencoder Model
│   └── deck_loader.py                # Deck Data Loader
├── requirements.txt
├── README.md
└── EMBEDDING_README.md               # Detailed Embedding Documentation
```

## Usage

### Index cards

```python
from src.card_embeddings import MTGCardEmbedder

embedder = MTGCardEmbedder()
embeddings, metadata = embedder.process_all_cards(limit=1000)
embedder.save_embeddings(embeddings, metadata)
```

### Load decks

```python
from src.deck_loader import DeckLoader
import pandas as pd

metadata = pd.read_csv('data/embeddings/card_metadata.csv')
loader = DeckLoader(metadata)

# Load single deck
deck_vector = loader.load_deck_from_file('data/sample_decks/mono_red.txt')

# Load multiple decks
deck_files = ['deck1.txt', 'deck2.txt', 'deck3.txt']
deck_matrix = loader.load_multiple_decks(deck_files)
```

### Get deck recommendations

```python
from src.deck_autoencoder import DeckRecommender
import torch

# Load model
checkpoint = torch.load('data/embeddings/deck_autoencoder.pth')
model = DeckAutoencoder(n_cards=checkpoint['n_cards'])
model.load_state_dict(checkpoint['model_state_dict'])

# Create recommender
recommender = DeckRecommender(model, metadata)

# Get recommendations for incomplete deck
partial_deck = loader.load_deck_from_file('my_partial_deck.txt')
recommendations = recommender.recommend_cards(partial_deck, top_k=10)
print(recommendations)
```

## Extracted Features

Each card is converted to a **32-dimensional** vector:

- **Mana Cost** (7): Generic, W, U, B, R, G, C
- **Mana Value** (1): CMC
- **Color Identity** (6): W, U, B, R, G, Colorless
- **Card Types** (8): Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle
- **Stats** (3): Power, Toughness, Loyalty
- **Rarity** (4): Common, Uncommon, Rare, Mythic
- **Other** (3): Keyword Count, Reserved, Alt Deck Limit

See [EMBEDDING_README.md](EMBEDDING_README.md) for details

## Performance

- **Indexing**: ~1500 cards/second
- **Full indexing**: ~90 seconds for 107,000 cards
- **Memory requirement**: ~14MB for all card embeddings

## Next Steps

1. ✅ Create card embeddings
2. ✅ Implement Denoising Autoencoder
3. ⏳ Collect real Pro Deck data
4. ⏳ Train model with real data
5. ⏳ Build web interface for deck recommendations

## License

See LICENSE file
