# MTG Deck Recommendation System

Ein Machine Learning basiertes Empfehlungssystem für Magic: The Gathering Decks, das Denoising Autoencoder verwendet, um Deck-Vorschläge zu generieren.

## Übersicht

Dieses System:
1. **Indexiert** Magic Karten aus der AllPrintings.sqlite Datenbank als numerische Vektoren
2. **Trainiert** einen Denoising Autoencoder mit Pro-Deck-Daten
3. **Empfiehlt** Karten basierend auf unvollständigen Decks

## Features

- 🎴 **Card Embedding System**: Extrahiert 32 Features pro Karte (Mana Cost, Colors, Types, Power/Toughness, etc.)
- 🧠 **Denoising Autoencoder**: Lernt Deck-Archetypen und Synergien
- 📊 **Deck Analysis**: Findet ähnliche Decks basierend auf Latent Space Distance
- 🔍 **Card Recommendations**: Schlägt passende Karten für unvollständige Decks vor

## Schnellstart

### 1. Installation

```bash
# Virtual Environment erstellen (falls noch nicht vorhanden)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Karten indexieren

```bash
# Test mit 1000 Karten
python src/index_cards.py --limit 1000

# Alle Karten indexieren (~107.000 Karten, dauert ~2 Minuten)
python src/index_cards.py
```

### 3. Deck-Daten vorbereiten

```bash
# Beispiel Decks erstellen (zum Testen)
python src/deck_loader.py

# Eigene Decks im Format erstellen:
# data/decks/my_deck.txt:
# 4 Lightning Bolt
# 4 Monastery Swiftspear
# ...
```

### 4. Autoencoder trainieren

```bash
# Demo mit Sample-Daten
python src/deck_autoencoder.py
```

## Projekt-Struktur

```
mtg-deck-recommendation/
├── data/
│   ├── AllPrintings.sqlite          # MTG JSON Datenbank
│   ├── embeddings/                   # Generierte Card Embeddings
│   │   ├── card_embeddings.npy
│   │   ├── card_metadata.csv
│   │   └── feature_names.txt
│   └── sample_decks/                 # Beispiel Deck-Listen
├── src/
│   ├── main.py                       # Original Script
│   ├── card_embeddings.py            # Card Embedding Generator
│   ├── index_cards.py                # Indexierungs-Script
│   ├── deck_autoencoder.py           # Autoencoder Modell
│   └── deck_loader.py                # Deck-Daten Loader
├── requirements.txt
├── README.md
└── EMBEDDING_README.md               # Detaillierte Embedding-Doku
```

## Verwendung

### Karten indexieren

```python
from src.card_embeddings import MTGCardEmbedder

embedder = MTGCardEmbedder()
embeddings, metadata = embedder.process_all_cards(limit=1000)
embedder.save_embeddings(embeddings, metadata)
```

### Decks laden

```python
from src.deck_loader import DeckLoader
import pandas as pd

metadata = pd.read_csv('data/embeddings/card_metadata.csv')
loader = DeckLoader(metadata)

# Einzelnes Deck laden
deck_vector = loader.load_deck_from_file('data/sample_decks/mono_red.txt')

# Mehrere Decks laden
deck_files = ['deck1.txt', 'deck2.txt', 'deck3.txt']
deck_matrix = loader.load_multiple_decks(deck_files)
```

### Deck-Empfehlungen erhalten

```python
from src.deck_autoencoder import DeckRecommender
import torch

# Modell laden
checkpoint = torch.load('data/embeddings/deck_autoencoder.pth')
model = DeckAutoencoder(n_cards=checkpoint['n_cards'])
model.load_state_dict(checkpoint['model_state_dict'])

# Recommender erstellen
recommender = DeckRecommender(model, metadata)

# Empfehlungen für unvollständiges Deck
partial_deck = loader.load_deck_from_file('my_partial_deck.txt')
recommendations = recommender.recommend_cards(partial_deck, top_k=10)
print(recommendations)
```

## Extrahierte Features

Jede Karte wird in einen **32-dimensionalen** Vektor umgewandelt:

- **Mana Cost** (7): Generic, W, U, B, R, G, C
- **Mana Value** (1): CMC
- **Color Identity** (6): W, U, B, R, G, Colorless
- **Card Types** (8): Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle
- **Stats** (3): Power, Toughness, Loyalty
- **Rarity** (4): Common, Uncommon, Rare, Mythic
- **Other** (3): Keyword Count, Reserved, Alt Deck Limit

Details siehe [EMBEDDING_README.md](EMBEDDING_README.md)

## Performance

- **Indexierung**: ~1500 Karten/Sekunde
- **Vollständige Indexierung**: ~90 Sekunden für 107.000 Karten
- **Speicherbedarf**: ~14MB für alle Card Embeddings

## Nächste Schritte

1. ✅ Card Embeddings erstellen
2. ✅ Denoising Autoencoder implementieren
3. ⏳ Echte Pro-Deck Daten sammeln
4. ⏳ Modell mit echten Daten trainieren
5. ⏳ Web-Interface für Deck-Empfehlungen

## Lizenz

Siehe LICENSE Datei
