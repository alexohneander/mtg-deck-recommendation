# MTG Card Embedding Indexer

Dieses System erstellt Vektorrepr├ñsentationen (Embeddings) von Magic The Gathering Karten aus der AllPrintings.sqlite Datenbank, um sie f├╝r Machine Learning Modelle wie Denoising Autoencoder nutzbar zu machen.

## ├£bersicht

Das System extrahiert relevante Features aus jeder Karte und erstellt numerische Vektoren, die f├╝r:
- Denoising Autoencoder Training
- Deck-Empfehlungssysteme
- Kartensuche und ├ähnlichkeitsanalysen

genutzt werden k├╢nnen.

## Extrahierte Features

Jede Karte wird in einen **33-dimensionalen** Feature-Vektor umgewandelt:

### 1. Mana Cost Features (7)
- Generic Mana
- White (W), Blue (U), Black (B), Red (R), Green (G)
- Colorless (C)

### 2. Mana Value (1)
- Converted Mana Cost (CMC)

### 3. Color Identity (6)
- One-hot Encoding f├╝r W, U, B, R, G, Colorless

### 4. Card Types (8)
- Binary Features: Creature, Instant, Sorcery, Enchantment, Artifact, Planeswalker, Land, Battle

### 5. Combat Stats (3)
- Power
- Toughness
- Loyalty (f├╝r Planeswalker)

### 6. Rarity (4)
- Common, Uncommon, Rare, Mythic

### 7. Weitere Features (4)
- Keyword Count
- Is Reserved
- Has Alternative Deck Limit

## Installation

```bash
# Dependencies installieren
pip install -r requirements.txt
```

## Verwendung

### Schnellstart - Test mit 1000 Karten

```bash
python src/index_cards.py --limit 1000
```

### Alle Karten indexieren (~107.000 Karten)

```bash
python src/index_cards.py
```

### Weitere Optionen

```bash
python src/index_cards.py --help

# Beispiele:
python src/index_cards.py --limit 5000 --batch-size 1000
python src/index_cards.py --output-dir data/my_embeddings
```

## Output

Das System erstellt folgende Dateien im `data/embeddings/` Verzeichnis:

1. **card_embeddings.npy** - Numpy Array mit allen Embedding-Vektoren
2. **card_metadata.csv** - Metadata (UUID, Name, Set, etc.)
3. **feature_names.txt** - Namen aller Features

## Embeddings Laden und Verwenden

```python
from card_embeddings import MTGCardEmbedder
import numpy as np
import pandas as pd

# Embeddings laden
embeddings = np.load('data/embeddings/card_embeddings.npy')
metadata = pd.read_csv('data/embeddings/card_metadata.csv')

print(f"Shape: {embeddings.shape}")  # (n_cards, 33)
print(f"Erste Karte: {metadata.iloc[0]['name']}")
print(f"Embedding: {embeddings[0]}")
```

## N├ñchste Schritte: Denoising Autoencoder

Die erstellten Embeddings k├╢nnen f├╝r einen Denoising Autoencoder verwendet werden:

1. **Deck-Daten sammeln**: Lade fertige Pro-Decks (z.B. von MTGO, Arena, oder Tournament-Daten)
2. **Deck-Matrix erstellen**: Erstelle bin├ñre Vektoren die repr├ñsentieren welche Karten in einem Deck sind
3. **Autoencoder trainieren**: 
   - Input: Deck-Vektor mit "Noise"
   - Output: Originales Deck
   - Latent Space lernt Deck-Archetypen
4. **Deck-Empfehlungen**: Nutze den Decoder um fehlende Karten vorzuschlagen

### Beispiel Autoencoder Architektur

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

- **Verarbeitungsgeschwindigkeit**: ~1000-2000 Karten/Sekunde
- **Speicherbedarf**: ~14MB f├╝r 107.000 Karten (float32)
- **Vollst├ñndige Indexierung**: ~60-120 Sekunden

## Datenbank Schema

Die Daten stammen aus der MTG JSON AllPrintings.sqlite Datenbank:
- Download: https://mtgjson.com/downloads/all-files/
- Version: Aktuelle AllPrintings.sqlite

## Lizenz

Siehe LICENSE Datei.

## Beitragende

Erstellt f├╝r MTG Deck Recommendation System.
