"""
Denoising Autoencoder f├╝r MTG Deck-Empfehlungen

Dieses Modell lernt die Struktur von Magic Decks und kann verwendet werden,
um Deck-Vorschl├ñge zu generieren oder fehlende Karten zu empfehlen.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class DeckDataset(Dataset):
    """Dataset für MTG Decks"""
    
    def __init__(self, deck_matrices: np.ndarray, noise_factor: float = 0.2):
        """
        Args:
            deck_matrices: Binary matrix (n_decks, n_cards) wo 1 = Karte ist im Deck
            noise_factor: Anteil der Karten die für Denoising entfernt werden
        """
        self.decks = torch.FloatTensor(deck_matrices)
        self.noise_factor = noise_factor
    
    def __len__(self):
        return len(self.decks)
    
    def __getitem__(self, idx):
        clean_deck = self.decks[idx]
        
        # Noise hinzuf├╝gen: zuf├ñllig Karten entfernen
        noisy_deck = clean_deck.clone()
        mask = torch.rand_like(noisy_deck) > self.noise_factor
        noisy_deck = noisy_deck * mask
        
        return noisy_deck, clean_deck


class DeckAutoencoder(nn.Module):
    """Denoising Autoencoder für MTG Decks"""
    
    def __init__(self, n_cards: int, embedding_dim: int = 128, 
                 hidden_dims: list = [1024, 512, 256]):
        """
        Args:
            n_cards: Anzahl verschiedener Karten
            embedding_dim: Dimension des Latent Space
            hidden_dims: Dimensionen der Hidden Layers
        """
        super(DeckAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = n_cards
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, n_cards),
            nn.Sigmoid()  # Output zwischen 0 und 1
        ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode ein Deck in den Latent Space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode einen Latent Vector zu einem Deck"""
        return self.decoder(z)


class DeckRecommender:
    """Wrapper für das trainierte Modell für Deck-Empfehlungen"""
    
    def __init__(self, model: DeckAutoencoder, card_metadata: pd.DataFrame):
        self.model = model
        self.model.eval()
        self.metadata = card_metadata
        self.n_cards = len(card_metadata)
    
    def recommend_cards(self, partial_deck: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """
        Empfiehlt Karten basierend auf einem unvollständigen Deck
        
        Args:
            partial_deck: Binary vector (n_cards,) mit 1 für vorhandene Karten
            top_k: Anzahl der Empfehlungen
        
        Returns:
            DataFrame mit empfohlenen Karten und Scores
        """
        with torch.no_grad():
            deck_tensor = torch.FloatTensor(partial_deck).unsqueeze(0)
            reconstruction = self.model(deck_tensor).squeeze(0).numpy()
        
        # Entferne bereits vorhandene Karten
        reconstruction[partial_deck == 1] = 0
        
        # Top-K Karten
        top_indices = np.argsort(reconstruction)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'name': self.metadata.iloc[idx]['name'],
                'score': reconstruction[idx],
                'manaCost': self.metadata.iloc[idx]['manaCost'],
                'type': self.metadata.iloc[idx]['type']
            })
        
        return pd.DataFrame(recommendations)
    
    def find_similar_decks(self, deck: np.ndarray, all_decks: np.ndarray, 
                          top_k: int = 5) -> np.ndarray:
        """
        Findet ähnliche Decks basierend auf Latent Space Distance
        
        Args:
            deck: Binary vector des Referenz-Decks
            all_decks: Matrix aller Decks (n_decks, n_cards)
            top_k: Anzahl ├ñhnlicher Decks
        
        Returns:
            Indices der ├ñhnlichsten Decks
        """
        with torch.no_grad():
            # Encode alle Decks
            deck_tensor = torch.FloatTensor(deck).unsqueeze(0)
            all_decks_tensor = torch.FloatTensor(all_decks)
            
            deck_embedding = self.model.encode(deck_tensor)
            all_embeddings = self.model.encode(all_decks_tensor)
            
            # Berechne Distanzen
            distances = torch.cdist(deck_embedding, all_embeddings).squeeze(0)
            
            # Top-K ├ñhnliche Decks
            top_indices = torch.argsort(distances)[:top_k].numpy()
        
        return top_indices


def create_sample_deck_matrix(n_decks: int, n_cards: int, 
                             deck_size: int = 60) -> np.ndarray:
    """
    Erstellt Sample Deck Daten für Tests
    In der Praxis würden hier echte Deck-Listen geladen werden
    """
    deck_matrix = np.zeros((n_decks, n_cards), dtype=np.float32)
    
    for i in range(n_decks):
        # Zuf├ñllige Karten f├╝r jedes Deck
        card_indices = np.random.choice(n_cards, deck_size, replace=False)
        deck_matrix[i, card_indices] = 1
    
    return deck_matrix


def train_autoencoder(model: DeckAutoencoder, train_loader: DataLoader, 
                     n_epochs: int = 50, lr: float = 0.001) -> list:
    """
    Trainiert den Autoencoder
    
    Returns:
        Liste mit Loss-Werten pro Epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for noisy_deck, clean_deck in train_loader:
            noisy_deck = noisy_deck.to(device)
            clean_deck = clean_deck.to(device)
            
            # Forward pass
            reconstructed = model(noisy_deck)
            loss = criterion(reconstructed, clean_deck)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
    return losses


if __name__ == "__main__":
    print("=== MTG Deck Autoencoder Demo ===\n")
    
    # Lade Embeddings
    print("Lade Card Embeddings...")
    embeddings = np.load('data/embeddings/card_embeddings.npy')
    metadata = pd.read_csv('data/embeddings/card_metadata.csv')
    n_cards = len(metadata)
    
    print(f"Anzahl Karten: {n_cards}")
    
    # Erstelle Sample Deck Daten
    # In der Praxis: Lade echte Deck-Listen hier
    print("\nErstelle Sample Deck Daten...")
    n_decks = 100
    deck_matrix = create_sample_deck_matrix(n_decks, n_cards, deck_size=60)
    print(f"Anzahl Sample Decks: {n_decks}")
    
    # Dataset und DataLoader
    dataset = DeckDataset(deck_matrix, noise_factor=0.2)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Modell erstellen
    print("\nErstelle Autoencoder Modell...")
    model = DeckAutoencoder(
        n_cards=n_cards,
        embedding_dim=128,
        hidden_dims=[1024, 512, 256]
    )
    
    print(f"Modell Parameter: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\nStarte Training...")
    losses = train_autoencoder(model, train_loader, n_epochs=20, lr=0.001)
    
    # Speichere Modell
    print("\nSpeichere trainiertes Modell...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_cards': n_cards,
        'embedding_dim': 128,
    }, 'data/embeddings/deck_autoencoder.pth')
    
    print("\n=== Training abgeschlossen! ===")
    print("\nN├ñchste Schritte:")
    print("1. Lade echte Deck-Daten (z.B. von MTGO, Arena, oder Turnieren)")
    print("2. Trainiere das Modell mit echten Decks")
    print("3. Nutze DeckRecommender f├╝r Empfehlungen")
