"""
Denoising Autoencoder f├╝r MTG Deck-Empfehlungen

Dieses Modell lernt die Struktur von Magic Decks und kann verwendet werden,
um Deck-Vorschl├ñge zu generieren oder fehlende Karten zu empfehlen.

GPU-Beschleunigung:
- Unterst├╝tzt CUDA f├╝r NVIDIA GPUs
- Mixed Precision Training f├╝r bessere Performance
- Optimiertes DataLoader Setup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def get_device(verbose: bool = True):
    """
    Ermittelt das beste verfürgbare Device (CUDA GPU oder CPU)
    
    Args:
        verbose: Wenn True, gibt Geräte-Informationen aus
    
    Returns:
        torch.device: Das zu verwendende Device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print(f"✔  GPU verfügbar: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   Aktuelle GPU Memory Nutzung: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    else:
        device = torch.device('cpu')
        if verbose:
            print("✘ Keine GPU verfügbar, nutze CPU")
    
    return device


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
            # Ausgabe als Logits (nicht mit Sigmoid aktivieren).
            # Wir verwenden BCEWithLogitsLoss während des Trainings
            # und wenden Sigmoid nur bei der Inferenz an.
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
    
    def __init__(self, model: DeckAutoencoder, card_metadata: pd.DataFrame, device=None):
        self.device = device if device is not None else get_device(verbose=False)
        self.model = model.to(self.device)
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
            deck_tensor = torch.FloatTensor(partial_deck).unsqueeze(0).to(self.device)
            # Modell gibt Logits zurück; wandle in Wahrscheinlichkeiten um
            reconstruction = torch.sigmoid(self.model(deck_tensor)).squeeze(0).cpu().numpy()
        
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
            deck_tensor = torch.FloatTensor(deck).unsqueeze(0).to(self.device)
            all_decks_tensor = torch.FloatTensor(all_decks).to(self.device)
            
            deck_embedding = self.model.encode(deck_tensor)
            all_embeddings = self.model.encode(all_decks_tensor)
            
            # Berechne Distanzen
            distances = torch.cdist(deck_embedding, all_embeddings).squeeze(0)
            
            # Top-K ├ñhnliche Decks
            top_indices = torch.argsort(distances)[:top_k].cpu().numpy()
        
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
                     n_epochs: int = 50, lr: float = 0.001, 
                     use_amp: bool = True, device=None) -> list:
    """
    Trainiert den Autoencoder mit GPU-Unterst├╝tzung
    
    Args:
        model: Das zu trainierende Modell
        train_loader: DataLoader mit Trainingsdaten
        n_epochs: Anzahl der Trainings-Epochen
        lr: Learning Rate
        use_amp: Nutze Automatic Mixed Precision f├╝r schnelleres Training (nur GPU)
        device: Device zum Training (None = automatisch ermitteln)
    
    Returns:
        Liste mit Loss-Werten pro Epoch
    """
    # Device Setup
    if device is None:
        device = get_device(verbose=True)
    
    model = model.to(device)
    
    # Mixed Precision nur auf GPU verwenden
    use_amp = use_amp and torch.cuda.is_available()
    if use_amp:
        print("ÔťĽ Mixed Precision Training aktiviert")
        scaler = GradScaler()
    
    # Verwende numerisch stabile Variante, die Logits erwartet
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    print(f"\nStarte Training auf {device}...")
    print(f"Epochen: {n_epochs}, Batch Size: {train_loader.batch_size}, Learning Rate: {lr}\n")
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = len(train_loader)
        
        for batch_idx, (noisy_deck, clean_deck) in enumerate(train_loader):
            noisy_deck = noisy_deck.to(device)
            clean_deck = clean_deck.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed Precision Training
                with autocast():
                    reconstructed = model(noisy_deck)
                    loss = criterion(reconstructed, clean_deck)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard Training
                reconstructed = model(noisy_deck)
                loss = criterion(reconstructed, clean_deck)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Ausgabe alle 10 Epochen oder bei letzter Epoche
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(device) / 1e9
                print(f"Epoch [{epoch+1:3d}/{n_epochs}] | Loss: {avg_loss:.4f} | GPU Memory: {gpu_mem:.2f} GB")
            else:
                print(f"Epoch [{epoch+1:3d}/{n_epochs}] | Loss: {avg_loss:.4f}")
    
    # GPU Memory freigeben
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return losses


if __name__ == "__main__":
    print("=" * 60)
    print("MTG Deck Autoencoder - GPU-accelerated Training")
    print("=" * 60)
    print()

    # Determine device
    device = get_device(verbose=True)
    print()

    # Load embeddings
    print("Loading card embeddings...")
    embeddings = np.load('data/embeddings/card_embeddings.npy')
    metadata = pd.read_csv('data/embeddings/card_metadata.csv')
    n_cards = len(metadata)
    
    print(f"Number of cards: {n_cards}")

    # Create sample deck data
    # In practice: load real deck lists here
    print("\nCreating sample deck data...")
    n_decks = 1000  # More decks for better training
    deck_matrix = create_sample_deck_matrix(n_decks, n_cards, deck_size=60)
    print(f"Number of sample decks: {n_decks}")

    # Dataset and DataLoader with GPU optimizations
    dataset = DeckDataset(deck_matrix, noise_factor=0.2)
    
    # Larger batch size for GPU, smaller for CPU
    batch_size = 64 if torch.cuda.is_available() else 16

    # pin_memory and num_workers for faster GPU transfer
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Modell erstellen
    print("\nCreating autoencoder model...")
    model = DeckAutoencoder(
        n_cards=n_cards,
        embedding_dim=128,
        hidden_dims=[1024, 512, 256]
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: ~{n_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Training mit GPU-Unterst├╝tzung
    losses = train_autoencoder(
        model, 
        train_loader, 
        n_epochs=50,  # Mehr Epochen da wir GPU haben
        lr=0.001,
        use_amp=True,  # Mixed precision for faster training
        device=device
    )
    
    # Speichere Modell
    print("\nSaving trained model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_cards': n_cards,
        'embedding_dim': 128,
        'hidden_dims': [1024, 512, 256],
    }, 'data/embeddings/deck_autoencoder.pth')
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load real deck data (e.g., from MTGO, Arena, or tournaments)")
    print("2. Train the model with real decks")
    print("3. Use DeckRecommender for recommendations")
    print(f"4. Model saved at: data/embeddings/deck_autoencoder.pth")
