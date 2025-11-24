"""
Denoising Autoencoder for MTG deck recommendations

This model learns the structure of Magic decks and can be used
to generate deck suggestions or recommend missing cards.

GPU acceleration:
- Supports CUDA for NVIDIA GPUs
- Mixed precision training for better performance
- Optimized DataLoader setup
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
    Determine the best available device (CUDA GPU or CPU)

    Args:
        verbose: If True, print device information

    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print(f"✔  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    else:
        device = torch.device('cpu')
        if verbose:
            print("✘ No GPU available, using CPU")

    return device


class DeckDataset(Dataset):
    """Dataset for MTG decks"""

    def __init__(self, deck_matrices: np.ndarray, noise_factor: float = 0.2):
        """
        Args:
            deck_matrices: Binary matrix (n_decks, n_cards) where 1 = card is in deck
            noise_factor: Fraction of cards removed for denoising
        """
        self.decks = torch.FloatTensor(deck_matrices)
        self.noise_factor = noise_factor
    
    def __len__(self):
        return len(self.decks)
    
    def __getitem__(self, idx):
        clean_deck = self.decks[idx]
        
        # Add noise: randomly remove cards
        noisy_deck = clean_deck.clone()
        mask = torch.rand_like(noisy_deck) > self.noise_factor
        noisy_deck = noisy_deck * mask
        
        return noisy_deck, clean_deck


class DeckAutoencoder(nn.Module):
    """Denoising autoencoder for MTG decks"""

    def __init__(self, n_cards: int, embedding_dim: int = 128, 
                 hidden_dims: list = [1024, 512, 256]):
        """
        Args:
            n_cards: Number of distinct cards
            embedding_dim: Dimension of the latent space
            hidden_dims: Dimensions of the hidden layers
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
            # Output as logits (do not apply Sigmoid here).
            # We use BCEWithLogitsLoss during training
            # and apply Sigmoid only at inference time.
        ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode a deck into the latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode a latent vector into a deck"""
        return self.decoder(z)


class DeckRecommender:
    """Wrapper for the trained model used for deck recommendations"""

    def __init__(self, model: DeckAutoencoder, card_metadata: pd.DataFrame, device=None):
        self.device = device if device is not None else get_device(verbose=False)
        self.model = model.to(self.device)
        self.model.eval()
        self.metadata = card_metadata
        self.n_cards = len(card_metadata)

    def recommend_cards(self, partial_deck: np.ndarray, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend cards based on a partial/incomplete deck

        Args:
            partial_deck: Binary vector (n_cards,) with 1 for cards already present
            top_k: Number of recommendations

        Returns:
            DataFrame with recommended cards and scores
        """
        with torch.no_grad():
            deck_tensor = torch.FloatTensor(partial_deck).unsqueeze(0).to(self.device)
            # Model returns logits; convert to probabilities
            reconstruction = torch.sigmoid(self.model(deck_tensor)).squeeze(0).cpu().numpy()

        # Remove cards already present
        reconstruction[partial_deck == 1] = 0

        # Top-K cards
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
        Find similar decks based on latent space distance

        Args:
            deck: Binary vector of the reference deck
            all_decks: Matrix of all decks (n_decks, n_cards)
            top_k: Number of similar decks to return

        Returns:
            Indices of the most similar decks
        """
        with torch.no_grad():
            # Encode decks
            deck_tensor = torch.FloatTensor(deck).unsqueeze(0).to(self.device)
            all_decks_tensor = torch.FloatTensor(all_decks).to(self.device)

            deck_embedding = self.model.encode(deck_tensor)
            all_embeddings = self.model.encode(all_decks_tensor)

            # Compute distances
            distances = torch.cdist(deck_embedding, all_embeddings).squeeze(0)

            # Top-K similar decks
            top_indices = torch.argsort(distances)[:top_k].cpu().numpy()

        return top_indices


def create_sample_deck_matrix(n_decks: int, n_cards: int, 
                             deck_size: int = 60) -> np.ndarray:
    """
    Create sample deck data for testing
    In practice: load real deck lists here
    """
    deck_matrix = np.zeros((n_decks, n_cards), dtype=np.float32)

    for i in range(n_decks):
        # Random cards for each deck
        card_indices = np.random.choice(n_cards, deck_size, replace=False)
        deck_matrix[i, card_indices] = 1

    return deck_matrix


def train_autoencoder(model: DeckAutoencoder, train_loader: DataLoader, 
                     n_epochs: int = 50, lr: float = 0.001, 
                     use_amp: bool = True, device=None) -> list:
    """
    Train the autoencoder with optional GPU support

    Args:
        model: The model to train
        train_loader: DataLoader with training data
        n_epochs: Number of training epochs
        lr: Learning rate
        use_amp: Use automatic mixed precision for faster training (GPU only)
        device: Device to train on (None = auto-detect)

    Returns:
        List of loss values per epoch
    """
    # Device setup
    if device is None:
        device = get_device(verbose=True)

    model = model.to(device)

    # Use mixed precision only on GPU
    use_amp = use_amp and torch.cuda.is_available()
    if use_amp:
        print("Mixed Precision training enabled")
        scaler = GradScaler()

    # Use numerically stable variant that expects logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(f"\nStarting training on {device}...")
    print(f"Epochs: {n_epochs}, Batch Size: {train_loader.batch_size}, Learning Rate: {lr}\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = len(train_loader)

        for batch_idx, (noisy_deck, clean_deck) in enumerate(train_loader):
            noisy_deck = noisy_deck.to(device)
            clean_deck = clean_deck.to(device)

            optimizer.zero_grad()

            if use_amp:
                # Mixed precision training
                with autocast():
                    reconstructed = model(noisy_deck)
                    loss = criterion(reconstructed, clean_deck)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                reconstructed = model(noisy_deck)
                loss = criterion(reconstructed, clean_deck)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        # Print every 10 epochs or on the final epoch
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(device) / 1e9
                print(f"Epoch [{epoch+1:3d}/{n_epochs}] | Loss: {avg_loss:.4f} | GPU Memory: {gpu_mem:.2f} GB")
            else:
                print(f"Epoch [{epoch+1:3d}/{n_epochs}] | Loss: {avg_loss:.4f}")

    # Free GPU memory
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
    
    # Create model
    print("\nCreating autoencoder model...")
    model = DeckAutoencoder(
        n_cards=n_cards,
        embedding_dim=128,
        hidden_dims=[1024, 512, 256]
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Model size: ~{n_params * 4 / 1e6:.1f} MB (FP32)")
    
    # Training with GPU support
    losses = train_autoencoder(
        model, 
        train_loader, 
        n_epochs=50,  # More epochs since we have a GPU
        lr=0.001,
        use_amp=True,  # Mixed precision for faster training
        device=device
    )
    
    # Save model
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
