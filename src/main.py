"""
MTG Deck Recommendation System - Main CLI
Central command-line interface for all workflow steps
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from card_embeddings import MTGCardEmbedder
from deck_loader import DeckLoader, create_example_decklists
from deck_autoencoder import (
    DeckAutoencoder, 
    DeckDataset, 
    DeckRecommender,
    train_autoencoder,
    create_sample_deck_matrix
)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import time


def cmd_index_cards(args):
    """Index MTG cards and create embeddings"""
    print("=" * 70)
    print("STEP 1: Card Indexing")
    print("=" * 70)
    
    embedder = MTGCardEmbedder(db_path=args.db_path)
    
    # Build vocabularies
    print("\n[1/3] Building vocabularies...")
    start_time = time.time()
    embedder.build_vocabularies(sample_size=args.sample_vocab)
    vocab_time = time.time() - start_time
    print(f"✓ Vocabulary created in {vocab_time:.2f}s")
    
    # Create embeddings
    print("\n[2/3] Processing cards and creating embeddings...")
    start_time = time.time()
    embeddings, metadata = embedder.process_all_cards(
        batch_size=args.batch_size,
        limit=args.limit
    )
    processing_time = time.time() - start_time
    print(f"✓ {len(metadata):,} cards processed in {processing_time:.2f}s")
    print(f"  ({len(metadata) / processing_time:.1f} cards/second)")
    
    # Save embeddings
    print("\n[3/3] Saving embeddings...")
    embedder.save_embeddings(embeddings, metadata, output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Indexing completed!")
    print(f"  Number of cards:     {len(metadata):,}")
    print(f"  Feature dimensions:  {embeddings.shape[1]}")
    print(f"  Output directory:    {args.output_dir}")
    print("=" * 70)


def cmd_create_sample_decks(args):
    """Creates example deck files"""
    print("=" * 70)
    print("STEP 2: Create Sample Decks")
    print("=" * 70)
    
    create_example_decklists()
    
    print("\n✓ Sample decks created in data/sample_decks/")
    print("  - mono_red.txt")
    print("  - blue_control.txt")
    print("\nYou can now add your own decks in the same format!")


def cmd_load_decks(args):
    """Loads deck data and converts them to vectors"""
    print("=" * 70)
    print("STEP 3: Load Deck Data")
    print("=" * 70)
    
    # Load metadata
    print("\nLoading card metadata...")
    metadata = pd.read_csv(os.path.join(args.embeddings_dir, 'card_metadata.csv'))
    print(f"✓ {len(metadata):,} cards loaded")
    
    # Create loader
    loader = DeckLoader(metadata)
    
    # Find all deck files
    deck_files = []
    deck_dir = Path(args.deck_dir)
    
    if deck_dir.exists():
        deck_files = list(deck_dir.glob('*.txt'))
        print(f"\n✓ {len(deck_files)} deck files found in {args.deck_dir}")
    else:
        print(f"\n⚠ Directory {args.deck_dir} does not exist!")
        print("  Create sample decks first with: python src/main.py create-samples")
        return
    
    # Load decks
    if deck_files:
        print("\nLoading decks...")
        deck_matrix = loader.load_multiple_decks(
            [str(f) for f in deck_files],
            use_counts=args.use_counts
        )
        
        # Save deck matrix
        output_file = os.path.join(args.embeddings_dir, 'deck_matrix.npy')
        np.save(output_file, deck_matrix)
        
        print(f"\n✓ Deck matrix saved: {output_file}")
        print(f"  Shape: {deck_matrix.shape}")
        print(f"  (n_decks={deck_matrix.shape[0]}, n_cards={deck_matrix.shape[1]})")


def cmd_train(args):
    """Trains the denoising autoencoder"""
    print("=" * 70)
    print("STEP 4: Autoencoder Training")
    print("=" * 70)
    
    # Load metadata
    metadata_path = os.path.join(args.embeddings_dir, 'card_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    n_cards = len(metadata)
    print(f"\n✓ {n_cards:,} cards loaded")
    
    # Load or create deck data
    deck_matrix_path = os.path.join(args.embeddings_dir, 'deck_matrix.npy')
    
    if os.path.exists(deck_matrix_path) and not args.use_sample_data:
        print(f"\nLoading deck matrix from {deck_matrix_path}...")
        deck_matrix = np.load(deck_matrix_path)
        print(f"✓ {len(deck_matrix):,} decks loaded")
    else:
        if args.use_sample_data:
            print("\nCreating sample deck data for demo...")
        else:
            print(f"\n⚠ No deck matrix found at {deck_matrix_path}")
            print("  Creating sample data for demo...")
        
        deck_matrix = create_sample_deck_matrix(
            n_decks=args.n_sample_decks,
            n_cards=n_cards,
            deck_size=60
        )
        print(f"✓ {len(deck_matrix)} sample decks created")
    
    # Dataset and DataLoader
    print(f"\nCreating dataset (noise_factor={args.noise_factor})...")
    dataset = DeckDataset(deck_matrix, noise_factor=args.noise_factor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    print(f"\nCreating autoencoder (embedding_dim={args.embedding_dim})...")
    model = DeckAutoencoder(
        n_cards=n_cards,
        embedding_dim=args.embedding_dim,
        hidden_dims=[1024, 512, 256]
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters")
    
    # Training
    print(f"\nStarting training ({args.epochs} epochs)...")
    print("-" * 70)
    
    losses = train_autoencoder(
        model, 
        train_loader, 
        n_epochs=args.epochs,
        lr=args.learning_rate
    )
    
    # Save model
    print("\n" + "-" * 70)
    output_path = os.path.join(args.embeddings_dir, 'deck_autoencoder.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_cards': n_cards,
        'embedding_dim': args.embedding_dim,
        'losses': losses
    }, output_path)
    
    print(f"\n✓ Model saved: {output_path}")
    print(f"  Final Loss: {losses[-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Training completed!")
    print("=" * 70)


def cmd_recommend(args):
    """Generates deck recommendations"""
    print("=" * 70)
    print("STEP 5: Deck Recommendations")
    print("=" * 70)
    
    # Load metadata
    metadata_path = os.path.join(args.embeddings_dir, 'card_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    # Load model
    model_path = os.path.join(args.embeddings_dir, 'deck_autoencoder.pth')
    
    if not os.path.exists(model_path):
        print(f"\n✗ No trained model found: {model_path}")
        print("  Train a model first with: python src/main.py train")
        return
    
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = DeckAutoencoder(
        n_cards=checkpoint['n_cards'],
        embedding_dim=checkpoint['embedding_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")
    
    # Create recommender
    recommender = DeckRecommender(model, metadata)
    
    # Load deck or create empty deck
    if args.deck_file:
        print(f"\nLoading deck from {args.deck_file}...")
        loader = DeckLoader(metadata)
        partial_deck = loader.load_deck_from_file(args.deck_file)
        
        # Show loaded deck
        decklist = loader.vector_to_decklist(partial_deck, threshold=0.5)
        print(f"\n✓ Deck loaded with {len(decklist)} cards:")
        print(decklist[['name', 'manaCost', 'type']].to_string(index=False))
    else:
        print("\nNo deck provided - creating an empty deck for general recommendations")
        partial_deck = np.zeros(len(metadata), dtype=np.float32)
    
    # Generate recommendations
    print(f"\nGenerating top-{args.top_k} recommendations...")
    recommendations = recommender.recommend_cards(partial_deck, top_k=args.top_k)
    
    print("\n" + "=" * 70)
    print("RECOMMENDED CARDS:")
    print("=" * 70)
    print(recommendations.to_string(index=False))
    
    # Optionally save
    if args.output:
        recommendations.to_csv(args.output, index=False)
        print(f"\n✓ Recommendations saved to: {args.output}")


def cmd_full_pipeline(args):
    """Runs the full workflow"""
    print("=" * 70)
    print("FULL WORKFLOW")
    print("=" * 70)
    print("\nRunning all steps automatically...\n")
    
    # Step 1: Indexing
    cmd_index_cards(args)
    print("\n")
    
    # Step 2: Sample Decks
    if args.create_samples:
        cmd_create_sample_decks(args)
        print("\n")
    
    # Step 3: Training
    cmd_train(args)
    print("\n")
    
    # Step 4: Recommendations (optional)
    if args.show_recommendations:
        cmd_recommend(args)
    
    print("\n" + "=" * 70)
    print("✓ FULL WORKFLOW COMPLETED!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='MTG Deck Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all cards
  python src/main.py index
  
  # Index only 1000 cards (for testing)
  python src/main.py index --limit 1000
  
  # Create sample decks
  python src/main.py create-samples
  
  # Load decks
  python src/main.py load-decks --deck-dir data/sample_decks
  
  # Train the autoencoder
  python src/main.py train --epochs 50
  
  # Get recommendations for a deck
  python src/main.py recommend --deck-file data/sample_decks/mono_red.txt
  
  # Full workflow
  python src/main.py full-pipeline --limit 1000 --epochs 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # INDEX Command
    index_parser = subparsers.add_parser('index', help='Index MTG cards')
    index_parser.add_argument('--db-path', default='data/AllPrintings.sqlite',
                            help='Path to SQLite database')
    index_parser.add_argument('--limit', type=int, default=None,
                            help='Max number of cards (for testing)')
    index_parser.add_argument('--batch-size', type=int, default=5000,
                            help='Batch size for processing')
    index_parser.add_argument('--output-dir', default='data/embeddings',
                            help='Output directory')
    index_parser.add_argument('--sample-vocab', type=int, default=10000,
                            help='Sample size for vocabulary')
    
    # CREATE-SAMPLES Command
    subparsers.add_parser('create-samples', help='Create sample decks')
    
    # LOAD-DECKS Command
    load_parser = subparsers.add_parser('load-decks', help='Load deck data')
    load_parser.add_argument('--deck-dir', default='data/sample_decks',
                           help='Directory with deck files')
    load_parser.add_argument('--embeddings-dir', default='data/embeddings',
                           help='Directory with card embeddings')
    load_parser.add_argument('--use-counts', action='store_true',
                           help='Use card counts instead of binary')
    
    # TRAIN Command
    train_parser = subparsers.add_parser('train', help='Train autoencoder')
    train_parser.add_argument('--embeddings-dir', default='data/embeddings',
                            help='Directory with embeddings')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                            help='Learning rate')
    train_parser.add_argument('--embedding-dim', type=int, default=128,
                            help='Latent space dimension')
    train_parser.add_argument('--noise-factor', type=float, default=0.2,
                            help='Noise factor for denoising')
    train_parser.add_argument('--use-sample-data', action='store_true',
                            help='Use generated sample data')
    train_parser.add_argument('--n-sample-decks', type=int, default=100,
                            help='Number of sample decks (when --use-sample-data)')
    
    # RECOMMEND Command
    recommend_parser = subparsers.add_parser('recommend', help='Generate recommendations')
    recommend_parser.add_argument('--deck-file', type=str,
                                help='Path to deck file')
    recommend_parser.add_argument('--embeddings-dir', default='data/embeddings',
                                help='Directory with embeddings and model')
    recommend_parser.add_argument('--top-k', type=int, default=10,
                                help='Number of recommendations')
    recommend_parser.add_argument('--output', type=str,
                                help='Output file for recommendations')
    
    # FULL-PIPELINE Command
    pipeline_parser = subparsers.add_parser('full-pipeline', 
                                          help='Run the full workflow')
    pipeline_parser.add_argument('--db-path', default='data/AllPrintings.sqlite')
    pipeline_parser.add_argument('--limit', type=int, default=None)
    pipeline_parser.add_argument('--batch-size', type=int, default=5000)
    pipeline_parser.add_argument('--output-dir', default='data/embeddings')
    pipeline_parser.add_argument('--sample-vocab', type=int, default=10000)
    pipeline_parser.add_argument('--epochs', type=int, default=20)
    pipeline_parser.add_argument('--learning-rate', type=float, default=0.001)
    pipeline_parser.add_argument('--embedding-dim', type=int, default=128)
    pipeline_parser.add_argument('--noise-factor', type=float, default=0.2)
    pipeline_parser.add_argument('--use-sample-data', action='store_true')
    pipeline_parser.add_argument('--n-sample-decks', type=int, default=100)
    pipeline_parser.add_argument('--create-samples', action='store_true',
                                help='Create sample decks')
    pipeline_parser.add_argument('--show-recommendations', action='store_true',
                                help='Show recommendations at the end')
    pipeline_parser.add_argument('--top-k', type=int, default=10)
    pipeline_parser.add_argument('--embeddings-dir', default='data/embeddings')
    pipeline_parser.add_argument('--deck-dir', default='data/sample_decks')
    pipeline_parser.add_argument('--use-counts', action='store_true')
    pipeline_parser.add_argument('--deck-file', type=str)
    pipeline_parser.add_argument('--output', type=str)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route zu entsprechender Funktion
    commands = {
        'index': cmd_index_cards,
        'create-samples': cmd_create_sample_decks,
        'load-decks': cmd_load_decks,
        'train': cmd_train,
        'recommend': cmd_recommend,
        'full-pipeline': cmd_full_pipeline
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
