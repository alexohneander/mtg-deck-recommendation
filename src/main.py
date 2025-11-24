"""
MTG Deck Recommendation System - Main CLI
Zentrales Command-Line Interface für alle Workflow-Schritte
"""

import argparse
import sys
import os
from pathlib import Path

# Füge src Verzeichnis zum Python Path hinzu
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
    """Indexiert MTG Karten und erstellt Embeddings"""
    print("=" * 70)
    print("STEP 1: Card Indexierung")
    print("=" * 70)
    
    embedder = MTGCardEmbedder(db_path=args.db_path)
    
    # Vokabulare aufbauen
    print("\n[1/3] Erstelle Vokabulare...")
    start_time = time.time()
    embedder.build_vocabularies(sample_size=args.sample_vocab)
    vocab_time = time.time() - start_time
    print(f"✓ Vokabular erstellt in {vocab_time:.2f}s")
    
    # Embeddings erstellen
    print("\n[2/3] Verarbeite Karten und erstelle Embeddings...")
    start_time = time.time()
    embeddings, metadata = embedder.process_all_cards(
        batch_size=args.batch_size,
        limit=args.limit
    )
    processing_time = time.time() - start_time
    print(f"✓ {len(metadata):,} Karten verarbeitet in {processing_time:.2f}s")
    print(f"  ({len(metadata) / processing_time:.1f} Karten/Sekunde)")
    
    # Embeddings speichern
    print("\n[3/3] Speichere Embeddings...")
    embedder.save_embeddings(embeddings, metadata, output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("✓ Indexierung abgeschlossen!")
    print(f"  Anzahl Karten:       {len(metadata):,}")
    print(f"  Feature Dimensionen: {embeddings.shape[1]}")
    print(f"  Output Verzeichnis:  {args.output_dir}")
    print("=" * 70)


def cmd_create_sample_decks(args):
    """Erstellt Beispiel Deck-Dateien"""
    print("=" * 70)
    print("STEP 2: Beispiel Decks erstellen")
    print("=" * 70)
    
    create_example_decklists()
    
    print("\n✓ Beispiel Decks erstellt in data/sample_decks/")
    print("  - mono_red.txt")
    print("  - blue_control.txt")
    print("\nDu kannst jetzt eigene Decks im gleichen Format hinzufügen!")


def cmd_load_decks(args):
    """Lädt Deck-Daten und konvertiert sie zu Vektoren"""
    print("=" * 70)
    print("STEP 3: Deck-Daten laden")
    print("=" * 70)
    
    # Lade Metadata
    print("\nLade Card Metadata...")
    metadata = pd.read_csv(os.path.join(args.embeddings_dir, 'card_metadata.csv'))
    print(f"✓ {len(metadata):,} Karten geladen")
    
    # Erstelle Loader
    loader = DeckLoader(metadata)
    
    # Finde alle Deck-Dateien
    deck_files = []
    deck_dir = Path(args.deck_dir)
    
    if deck_dir.exists():
        deck_files = list(deck_dir.glob('*.txt'))
        print(f"\n✓ {len(deck_files)} Deck-Dateien gefunden in {args.deck_dir}")
    else:
        print(f"\n⚠ Verzeichnis {args.deck_dir} existiert nicht!")
        print("  Erstelle erst Beispiel-Decks mit: python src/main.py create-samples")
        return
    
    # Lade Decks
    if deck_files:
        print("\nLade Decks...")
        deck_matrix = loader.load_multiple_decks(
            [str(f) for f in deck_files],
            use_counts=args.use_counts
        )
        
        # Speichere Deck-Matrix
        output_file = os.path.join(args.embeddings_dir, 'deck_matrix.npy')
        np.save(output_file, deck_matrix)
        
        print(f"\n✓ Deck-Matrix gespeichert: {output_file}")
        print(f"  Shape: {deck_matrix.shape}")
        print(f"  (n_decks={deck_matrix.shape[0]}, n_cards={deck_matrix.shape[1]})")


def cmd_train(args):
    """Trainiert den Denoising Autoencoder"""
    print("=" * 70)
    print("STEP 4: Autoencoder Training")
    print("=" * 70)
    
    # Lade Metadata
    metadata_path = os.path.join(args.embeddings_dir, 'card_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    n_cards = len(metadata)
    print(f"\n✓ {n_cards:,} Karten geladen")
    
    # Lade oder erstelle Deck-Daten
    deck_matrix_path = os.path.join(args.embeddings_dir, 'deck_matrix.npy')
    
    if os.path.exists(deck_matrix_path) and not args.use_sample_data:
        print(f"\nLade Deck-Matrix von {deck_matrix_path}...")
        deck_matrix = np.load(deck_matrix_path)
        print(f"✓ {len(deck_matrix):,} Decks geladen")
    else:
        if args.use_sample_data:
            print("\nErstelle Sample Deck-Daten für Demo...")
        else:
            print(f"\n⚠ Keine Deck-Matrix gefunden in {deck_matrix_path}")
            print("  Erstelle Sample Daten für Demo...")
        
        deck_matrix = create_sample_deck_matrix(
            n_decks=args.n_sample_decks,
            n_cards=n_cards,
            deck_size=60
        )
        print(f"✓ {len(deck_matrix)} Sample Decks erstellt")
    
    # Dataset und DataLoader
    print(f"\nErstelle Dataset (noise_factor={args.noise_factor})...")
    dataset = DeckDataset(deck_matrix, noise_factor=args.noise_factor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Modell erstellen
    print(f"\nErstelle Autoencoder (embedding_dim={args.embedding_dim})...")
    model = DeckAutoencoder(
        n_cards=n_cards,
        embedding_dim=args.embedding_dim,
        hidden_dims=[1024, 512, 256]
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Modell erstellt mit {n_params:,} Parametern")
    
    # Training
    print(f"\nStarte Training ({args.epochs} Epochs)...")
    print("-" * 70)
    
    losses = train_autoencoder(
        model, 
        train_loader, 
        n_epochs=args.epochs,
        lr=args.learning_rate
    )
    
    # Speichere Modell
    print("\n" + "-" * 70)
    output_path = os.path.join(args.embeddings_dir, 'deck_autoencoder.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_cards': n_cards,
        'embedding_dim': args.embedding_dim,
        'losses': losses
    }, output_path)
    
    print(f"\n✓ Modell gespeichert: {output_path}")
    print(f"  Final Loss: {losses[-1]:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Training abgeschlossen!")
    print("=" * 70)


def cmd_recommend(args):
    """Generiert Deck-Empfehlungen"""
    print("=" * 70)
    print("STEP 5: Deck-Empfehlungen")
    print("=" * 70)
    
    # Lade Metadata
    metadata_path = os.path.join(args.embeddings_dir, 'card_metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    # Lade Modell
    model_path = os.path.join(args.embeddings_dir, 'deck_autoencoder.pth')
    
    if not os.path.exists(model_path):
        print(f"\n✗ Kein trainiertes Modell gefunden: {model_path}")
        print("  Trainiere erst ein Modell mit: python src/main.py train")
        return
    
    print(f"\nLade Modell von {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = DeckAutoencoder(
        n_cards=checkpoint['n_cards'],
        embedding_dim=checkpoint['embedding_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Modell geladen")
    
    # Erstelle Recommender
    recommender = DeckRecommender(model, metadata)
    
    # Lade Deck oder erstelle leeres Deck
    if args.deck_file:
        print(f"\nLade Deck von {args.deck_file}...")
        loader = DeckLoader(metadata)
        partial_deck = loader.load_deck_from_file(args.deck_file)
        
        # Zeige geladenes Deck
        decklist = loader.vector_to_decklist(partial_deck, threshold=0.5)
        print(f"\n✓ Deck geladen mit {len(decklist)} Karten:")
        print(decklist[['name', 'manaCost', 'type']].to_string(index=False))
    else:
        print("\nKein Deck angegeben - erstelle leeres Deck für allgemeine Empfehlungen")
        partial_deck = np.zeros(len(metadata), dtype=np.float32)
    
    # Generiere Empfehlungen
    print(f"\nGeneriere Top-{args.top_k} Empfehlungen...")
    recommendations = recommender.recommend_cards(partial_deck, top_k=args.top_k)
    
    print("\n" + "=" * 70)
    print("EMPFOHLENE KARTEN:")
    print("=" * 70)
    print(recommendations.to_string(index=False))
    
    # Speichere optional
    if args.output:
        recommendations.to_csv(args.output, index=False)
        print(f"\n✓ Empfehlungen gespeichert in: {args.output}")


def cmd_full_pipeline(args):
    """Führt den kompletten Workflow aus"""
    print("=" * 70)
    print("VOLLSTÄNDIGER WORKFLOW")
    print("=" * 70)
    print("\nFühre alle Schritte automatisch aus...\n")
    
    # Step 1: Indexierung
    cmd_index_cards(args)
    print("\n")
    
    # Step 2: Sample Decks
    if args.create_samples:
        cmd_create_sample_decks(args)
        print("\n")
    
    # Step 3: Training
    cmd_train(args)
    print("\n")
    
    # Step 4: Empfehlungen (optional)
    if args.show_recommendations:
        cmd_recommend(args)
    
    print("\n" + "=" * 70)
    print("✓ KOMPLETTER WORKFLOW ABGESCHLOSSEN!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='MTG Deck Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Alle Karten indexieren
  python src/main.py index
  
  # Nur 1000 Karten indexieren (zum Testen)
  python src/main.py index --limit 1000
  
  # Beispiel-Decks erstellen
  python src/main.py create-samples
  
  # Decks laden
  python src/main.py load-decks --deck-dir data/sample_decks
  
  # Autoencoder trainieren
  python src/main.py train --epochs 50
  
  # Empfehlungen für ein Deck
  python src/main.py recommend --deck-file data/sample_decks/mono_red.txt
  
  # Kompletter Workflow
  python src/main.py full-pipeline --limit 1000 --epochs 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Befehle')
    
    # INDEX Command
    index_parser = subparsers.add_parser('index', help='Indexiere MTG Karten')
    index_parser.add_argument('--db-path', default='data/AllPrintings.sqlite',
                            help='Pfad zur SQLite Datenbank')
    index_parser.add_argument('--limit', type=int, default=None,
                            help='Max. Anzahl Karten (für Tests)')
    index_parser.add_argument('--batch-size', type=int, default=5000,
                            help='Batch Size für Verarbeitung')
    index_parser.add_argument('--output-dir', default='data/embeddings',
                            help='Output Verzeichnis')
    index_parser.add_argument('--sample-vocab', type=int, default=10000,
                            help='Sample Size für Vokabular')
    
    # CREATE-SAMPLES Command
    subparsers.add_parser('create-samples', help='Erstelle Beispiel Decks')
    
    # LOAD-DECKS Command
    load_parser = subparsers.add_parser('load-decks', help='Lade Deck-Daten')
    load_parser.add_argument('--deck-dir', default='data/sample_decks',
                           help='Verzeichnis mit Deck-Dateien')
    load_parser.add_argument('--embeddings-dir', default='data/embeddings',
                           help='Verzeichnis mit Card Embeddings')
    load_parser.add_argument('--use-counts', action='store_true',
                           help='Verwende Kartenanzahl statt Binary')
    
    # TRAIN Command
    train_parser = subparsers.add_parser('train', help='Trainiere Autoencoder')
    train_parser.add_argument('--embeddings-dir', default='data/embeddings',
                            help='Verzeichnis mit Embeddings')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='Anzahl Training Epochs')
    train_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch Size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                            help='Learning Rate')
    train_parser.add_argument('--embedding-dim', type=int, default=128,
                            help='Latent Space Dimension')
    train_parser.add_argument('--noise-factor', type=float, default=0.2,
                            help='Noise Factor für Denoising')
    train_parser.add_argument('--use-sample-data', action='store_true',
                            help='Verwende generierte Sample-Daten')
    train_parser.add_argument('--n-sample-decks', type=int, default=100,
                            help='Anzahl Sample Decks (wenn --use-sample-data)')
    
    # RECOMMEND Command
    recommend_parser = subparsers.add_parser('recommend', help='Generiere Empfehlungen')
    recommend_parser.add_argument('--deck-file', type=str,
                                help='Pfad zu Deck-Datei')
    recommend_parser.add_argument('--embeddings-dir', default='data/embeddings',
                                help='Verzeichnis mit Embeddings und Modell')
    recommend_parser.add_argument('--top-k', type=int, default=10,
                                help='Anzahl Empfehlungen')
    recommend_parser.add_argument('--output', type=str,
                                help='Output Datei für Empfehlungen')
    
    # FULL-PIPELINE Command
    pipeline_parser = subparsers.add_parser('full-pipeline', 
                                          help='Führe kompletten Workflow aus')
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
                                help='Erstelle Beispiel-Decks')
    pipeline_parser.add_argument('--show-recommendations', action='store_true',
                                help='Zeige Empfehlungen am Ende')
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
