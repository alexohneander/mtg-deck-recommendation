"""
Indexierungs-Script f├╝r MTG Karten Embeddings
F├╝hre dieses Script aus, um alle Karten zu indexieren
"""

import argparse
from card_embeddings import MTGCardEmbedder
import time


def main():
    parser = argparse.ArgumentParser(description='Indexiere MTG Karten als Embedding Vektoren')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Maximale Anzahl an Karten (f├╝r Tests)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch Gr├╢├ƒe f├╝r Verarbeitung')
    parser.add_argument('--output-dir', type=str, default='data/embeddings',
                       help='Output Verzeichnis f├╝r Embeddings')
    parser.add_argument('--sample-vocab', type=int, default=10000,
                       help='Sample Gr├╢├ƒe f├╝r Vokabular-Erstellung')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MTG Card Embedding Indexer")
    print("=" * 60)
    
    # Embedder initialisieren
    embedder = MTGCardEmbedder()
    
    # Vokabulare aufbauen
    print("\n[1/3] Erstelle Vokabulare...")
    start_time = time.time()
    embedder.build_vocabularies(sample_size=args.sample_vocab)
    vocab_time = time.time() - start_time
    print(f"Vokabular-Erstellung abgeschlossen in {vocab_time:.2f} Sekunden")
    
    # Embeddings erstellen
    print("\n[2/3] Verarbeite Karten und erstelle Embeddings...")
    start_time = time.time()
    embeddings, metadata = embedder.process_all_cards(
        batch_size=args.batch_size,
        limit=args.limit
    )
    processing_time = time.time() - start_time
    print(f"Verarbeitung abgeschlossen in {processing_time:.2f} Sekunden")
    print(f"Durchschnitt: {len(metadata) / processing_time:.1f} Karten/Sekunde")
    
    # Embeddings speichern
    print("\n[3/3] Speichere Embeddings...")
    embedder.save_embeddings(embeddings, metadata, output_dir=args.output_dir)
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"Anzahl Karten:        {len(metadata):,}")
    print(f"Feature Dimensionen:  {embeddings.shape[1]}")
    print(f"Embedding Shape:      {embeddings.shape}")
    print(f"Output Verzeichnis:   {args.output_dir}")
    print(f"Gesamtzeit:           {vocab_time + processing_time:.2f} Sekunden")
    print("=" * 60)
    
    print("\n├£ Indexierung erfolgreich abgeschlossen!")
    print("\nN├ñchste Schritte:")
    print("1. Verwende die Embeddings f├╝r den Denoising Autoencoder")
    print("2. Trainiere das Modell mit Deck-Daten")
    print("3. Baue den Deck-Recommendation Helper")


if __name__ == "__main__":
    main()
