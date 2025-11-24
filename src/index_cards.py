"""
Indexing script for MTG card embeddings
Run this script to index all cards
"""

import argparse
from card_embeddings import MTGCardEmbedder
import time


def main():
    parser = argparse.ArgumentParser(description='Index MTG cards as embedding vectors')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Maximum number of cards (for tests)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size for processing')
    parser.add_argument('--output-dir', type=str, default='data/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--sample-vocab', type=int, default=10000,
                       help='Sample size for vocabulary building')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MTG Card Embedding Indexer")
    print("=" * 60)
    
    # Initialize embedder
    embedder = MTGCardEmbedder()
    
    # Build vocabularies
    print("\n[1/3] Building vocabularies...")
    start_time = time.time()
    embedder.build_vocabularies(sample_size=args.sample_vocab)
    vocab_time = time.time() - start_time
    print(f"Vocabulary build finished in {vocab_time:.2f} seconds")
    
    # Create embeddings
    print("\n[2/3] Processing cards and creating embeddings...")
    start_time = time.time()
    embeddings, metadata = embedder.process_all_cards(
        batch_size=args.batch_size,
        limit=args.limit
    )
    processing_time = time.time() - start_time
    print(f"Processing finished in {processing_time:.2f} seconds")
    print(f"Average: {len(metadata) / processing_time:.1f} cards/second")
    
    # Save embeddings
    print("\n[3/3] Saving embeddings...")
    embedder.save_embeddings(embeddings, metadata, output_dir=args.output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of cards:      {len(metadata):,}")
    print(f"Feature dimensions:   {embeddings.shape[1]}")
    print(f"Embedding shape:       {embeddings.shape}")
    print(f"Output directory:     {args.output_dir}")
    print(f"Total time:           {vocab_time + processing_time:.2f} seconds")
    print("=" * 60)
    
    print("\n✓ Indexing completed successfully!")
    print("\nNext steps:")
    print("1. Use the embeddings with the denoising autoencoder")
    print("2. Train the model with deck data")
    print("3. Build the deck recommendation helper")


if __name__ == "__main__":
    main()
