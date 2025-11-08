#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE Tokenizer Training for Telugu
Implements Byte Pair Encoding from scratch starting with 256 UTF-8 byte tokens
"""

import json
import pickle
import sys
from collections import Counter


def get_stats(ids):
    """
    Count frequency of consecutive byte pairs

    Args:
        ids: List of integers representing tokens

    Returns:
        Dictionary mapping (token1, token2) -> count
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_token_id):
    """
    Replace all occurrences of a pair with a new token

    Args:
        ids: List of token IDs
        pair: Tuple of (token1, token2) to replace
        new_token_id: New token ID to use for the pair

    Returns:
        New list with merged tokens
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # Check if current position matches the pair
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_token_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def train_bpe_tokenizer(text, vocab_size=5500, verbose=True):
    """
    Train a BPE tokenizer on the given text

    Args:
        text: Training text corpus (string)
        vocab_size: Desired vocabulary size (must be > 256)
        verbose: Print training progress

    Returns:
        Dictionary containing:
        - vocab: mapping token_id -> bytes
        - merges: list of merge operations (byte_pair, new_token_id)
        - vocab_size: final vocabulary size
    """

    if vocab_size <= 256:
        raise ValueError("vocab_size must be > 256 (base UTF-8 bytes)")

    if verbose:
        print(f"Training BPE tokenizer on {len(text):,} characters")
        print(f"Target vocabulary size: {vocab_size:,}")
        print("-" * 60)

    # Encode text as UTF-8 bytes
    tokens = list(text.encode("utf-8"))
    num_merges = vocab_size - 256

    if verbose:
        print(f"Initial byte sequence length: {len(tokens):,}")
        print(f"Number of merges to perform: {num_merges:,}")
        print("-" * 60)

    # Initialize vocabulary with base 256 byte tokens
    vocab = {idx: bytes([idx]) for idx in range(256)}

    # Track all merge operations
    merges = {}  # (int, int) -> int
    ids = list(tokens)

    # Perform iterative merging
    for i in range(num_merges):
        # Get pair statistics
        stats = get_stats(ids)

        if not stats:
            if verbose:
                print(f"No more pairs to merge at iteration {i}")
            break

        # Find most frequent pair
        pair = max(stats, key=stats.get)
        new_token_id = 256 + i

        # Merge the pair
        ids = merge(ids, pair, new_token_id)

        # Record the merge
        merges[pair] = new_token_id

        # Add to vocabulary
        vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]

        # Progress reporting
        if verbose and (i + 1) % 100 == 0:
            compression = len(tokens) / len(ids)
            print(f"Merge {i+1:,}/{num_merges:,} | "
                  f"Pair {pair} -> {new_token_id} | "
                  f"Freq: {stats[pair]:,} | "
                  f"Tokens: {len(ids):,} | "
                  f"Compression: {compression:.2f}x")

    final_vocab_size = len(vocab)
    final_compression = len(tokens) / len(ids)

    if verbose:
        print("-" * 60)
        print(f"Training complete!")
        print(f"Final vocabulary size: {final_vocab_size:,}")
        print(f"Final token count: {len(ids):,}")
        print(f"Compression ratio: {final_compression:.2f}x")

    return {
        'vocab': vocab,
        'merges': merges,
        'vocab_size': final_vocab_size,
        'original_length': len(tokens),
        'compressed_length': len(ids),
        'compression_ratio': final_compression
    }


def encode(text, merges):
    """
    Encode text using trained BPE merges

    Args:
        text: Text to encode
        merges: Dictionary of merge rules

    Returns:
        List of token IDs
    """
    # Start with UTF-8 bytes
    tokens = list(text.encode("utf-8"))

    # Apply merges in order
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # Find the pair with the lowest merge index (earliest merge)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))

        if pair not in merges:
            break  # No more merges to apply

        tokens = merge(tokens, pair, merges[pair])

    return tokens


def decode(tokens, vocab):
    """
    Decode token IDs back to text

    Args:
        tokens: List of token IDs
        vocab: Vocabulary dictionary

    Returns:
        Decoded text string
    """
    byte_sequence = b"".join(vocab[token] for token in tokens)
    return byte_sequence.decode("utf-8", errors="replace")


def save_tokenizer(tokenizer_data, vocab_file, merges_file):
    """Save trained tokenizer to files"""

    # Save vocabulary
    vocab_json = {str(k): list(v) for k, v in tokenizer_data['vocab'].items()}
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # Save merges
    with open(merges_file, 'wb') as f:
        pickle.dump(tokenizer_data['merges'], f)

    print(f"Tokenizer saved to {vocab_file} and {merges_file}")


def load_tokenizer(vocab_file, merges_file):
    """Load trained tokenizer from files"""

    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_json = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_json.items()}

    with open(merges_file, 'rb') as f:
        merges = pickle.load(f)

    return vocab, merges


if __name__ == "__main__":
    # Configuration
    CORPUS_FILE = "telugu_corpus.txt"
    VOCAB_SIZE = 5500
    VOCAB_FILE = "tokenizer_vocab.json"
    MERGES_FILE = "tokenizer_merges.pkl"

    print("=" * 60)
    print("BPE TOKENIZER TRAINING")
    print("=" * 60)

    # Load corpus
    print(f"\nLoading corpus from {CORPUS_FILE}...")
    try:
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {CORPUS_FILE} not found!")
        print("Please run scraper.py first to download the corpus.")
        sys.exit(1)

    # Train tokenizer
    print()
    tokenizer_data = train_bpe_tokenizer(text, vocab_size=VOCAB_SIZE, verbose=True)

    # Save tokenizer
    print()
    save_tokenizer(tokenizer_data, VOCAB_FILE, MERGES_FILE)

    # Display statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Vocabulary Size: {tokenizer_data['vocab_size']:,} tokens")
    print(f"Compression Ratio: {tokenizer_data['compression_ratio']:.2f}x")
    print(f"Original Length: {tokenizer_data['original_length']:,} bytes")
    print(f"Compressed Length: {tokenizer_data['compressed_length']:,} tokens")
    print("=" * 60)

    # Test tokenization
    print("\n" + "=" * 60)
    print("TESTING TOKENIZER")
    print("=" * 60)

    test_texts = [
        "తెలుగు",
        "నమస్కారం",
        "తెలుగు భాష చాలా అందంగా ఉంది",
    ]

    vocab = tokenizer_data['vocab']
    merges = tokenizer_data['merges']

    for test_text in test_texts:
        tokens = encode(test_text, merges)
        decoded = decode(tokens, vocab)
        print(f"\nOriginal: {test_text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Decoded: {decoded}")
        print(f"Match: {'✓' if decoded == test_text else '✗'}")

    print("\n" + "=" * 60)
