"""
================================================================================
LLM Learning Module 1: TOKENIZER
================================================================================

What is a Tokenizer?
--------------------
A tokenizer converts raw text into a sequence of integers (tokens) that the
model can process. It's the first step in any NLP pipeline.

Tokenization Approaches:
1. Character-level: Each character is a token
2. Word-level: Each word is a token
3. Subword-level (BPE, WordPiece, Unigram): Balance between char and word

This module implements Byte Pair Encoding (BPE) - the most common tokenizer
for modern LLMs (GPT, LLaMA, etc.)

================================================================================
ILLUSTRATION: How BPE Tokenization Works
================================================================================

Text: "hello world"

Step 1: Split into characters
    ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 2: Count character pairs and merge most frequent
    Most frequent pair: ('l', 'l') -> 'll'
    ['h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 3: Continue merging...
    ('o', ' ') -> 'o '
    ('o', 'r') -> 'or'
    ...eventually building up to common words/subwords

Final vocabulary might include:
    ['h', 'e', 'll', 'o', ' ', 'w', 'or', 'd', 'hello', 'world', ...]

================================================================================
"""

import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
import json


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer

    BPE Algorithm:
    1. Start with character-level vocabulary
    2. Count all adjacent pairs in corpus
    3. Merge most frequent pair into new token
    4. Repeat until desired vocab size reached

    Attributes:
        vocab: Mapping from token string to token ID
        merges: List of merge operations (pair -> new_token)
        vocab_size: Maximum vocabulary size
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count frequency of adjacent pairs across all words.

        Example:
            word_freqs = {('h', 'e', 'l', 'l', 'o'): 5}
            Returns: Counter({('l', 'l'): 5, ('h', 'e'): 5, ('e', 'l'): 5, ('l', 'o'): 5})
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(
        self,
        word_freqs: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str]
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a pair in the corpus."""
        new_word_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus: List[str], verbose: bool = True):
        """
        Train the BPE tokenizer on a corpus of text.

        Args:
            corpus: List of text strings to train on
            verbose: Print progress during training

        The training process:
        1. Pre-tokenize text into words (using GPT-2 style regex)
        2. Split words into characters
        3. Iteratively merge most frequent pairs
        """
        # Step 1: Pre-tokenization (GPT-2 style)
        # This regex splits on contractions, words, numbers, and punctuation
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")

        word_freqs: Dict[Tuple[str, ...], int] = Counter()

        for text in corpus:
            tokens = pat.findall(text)
            for token in tokens:
                # Split into characters (bytes would be better for full implementation)
                word = tuple(token)
                word_freqs[word] += 1

        # Step 2: Initialize vocabulary with all characters
        vocab_set = set()
        for word in word_freqs.keys():
            vocab_set.update(word)

        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

        # Add character tokens
        for char in sorted(vocab_set):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char

        if verbose:
            print(f"Initial vocabulary size: {len(self.vocab)}")
            print(f"Target vocabulary size: {self.vocab_size}")

        # Step 3: BPE merge loop
        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                if verbose:
                    print(f"No more pairs to merge after {i} iterations")
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge the pair
            word_freqs = self._merge_pair(word_freqs, best_pair)
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            idx = len(self.vocab)
            self.vocab[new_token] = idx
            self.inverse_vocab[idx] = new_token
            self.merges.append(best_pair)

            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' (freq: {pairs[best_pair]})")

        if verbose:
            print(f"Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into token IDs.

        Args:
            text: Input string to tokenize

        Returns:
            List of token IDs
        """
        # Pre-tokenize
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
        tokens = pat.findall(text)

        token_ids = []
        for token in tokens:
            # Start with characters
            word = list(token)

            # Apply merges in order
            for merge in self.merges:
                i = 0
                while i < len(word) - 1:
                    if word[i] == merge[0] and word[i + 1] == merge[1]:
                        word = word[:i] + [merge[0] + merge[1]] + word[i + 2:]
                    else:
                        i += 1

            # Convert to IDs
            for w in word:
                if w in self.vocab:
                    token_ids.append(self.vocab[w])
                else:
                    token_ids.append(self.vocab["<unk>"])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to string.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded string
        """
        tokens = []
        for idx in token_ids:
            if idx in self.inverse_vocab:
                tokens.append(self.inverse_vocab[idx])
            else:
                tokens.append("<unk>")
        return "".join(tokens)

    def save(self, path: str):
        """Save tokenizer to JSON file."""
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load tokenizer from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """
    Demonstrate the BPE tokenizer with examples.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                         TOKENIZER DEMO                                    ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("=" * 80)
    print("BPE TOKENIZER DEMONSTRATION")
    print("=" * 80)

    # Training corpus
    corpus = [
        "hello world",
        "hello there",
        "world of tokens",
        "tokenization is important",
        "the transformer architecture uses attention",
        "attention is all you need",
        "neural networks learn representations",
        "deep learning is powerful",
        "machine learning models",
        "natural language processing",
    ] * 10  # Repeat to simulate larger corpus

    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus, verbose=True)

    print("\n" + "-" * 80)
    print("ENCODING EXAMPLES")
    print("-" * 80)

    test_texts = [
        "hello world",
        "tokenization is important",
        "deep learning"
    ]

    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\nOriginal:  '{text}'")
        print(f"Encoded:   {ids}")
        print(f"Decoded:   '{decoded}'")

    print("\n" + "-" * 80)
    print("SAMPLE VOCABULARY")
    print("-" * 80)

    # Show some merged tokens
    print("\nSpecial tokens:", tokenizer.special_tokens)
    print("\nMerged tokens (last 20):")
    for merge in tokenizer.merges[-20:]:
        merged = merge[0] + merge[1]
        print(f"  '{merge[0]}' + '{merge[1]}' -> '{merged}' (id: {tokenizer.vocab[merged]})")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. BPE starts with characters and iteratively merges frequent pairs
    2. Common words become single tokens, rare words are split into subwords
    3. This balances vocabulary size with sequence length
    4. Modern tokenizers (GPT-4, LLaMA) use byte-level BPE for better handling
       of unknown characters and multilingual text

    Next: 02_embeddings.py - How tokens become dense vectors
    """)


if __name__ == "__main__":
    demo()
