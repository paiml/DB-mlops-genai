# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1.5: Build a BPE Tokenizer
# MAGIC
# MAGIC **Course 4, Week 1: LLM Serving**
# MAGIC
# MAGIC ## Objectives
# MAGIC - Understand tokenization concepts
# MAGIC - Implement basic tokenizer functionality
# MAGIC - Handle special tokens

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from typing import List, Dict
from dataclasses import dataclass

print("Tokenizer Lab - Course 4 Week 1")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Vocabulary Building
# MAGIC
# MAGIC First, let's build a simple vocabulary from text.

# COMMAND ----------

def build_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from list of texts."""
    word_counts = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1

    # Filter by frequency and assign IDs
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    next_id = 4

    for word, count in sorted(word_counts.items()):
        if count >= min_freq:
            vocab[word] = next_id
            next_id += 1

    return vocab


# Test vocabulary building
sample_texts = [
    "machine learning is great",
    "deep learning uses neural networks",
    "machine learning and deep learning",
    "neural networks are powerful",
]

vocab = build_vocab(sample_texts, min_freq=2)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample entries: {dict(list(vocab.items())[:10])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Implement Tokenizer Class
# MAGIC
# MAGIC TODO: Implement a Tokenizer class with encode/decode methods.

# COMMAND ----------

class Tokenizer:
    """Simple word-level tokenizer."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        # TODO: Create reverse vocabulary (id -> word)
        self.reverse_vocab = {}  # YOUR CODE HERE

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Convert text to token IDs.

        TODO: Implement encoding:
        1. Split text into words
        2. Look up each word in vocab (use <unk> for unknown)
        3. Optionally add <bos> at start and <eos> at end
        """
        # YOUR CODE HERE
        pass

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to text.

        TODO: Implement decoding:
        1. Look up each ID in reverse_vocab
        2. Optionally skip special tokens
        3. Join words with spaces
        """
        # YOUR CODE HERE
        pass

    def vocab_size(self) -> int:
        return len(self.vocab)


# Test your tokenizer
# tokenizer = Tokenizer(vocab)
# encoded = tokenizer.encode("machine learning is powerful")
# print(f"Encoded: {encoded}")
# decoded = tokenizer.decode(encoded)
# print(f"Decoded: {decoded}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Token Counting
# MAGIC
# MAGIC TODO: Implement token counting for cost estimation.

# COMMAND ----------

def count_tokens(tokenizer: Tokenizer, text: str) -> Dict[str, int]:
    """
    Count tokens in text.

    TODO: Return dict with:
    - total_tokens: total number of tokens
    - unique_tokens: number of unique tokens
    - unknown_tokens: number of <unk> tokens
    """
    # YOUR CODE HERE
    pass


# Test token counting
# counts = count_tokens(tokenizer, "machine learning is the future of AI")
# print(f"Token counts: {counts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Batch Encoding
# MAGIC
# MAGIC TODO: Implement batch encoding with padding.

# COMMAND ----------

def encode_batch(tokenizer: Tokenizer, texts: List[str], max_length: int = 20) -> List[List[int]]:
    """
    Encode multiple texts with padding to same length.

    TODO:
    1. Encode each text
    2. Truncate if longer than max_length
    3. Pad with <pad> token if shorter
    """
    # YOUR CODE HERE
    pass


# Test batch encoding
# batch = encode_batch(tokenizer, [
#     "machine learning",
#     "deep learning uses neural networks",
#     "AI"
# ], max_length=10)
# for i, encoded in enumerate(batch):
#     print(f"Text {i}: {encoded}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Token Analysis
# MAGIC
# MAGIC TODO: Analyze tokenization of a corpus.

# COMMAND ----------

def analyze_corpus(tokenizer: Tokenizer, texts: List[str]) -> Dict:
    """
    Analyze tokenization statistics for a corpus.

    TODO: Return dict with:
    - total_texts: number of texts
    - total_tokens: total tokens across all texts
    - avg_tokens_per_text: average tokens per text
    - unknown_rate: percentage of unknown tokens
    """
    # YOUR CODE HERE
    pass


# Test corpus analysis
# analysis = analyze_corpus(tokenizer, sample_texts)
# print(f"Corpus analysis: {analysis}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

def validate_lab():
    """Validate lab completion."""
    checks = []

    try:
        tokenizer = Tokenizer(vocab)

        # Check encode
        encoded = tokenizer.encode("machine learning")
        checks.append(("Encode works", encoded is not None and len(encoded) > 0))

        # Check decode
        decoded = tokenizer.decode(encoded)
        checks.append(("Decode works", decoded is not None and len(decoded) > 0))

        # Check special tokens
        has_special = encoded[0] == vocab["<bos>"] and encoded[-1] == vocab["<eos>"]
        checks.append(("Special tokens added", has_special))

        # Check batch encoding
        if 'encode_batch' in dir():
            batch = encode_batch(tokenizer, ["test", "test two"], max_length=5)
            all_same_len = all(len(b) == 5 for b in batch)
            checks.append(("Batch padding works", all_same_len))

    except Exception as e:
        checks.append(("Implementation complete", False))
        print(f"Error: {e}")

    # Display results
    print("Lab Validation Results:")
    print("-" * 40)
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All checks passed! Lab complete.")
    else:
        print("\n⚠️ Some checks failed. Review your code.")

validate_lab()
