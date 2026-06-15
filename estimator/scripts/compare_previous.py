"""Compare two strings by cosine similarity over their OpenAI embeddings.

Usage:
    uv run python scripts/compare.py \\
        --text-a "OAuth 2.0 authentication backend for fintech" \\
        --text-b "JWT-based authorization service for banking app"

Reuses the project's OpenAIEmbedder; cosine similarity is computed by
hand (dot / (||a|| * ||b||)) without numpy.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Make the project root importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embedding_pipeline.embedder import OpenAIEmbedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"vector length mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compute cosine similarity with a zero-norm vector")
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two strings by cosine similarity over OpenAI embeddings.",
    )
    parser.add_argument("--text-a", required=True, help="First text to embed.")
    parser.add_argument("--text-b", required=True, help="Second text to embed.")
    args = parser.parse_args()

    embedder = OpenAIEmbedder()
    vec_a = embedder.embed_one(args.text_a)
    vec_b = embedder.embed_one(args.text_b)
    similarity = cosine_similarity(vec_a, vec_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
