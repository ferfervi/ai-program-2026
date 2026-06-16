"""Retrieval-augmented generation.

- ``chunking/`` — the ``Chunker`` interface, the structural chunker and the
  comparison strategies (Session 7).
- ``embedding/`` — the OpenAI embedder.
- ``analysis/`` — similarity + strategy-comparison tooling.
- ``store/`` — vector persistence (pgvector). Reserved for Session 8.
- ``retriever.py`` — semantic retrieval over the store. Reserved for Session 8.

Today vectors are produced in memory and returned over HTTP; persistence and
retrieval land in Session 8.
"""
