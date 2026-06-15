# Embedding Pipeline — Sanity Check

End-to-end smoke test of the embedding pipeline using
`scripts/compare.py` (cosine similarity over `text-embedding-3-small`,
1536 dimensions).

## Pair A — Semantically close (expected > 0.6)

- **Text 1:** `OAuth 2.0 authentication backend with JWT tokens for fintech mobile app`
- **Text 2:** `Authorization service using JSON Web Tokens for a banking application`
- **Cosine similarity:** `0.5958`

## Pair B — Unrelated (expected < 0.4)

- **Text 1:** `OAuth 2.0 authentication backend with JWT tokens for fintech mobile app`
- **Text 2:** `Database migration from MySQL to PostgreSQL with zero downtime`
- **Cosine similarity:** `0.1920`

## Pair C — Generic, ambiguous (no fixed expectation)

- **Text 1:** `Backend services`
- **Text 2:** `API development`
- **Cosine similarity:** `0.5407`

## Comments

The ordering matches intuition: A (paraphrase of the same concept) >
C (generic, partially overlapping) >> B (unrelated topics). Pair B sits
comfortably below the 0.4 threshold, so the pipeline clearly separates
distant concepts. What stands out is Pair A landing just under 0.6
despite being almost a direct paraphrase — "OAuth/JWT/fintech" vs
"Authorization/JSON Web Tokens/banking" share meaning but barely any
surface vocabulary, and `text-embedding-3-small` seems to penalise that
lexical distance more than expected. Pair C at 0.54 is also worth
flagging: two very short, vague strings end up nearly as close as the
paraphrase pair, which is a useful reminder that low-information inputs
inflate similarity scores. Overall the pipeline is healthy enough for
the exercise — discriminates near vs. far — but the A/C gap is small
and worth discussing live.
