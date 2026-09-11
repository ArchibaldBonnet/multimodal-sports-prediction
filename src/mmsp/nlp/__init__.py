"""
mmsp.nlp — text embedding pipelines (design decision b).

Two distinct, non-interchangeable embedding types:

- Static team embeddings: historical articles filtered per team,
  encoded with xlm-roberta-base (CLS token), mean-pooled per team.
  One vector per team, indexed by normalize_team(name). Represents
  team "profile/style". Stored as team_embeddings.pt.

- Dynamic news embeddings: one vector per article, tagged with date
  and mentioned teams (via match_id-based team lookup, not free-text
  fuzzy matching), consumed via a sliding window before each match.
  Short-term signal (injuries, morale). Stored as news_embeddings.pt.

These must never be fed into a module expecting the other shape —
that mismatch was the root cause of the original rebuild.
"""
