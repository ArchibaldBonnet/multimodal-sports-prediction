"""
Canonical match key — design decision (a).

    match_id = hash(date_iso + "_" + normalize_team(home) + "_" + normalize_team(away))

Computed ONCE, here, and propagated everywhere else in the codebase.
No positional joins. No ad hoc fuzzy matching scattered across files.
This module is the single source of truth for identifying a match
across every data source (raw matches, odds, embeddings, news).

Status: reserved in Phase 0 (skeleton). Implemented + tested in Phase 1
(ingestion + canonical key) against real, audited data — not before.
"""


def normalize_team(name: str) -> str:
    """Normalize a team name to a canonical form (lowercase, no suffixes/accents/punctuation)."""
    raise NotImplementedError("normalize_team: implemented in Phase 1 (ingestion + canonical key)")


def match_id(date_iso: str, home: str, away: str) -> str:
    """Compute the canonical match_id for one match. The only place this is ever computed."""
    raise NotImplementedError("match_id: implemented in Phase 1 (ingestion + canonical key)")
