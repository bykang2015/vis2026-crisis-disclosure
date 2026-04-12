"""
keyword_anchoring.py
====================
Temporal anchoring pipeline for first risk-keyword disclosure
in Korean child counseling sessions.

Two paradigms:
  1. Exact substring matching (primary, transparent, auditable)
  2. Semantic similarity anchoring (robustness validation)

Paper: Convergent Timing, Divergent Trajectories:
       Visualizing the Temporal Paradox of Child Crisis Disclosure
Venue: IEEE VIS 2026 Short Papers (Anonymous Submission)

Dataset: AI Hub Korean Child Counseling Corpus
         N=3,236 sessions, 360,816 turns, ages 7-13, 2021-2023

Requirements:
    pip install pandas numpy sentence-transformers scikit-learn
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ── Negation patterns (Korean) ──────────────────────────────
# These patterns precede a keyword to indicate negation/denial.
# Matches are excluded from temporal anchoring.
NEGATION_PATTERNS = [
    r'아니',      # no / not
    r'없',        # absence
    r'안\s*',     # negation prefix
    r'못\s*',     # inability prefix
    r'절대',      # never/absolutely not
    r'그런\s*적\s*없',  # never happened
]

# Compiled negation regex (look-behind window: 10 chars)
NEGATION_RE = re.compile(
    '(' + '|'.join(NEGATION_PATTERNS) + ')',
    re.UNICODE
)

# ── Referential / hypothetical patterns ────────────────────
# Exclude utterances that discuss keywords abstractly,
# not as first-person disclosures.
REFERENTIAL_PATTERNS = [
    r'만약',      # if / hypothetically
    r'혹시',      # perhaps / just in case
    r'다른\s*애',  # other child
    r'친구가',    # friend (third-person)
    r'선생님이',   # teacher (third-person)
]

REFERENTIAL_RE = re.compile(
    '(' + '|'.join(REFERENTIAL_PATTERNS) + ')',
    re.UNICODE
)


# ── Primary: Exact Substring Matching ──────────────────────
def load_keyword_taxonomy(path: str = 'data/keyword_taxonomy.csv'
                          ) -> dict:
    """
    Load keyword taxonomy CSV.
    Returns dict: category -> list of Korean keywords.
    """
    df = pd.read_csv(path)
    taxonomy = {}
    for _, row in df.iterrows():
        cat = row['category']
        kw  = row['keyword_korean']
        taxonomy.setdefault(cat, []).append(kw)
    return taxonomy


def is_negated(text: str, match_start: int, window: int = 10) -> bool:
    """
    Check whether a keyword match at match_start is preceded
    by a negation pattern within a look-behind window.
    """
    start = max(0, match_start - window)
    prefix = text[start:match_start]
    return bool(NEGATION_RE.search(prefix))


def is_referential(text: str) -> bool:
    """
    Check whether the utterance is referential/hypothetical
    rather than a first-person disclosure.
    """
    return bool(REFERENTIAL_RE.search(text))


def detect_first_disclosure_exact(
        session_turns: list[dict],
        keywords: list[str],
        speaker_col: str = 'speaker',
        text_col: str = 'text',
        turn_col: str = 'turn_index',
        child_speaker: str = 'child'
) -> Optional[int]:
    """
    Find the first child turn containing any keyword,
    excluding negated and referential uses.

    Parameters
    ----------
    session_turns : list of dicts with keys [turn_index, speaker, text]
    keywords      : flat list of Korean keyword strings
    speaker_col   : column name for speaker identifier
    text_col      : column name for utterance text
    turn_col      : column name for turn index (1-based)
    child_speaker : string identifier for child speaker

    Returns
    -------
    int  : 1-based turn index of first disclosure, or None if not found
    """
    # Filter to child turns only
    child_turns = [t for t in session_turns
                   if t[speaker_col] == child_speaker]

    for turn in sorted(child_turns, key=lambda x: x[turn_col]):
        text = turn[text_col]

        # Skip referential utterances
        if is_referential(text):
            continue

        for kw in keywords:
            match = re.search(re.escape(kw), text, re.UNICODE)
            if match and not is_negated(text, match.start()):
                return int(turn[turn_col])

    return None  # Right-censored


def anchor_corpus_exact(
        sessions: dict,
        taxonomy: dict,
        **kwargs
) -> pd.DataFrame:
    """
    Apply exact keyword anchoring to all sessions.

    Parameters
    ----------
    sessions : dict mapping session_id -> list of turn dicts
    taxonomy : dict mapping category -> list of keywords

    Returns
    -------
    DataFrame with columns:
        session_id, event_turn, detected, duration,
        matched_category, matched_keyword
    """
    all_keywords = [kw for kws in taxonomy.values() for kw in kws]
    records = []

    for sid, turns in sessions.items():
        duration = max(t['turn_index'] for t in turns)
        event_turn = detect_first_disclosure_exact(
            turns, all_keywords, **kwargs)

        # Also record which category/keyword triggered the anchor
        matched_cat = None
        matched_kw  = None
        if event_turn is not None:
            child_turns = [t for t in turns
                           if t.get('speaker') == 'child']
            trigger_turn = next(
                t for t in child_turns
                if t['turn_index'] == event_turn)
            text = trigger_turn['text']
            for cat, kws in taxonomy.items():
                for kw in kws:
                    m = re.search(re.escape(kw), text, re.UNICODE)
                    if m and not is_negated(text, m.start()):
                        matched_cat = cat
                        matched_kw  = kw
                        break
                if matched_cat:
                    break

        records.append({
            'session_id':      sid,
            'event_turn':      event_turn,
            'detected':        int(event_turn is not None),
            'duration':        duration,
            'matched_category': matched_cat,
            'matched_keyword':  matched_kw
        })

    df = pd.DataFrame(records)
    n_detected = df['detected'].sum()
    pct = n_detected / len(df) * 100
    print(f"Exact anchoring: {n_detected}/{len(df)} sessions "
          f"detected ({pct:.1f}%)")
    return df


# ── Robustness: Semantic Similarity Anchoring ───────────────
def load_sentence_model(model_name: str =
                         'paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Load multilingual sentence transformer.
    Default model supports Korean + 50 languages.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        print(f"Loaded sentence model: {model_name}")
        return model
    except ImportError:
        raise ImportError(
            "Install sentence-transformers: "
            "pip install sentence-transformers")


def compute_keyword_embeddings(keywords: list[str],
                                model) -> np.ndarray:
    """Encode all keywords into embedding space."""
    return model.encode(keywords, convert_to_numpy=True,
                         show_progress_bar=False)


def detect_first_disclosure_semantic(
        session_turns: list[dict],
        keyword_embeddings: np.ndarray,
        model,
        threshold: float = 0.50,
        speaker_col: str = 'speaker',
        text_col: str = 'text',
        turn_col: str = 'turn_index',
        child_speaker: str = 'child'
) -> Optional[int]:
    """
    Find the first child turn whose embedding exceeds cosine
    similarity threshold with any keyword embedding.

    Parameters
    ----------
    threshold : float, cosine similarity cutoff (default 0.50)

    Returns
    -------
    int : 1-based turn index, or None if not found
    """
    from sklearn.metrics.pairwise import cosine_similarity

    child_turns = [t for t in session_turns
                   if t[speaker_col] == child_speaker]

    for turn in sorted(child_turns, key=lambda x: x[turn_col]):
        text = turn[text_col]
        if not text.strip():
            continue

        turn_emb = model.encode([text], convert_to_numpy=True)
        sims = cosine_similarity(turn_emb,
                                  keyword_embeddings)[0]

        if sims.max() >= threshold:
            return int(turn[turn_col])

    return None


def compare_anchoring_paradigms(
        exact_df: pd.DataFrame,
        sessions: dict,
        taxonomy: dict,
        threshold: float = 0.50
) -> pd.DataFrame:
    """
    Compare exact vs semantic anchoring paradigms.

    Reports:
      - Median timing convergence (|delta| < 1 turn)
      - Additional coverage from semantic anchoring
      - Agreement rate on detected sessions

    Returns DataFrame with per-session comparison.
    """
    print("\nLoading sentence transformer for semantic anchoring...")
    model = load_sentence_model()
    all_keywords = [kw for kws in taxonomy.values() for kw in kws]
    kw_embs = compute_keyword_embeddings(all_keywords, model)

    records = []
    for sid, turns in sessions.items():
        sem_turn = detect_first_disclosure_semantic(
            turns, kw_embs, model, threshold=threshold)
        records.append({
            'session_id': sid,
            'semantic_event_turn': sem_turn,
            'semantic_detected': int(sem_turn is not None)
        })

    sem_df = pd.DataFrame(records)
    merged = exact_df.merge(sem_df, on='session_id')

    # Sessions detected by both
    both = merged[
        (merged['detected'] == 1) &
        (merged['semantic_detected'] == 1)].copy()
    both['delta'] = (both['semantic_event_turn']
                     - both['event_turn']).abs()

    # Additional coverage
    extra = merged[
        (merged['detected'] == 0) &
        (merged['semantic_detected'] == 1)]

    print(f"\nAnchoring comparison (threshold={threshold}):")
    print(f"  Exact detected:    {merged['detected'].sum():,}")
    print(f"  Semantic detected: {merged['semantic_detected'].sum():,}")
    print(f"  Additional coverage: {len(extra):,} sessions "
          f"({len(extra)/len(merged)*100:.1f}%)")
    print(f"  Median |Δ turn| (both detected): "
          f"{both['delta'].median():.1f} turns")
    print(f"  Agreement (Δ < 2 turns): "
          f"{(both['delta'] < 2).mean()*100:.1f}%")

    return merged


# ── Category-Level Analysis ─────────────────────────────────
def compute_category_rates(
        anchored_df: pd.DataFrame,
        crisis_col: str = 'crisis_level'
) -> pd.DataFrame:
    """
    Compute keyword category detection rates per crisis level.
    Requires 'matched_category' column from anchor_corpus_exact().
    """
    detected = anchored_df[anchored_df['detected'] == 1].copy()
    table = pd.crosstab(
        detected[crisis_col],
        detected['matched_category'],
        normalize='index'
    ) * 100
    return table.round(1)


# ── Main Pipeline ───────────────────────────────────────────
def main(sessions_path: str = 'data/sessions.csv',
         taxonomy_path: str = 'data/keyword_taxonomy.csv',
         run_semantic:  bool = False):
    """
    Full anchoring pipeline.

    Parameters
    ----------
    sessions_path : path to session-turn CSV
    taxonomy_path : path to keyword taxonomy CSV
    run_semantic  : whether to run semantic robustness check
                    (requires sentence-transformers, slower)
    """
    print("=" * 60)
    print("Keyword Temporal Anchoring Pipeline")
    print("=" * 60)

    # Load taxonomy
    taxonomy = load_keyword_taxonomy(taxonomy_path)
    n_kw = sum(len(v) for v in taxonomy.values())
    print(f"Loaded {n_kw} keywords across "
          f"{len(taxonomy)} categories")

    # Load session-turn data
    # Expected format: session_id, turn_index, speaker, text
    turns_df = pd.read_csv(sessions_path)
    sessions = {
        sid: grp.to_dict('records')
        for sid, grp in turns_df.groupby('session_id')
    }
    print(f"Loaded {len(sessions):,} sessions")

    # Exact anchoring
    anchored_df = anchor_corpus_exact(sessions, taxonomy)
    anchored_df.to_csv('results/anchored_sessions.csv', index=False)

    # Category breakdown
    if 'crisis_level' in turns_df.columns:
        crisis_map = (turns_df[['session_id', 'crisis_level']]
                      .drop_duplicates()
                      .set_index('session_id')['crisis_level'])
        anchored_df['crisis_level'] = (anchored_df['session_id']
                                        .map(crisis_map))
        cat_rates = compute_category_rates(anchored_df)
        cat_rates.to_csv('results/category_rates.csv')
        print("\n--- Category Detection Rates (%) ---")
        print(cat_rates.to_string())

    # Semantic robustness (optional)
    if run_semantic:
        compare_anchoring_paradigms(
            anchored_df, sessions, taxonomy, threshold=0.50)

    print("\nDone. Results saved to results/")
    return anchored_df


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Keyword temporal anchoring pipeline')
    parser.add_argument('--sessions',  default='data/sessions.csv')
    parser.add_argument('--taxonomy',  default='data/keyword_taxonomy.csv')
    parser.add_argument('--semantic',  action='store_true',
                        help='Run semantic robustness check')
    args = parser.parse_args()

    main(sessions_path=args.sessions,
         taxonomy_path=args.taxonomy,
         run_semantic=args.semantic)
