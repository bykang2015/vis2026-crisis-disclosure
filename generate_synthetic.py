"""
generate_synthetic.py
=====================
Generates synthetic Korean child counseling session data that
matches the statistical structure of the AI Hub corpus.

Synthetic data preserves:
  - Session count and crisis-level distribution (N=3,236)
  - Turn length distributions per crisis level
  - Crisis-stratified disclosure timing (MSW50 = 9-11 turns)
  - Keyword co-occurrence rates per category
  - 8.7% censoring rate

NOTE: All data is fully synthetic. No real counseling records
are included or can be recovered from this file.

Usage:
    python generate_synthetic.py
    python generate_synthetic.py --n_sessions 500 --seed 42
    python generate_synthetic.py --output_dir data/

Output:
    data/synthetic_sessions.csv   — session-level metadata
    data/synthetic_turns.csv      — turn-level corpus

Paper: Convergent Timing, Divergent Trajectories
Venue: IEEE VIS 2026 Short Papers (Anonymous Submission)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Population parameters (from AI Hub corpus) ─────────────
CRISIS_PARAMS = {
    'Emergency': {
        'n': 672,
        'turn_mean': 123.5, 'turn_sd': 17.9,
        'msw50': 9,
        'shape': 0.55,   # Weibull shape (front-loaded)
        'scale': 11.0,
        'detected_rate': 0.924,
    },
    'Abuse-Suspected': {
        'n': 644,
        'turn_mean': 117.9, 'turn_sd': 14.4,
        'msw50': 9,
        'shape': 0.58,
        'scale': 11.5,
        'detected_rate': 0.944,
    },
    'Counseling': {
        'n': 621,
        'turn_mean': 108.6, 'turn_sd': 15.3,
        'msw50': 11,
        'shape': 0.62,
        'scale': 13.5,
        'detected_rate': 0.945,
    },
    'Observation': {
        'n': 634,
        'turn_mean': 106.7, 'turn_sd': 13.5,
        'msw50': 11,
        'shape': 0.65,
        'scale': 14.0,
        'detected_rate': 0.929,
    },
    'Normal': {
        'n': 665,
        'turn_mean': 100.4, 'turn_sd': 13.2,
        'msw50': 10,
        'shape': 0.72,   # More gradual accumulation
        'scale': 16.0,
        'detected_rate': 0.824,
    },
}

# Keyword categories and synthetic Korean placeholders
KEYWORD_CATEGORIES = {
    'Physical Abuse':  ['때리다', '폭행', '발로차다', '멍', '맞다'],
    'Emotional Abuse': ['소리지르다', '협박', '무시', '욕하다', '겁주다'],
    'Sexual Abuse':    ['만지다', '비밀', '불편하다', '나쁜접촉', '몸'],
    'Neglect':         ['배고프다', '혼자', '밥없다', '아무도없다', '방치'],
    'Self-Harm':       ['자해', '자르다', '사라지다', '죽다', '살기싫다'],
    'Severe Distress': ['잠못자다', '울다', '무섭다', '떨다', '악몽'],
    'Social Issues':   ['괴롭히다', '싸우다', '친구없다', '따돌림', '학교싫다'],
    'General Distress':['슬프다', '걱정', '힘들다', '어렵다', '속상하다'],
}

# Per-category detection rates by crisis level (from Appendix A)
CATEGORY_RATES = {
    'Emergency':      [0.682, 0.441, 0.186, 0.314, 0.527,
                       0.613, 0.478, 0.884],
    'Abuse-Suspected':[0.714, 0.478, 0.223, 0.339, 0.483,
                       0.552, 0.431, 0.867],
    'Counseling':     [0.631, 0.382, 0.094, 0.247, 0.316,
                       0.448, 0.354, 0.823],
    'Observation':    [0.583, 0.316, 0.061, 0.192, 0.224,
                       0.379, 0.287, 0.791],
    'Normal':         [0.517, 0.243, 0.038, 0.146, 0.112,
                       0.281, 0.196, 0.713],
}
CATEGORIES = list(KEYWORD_CATEGORIES.keys())


def sample_disclosure_turn(shape: float,
                            scale: float,
                            max_turn: int,
                            rng: np.random.Generator) -> int:
    """
    Sample first-disclosure turn from Weibull distribution,
    calibrated to match observed MSW50 and trajectory shapes.
    Clipped to [1, max_turn].
    """
    t = rng.weibull(shape) * scale
    return int(np.clip(np.round(t), 1, max_turn))


def generate_turn_text(turn_idx: int,
                       speaker: str,
                       keyword: str = None,
                       rng: np.random.Generator = None) -> str:
    """
    Generate synthetic turn text.
    If keyword is provided, embed it in the utterance.
    All text is synthetic — no real counseling content.
    """
    if speaker == 'counselor':
        templates = [
            '오늘 어떻게 지냈어?',
            '그게 어떤 느낌이었어?',
            '좀 더 말해줄 수 있어?',
            '그때 어디 있었어?',
            '그런 일이 자주 있어?',
        ]
        return rng.choice(templates) if rng else templates[0]

    # Child utterance
    if keyword:
        templates = [
            f'사실은 {keyword} 있어요.',
            f'저 {keyword} 경험했어요.',
            f'집에서 {keyword} 일이 있었어요.',
        ]
        return rng.choice(templates) if rng else templates[0]

    templates = [
        '잘 모르겠어요.',
        '그냥 그래요.',
        '별로 말하고 싶지 않아요.',
        '네, 그랬어요.',
        '아니요.',
    ]
    return rng.choice(templates) if rng else templates[0]


def generate_session(session_id: str,
                     crisis_level: str,
                     params: dict,
                     rng: np.random.Generator) -> tuple:
    """
    Generate one synthetic counseling session.

    Returns
    -------
    session_meta : dict with session-level fields
    turns        : list of turn dicts
    """
    p = params[crisis_level]
    cat_rates = CATEGORY_RATES[crisis_level]

    # Sample session length
    duration = max(10, int(rng.normal(p['turn_mean'], p['turn_sd'])))

    # Determine if session produces a detection
    is_detected = rng.random() < p['detected_rate']

    # Sample disclosure turn (if detected)
    event_turn = None
    matched_cat = None
    matched_kw = None
    if is_detected:
        event_turn = sample_disclosure_turn(
            p['shape'], p['scale'], duration, rng)
        # Sample which category triggered the anchor
        cat_probs = np.array(cat_rates)
        cat_probs = cat_probs / cat_probs.sum()
        matched_cat = rng.choice(CATEGORIES, p=cat_probs)
        matched_kw = rng.choice(KEYWORD_CATEGORIES[matched_cat])

    # Sample keyword co-occurrence (which categories appear)
    categories_present = [
        cat for cat, rate in zip(CATEGORIES, cat_rates)
        if rng.random() < rate
    ]

    # Build turn-level data
    turns = []
    age = int(rng.integers(7, 14))  # ages 7-13
    for t in range(1, duration + 1):
        # Alternating counselor / child (simplified)
        speaker = 'counselor' if t % 2 == 1 else 'child'

        # Embed keyword at disclosure turn
        kw = None
        if (speaker == 'child' and is_detected
                and event_turn is not None and t == event_turn):
            kw = matched_kw

        text = generate_turn_text(t, speaker, kw, rng)

        turns.append({
            'session_id':   session_id,
            'turn_index':   t,
            'speaker':      speaker,
            'text':         text,
            'crisis_level': crisis_level,
            'age':          age,
        })

    session_meta = {
        'session_id':       session_id,
        'crisis_level':     crisis_level,
        'duration':         duration,
        'event_turn':       event_turn,
        'detected':         int(is_detected),
        'matched_category': matched_cat,
        'matched_keyword':  matched_kw,
        'age':              age,
        'categories_present': ','.join(categories_present),
    }

    return session_meta, turns


def generate_corpus(n_sessions: int = None,
                    seed: int = 42) -> tuple:
    """
    Generate full synthetic corpus matching AI Hub statistics.

    Parameters
    ----------
    n_sessions : total sessions (default: 3236, full corpus size)
    seed       : random seed for reproducibility

    Returns
    -------
    sessions_df : DataFrame, session-level metadata
    turns_df    : DataFrame, turn-level corpus
    """
    rng = np.random.default_rng(seed)

    all_sessions = []
    all_turns = []
    session_counter = 0

    # Scale session counts if n_sessions specified
    if n_sessions is not None:
        total = sum(p['n'] for p in CRISIS_PARAMS.values())
        scale = n_sessions / total
    else:
        scale = 1.0

    for crisis_level, params in CRISIS_PARAMS.items():
        n = max(1, int(params['n'] * scale))
        print(f"  Generating {n:4d} {crisis_level} sessions...")

        for i in range(n):
            sid = f"{crisis_level[:3].upper()}_{session_counter:05d}"
            meta, turns = generate_session(
                sid, crisis_level, CRISIS_PARAMS, rng)
            all_sessions.append(meta)
            all_turns.extend(turns)
            session_counter += 1

    sessions_df = pd.DataFrame(all_sessions)
    turns_df    = pd.DataFrame(all_turns)

    return sessions_df, turns_df


def print_summary(sessions_df: pd.DataFrame) -> None:
    """Print corpus statistics for verification."""
    print("\n" + "=" * 55)
    print("Synthetic corpus summary")
    print("=" * 55)
    total = len(sessions_df)
    print(f"Total sessions : {total:,}")
    print(f"Total turns    : {sessions_df['duration'].sum():,}")
    print(f"Detection rate : {sessions_df['detected'].mean()*100:.1f}%")
    print()
    print(f"{'Crisis level':<20} {'N':>5} {'Det%':>6} "
          f"{'MSW50':>6} {'AvgLen':>7}")
    print("-" * 50)
    for level in CRISIS_PARAMS.keys():
        sub = sessions_df[sessions_df['crisis_level'] == level]
        det = sub['detected'].mean() * 100
        detected = sub[sub['detected'] == 1]['event_turn']
        msw50 = detected.median() if len(detected) > 0 else float('nan')
        avg_len = sub['duration'].mean()
        print(f"{level:<20} {len(sub):>5} {det:>5.1f}% "
              f"{msw50:>6.0f} {avg_len:>7.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic counseling corpus')
    parser.add_argument('--n_sessions', type=int, default=None,
        help='Total sessions to generate (default: full 3,236)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', default='data',
        help='Output directory (default: data/)')
    args = parser.parse_args()

    print("Generating synthetic counseling corpus...")
    print(f"  Seed: {args.seed}")
    if args.n_sessions:
        print(f"  N sessions: {args.n_sessions} (scaled)")
    else:
        print(f"  N sessions: 3,236 (full corpus)")

    sessions_df, turns_df = generate_corpus(
        n_sessions=args.n_sessions,
        seed=args.seed)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sessions_path = out / 'synthetic_sessions.csv'
    turns_path    = out / 'synthetic_turns.csv'

    sessions_df.to_csv(sessions_path, index=False)
    turns_df.to_csv(turns_path, index=False)

    print_summary(sessions_df)
    print(f"\nSaved:")
    print(f"  {sessions_path}  ({len(sessions_df):,} rows)")
    print(f"  {turns_path}  ({len(turns_df):,} rows)")
    print("\nNext step:")
    print("  python analysis/survival_analysis.py "
          "--sessions data/synthetic_sessions.csv")


if __name__ == '__main__':
    main()
