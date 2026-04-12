"""
survival_analysis.py
====================
Kaplan-Meier survival analysis and GLM discrete-time hazard models
for crisis-stratified temporal disclosure trajectories.

Paper: Convergent Timing, Divergent Trajectories:
       Visualizing the Temporal Paradox of Child Crisis Disclosure
Venue: IEEE VIS 2026 Short Papers (Anonymous Submission)

Dataset: AI Hub Korean Child Counseling Corpus
         N=3,236 sessions, 360,816 turns, ages 7-13, 2021-2023
         https://aihub.or.kr

Requirements:
    pip install lifelines statsmodels pandas numpy scipy matplotlib
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
CRISIS_LEVELS = ['Emergency', 'Abuse-Suspected', 'Counseling',
                 'Observation', 'Normal']
CRISIS_ORDER  = {c: i for i, c in enumerate(CRISIS_LEVELS)}
COLORS        = ['#B04848', '#C87060', '#5A8EA8', '#3A7090', '#1E4A6E']
CHECKPOINTS   = [5, 10, 20, 30, 50, 80, 100]

# ── Data Loading ────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Load and validate the counseling corpus.

    Expected columns:
        session_id    : unique session identifier
        crisis_level  : one of CRISIS_LEVELS
        duration      : total number of turns in session
        event_turn    : first disclosure turn (NaN if not detected)
        detected      : 1 if keyword detected, 0 if right-censored

    Returns preprocessed DataFrame with survival variables.
    """
    df = pd.read_csv(path)

    required = ['session_id', 'crisis_level', 'duration',
                'event_turn', 'detected']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Right-censor non-detected sessions at final turn
    df['T'] = df['event_turn'].fillna(df['duration'])
    df['E'] = df['detected'].astype(int)

    # Validate crisis levels
    invalid = set(df['crisis_level'].unique()) - set(CRISIS_LEVELS)
    if invalid:
        raise ValueError(f"Unknown crisis levels: {invalid}")

    print(f"Loaded {len(df):,} sessions "
          f"({df['E'].mean()*100:.1f}% detected)")
    return df


# ── Kaplan-Meier Estimation ─────────────────────────────────
def fit_km_by_crisis(df: pd.DataFrame) -> dict:
    """
    Fit Kaplan-Meier estimators for each crisis level.

    Returns dict mapping crisis_level -> KaplanMeierFitter.
    Note: KM estimates S(t) = P(T > t); we plot 1-S(t) = cumulative
    detection probability.
    """
    kmf_dict = {}
    for level in CRISIS_LEVELS:
        sub = df[df['crisis_level'] == level]
        kmf = KaplanMeierFitter(label=level)
        kmf.fit(sub['T'], event_observed=sub['E'],
                timeline=range(0, 121))
        kmf_dict[level] = kmf
        msw50 = kmf.median_survival_time_
        print(f"{level:20s} N={len(sub):4d}  "
              f"MSW50={msw50:.0f}t  "
              f"T50={1 - kmf.survival_function_at_times([50]).values[0]:.3f}")
    return kmf_dict


def compute_detection_table(kmf_dict: dict,
                             checkpoints: list = CHECKPOINTS) -> pd.DataFrame:
    """
    Compute cumulative detection rates at specified checkpoints.
    Returns DataFrame (crisis_level x checkpoint).
    """
    rows = []
    for level, kmf in kmf_dict.items():
        row = {'crisis_level': level,
               'N': int(kmf.event_table['observed'].sum()
                        + kmf.event_table['censored'].sum()),
               'MSW50': kmf.median_survival_time_}
        for t in checkpoints:
            sf = kmf.survival_function_at_times([t]).values[0]
            row[f'T{t}'] = round((1 - sf) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def log_rank_test(df: pd.DataFrame) -> None:
    """Run multivariate log-rank test across all crisis levels."""
    result = multivariate_logrank_test(
        df['T'], df['crisis_level'], df['E'])
    print(f"\nLog-rank test: χ²={result.test_statistic:.1f}, "
          f"p={result.p_value:.3e}")
    return result


# ── Censoring Sensitivity Analysis ─────────────────────────
def censoring_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare KM estimates under three censoring assumptions:
      - original : right-censor at final turn
      - lower    : assume all censored sessions disclose at final turn
      - upper    : assume all censored sessions never disclose
    """
    results = []
    for scenario in ['original', 'lower_bound', 'upper_bound']:
        df_s = df.copy()
        if scenario == 'lower_bound':
            # Optimistic: censored → detected at final turn
            df_s.loc[df_s['E'] == 0, 'E'] = 1
            df_s.loc[df_s['E'] == 1, 'T'] = df_s.loc[
                df_s['E'] == 1, 'T'].clip(upper=df_s['duration'])
        elif scenario == 'upper_bound':
            # Conservative: censored → never detected (extend T to 999)
            df_s.loc[df_s['E'] == 0, 'T'] = 999

        for level in ['Emergency', 'Normal']:
            sub = df_s[df_s['crisis_level'] == level]
            kmf = KaplanMeierFitter()
            kmf.fit(sub['T'], sub['E'])
            msw50 = kmf.median_survival_time_
            t50   = 1 - kmf.survival_function_at_times([50]).values[0]
            results.append({
                'scenario': scenario,
                'crisis_level': level,
                'MSW50': msw50,
                'T50_pct': round(t50 * 100, 1)
            })

    return pd.DataFrame(results)


# ── GLM Discrete-Time Hazard Model ──────────────────────────
def prepare_person_period(df: pd.DataFrame,
                           max_turn: int = 120) -> pd.DataFrame:
    """
    Expand session-level data into person-period (turn-level) format
    for discrete-time hazard modeling.

    Each row = one session-turn combination.
    Event indicator = 1 only at the disclosure turn.
    """
    records = []
    for _, row in df.iterrows():
        T = int(min(row['T'], max_turn))
        for t in range(1, T + 1):
            event = 1 if (row['E'] == 1 and t == T) else 0
            # Bin turn into 10-turn intervals
            if   t <= 10:  turn_bin = '01_T1_10'
            elif t <= 20:  turn_bin = '02_T11_20'
            elif t <= 30:  turn_bin = '03_T21_30'
            elif t <= 50:  turn_bin = '04_T31_50'
            elif t <= 100: turn_bin = '05_T51_100'
            else:          turn_bin = '06_T101plus'
            records.append({
                'session_id':   row['session_id'],
                'turn':         t,
                'turn_bin':     turn_bin,
                'crisis_level': row['crisis_level'],
                'event':        event
            })
    return pd.DataFrame(records)


def fit_hazard_model(pp_df: pd.DataFrame) -> sm.GEE:
    """
    Fit discrete-time logistic hazard model via GEE
    with session-clustered standard errors.

    Formula: event ~ turn_bin + crisis_level
    Reference: turn_bin=T1_10, crisis_level=Observation
    """
    # Set reference categories
    pp_df['turn_bin'] = pd.Categorical(
        pp_df['turn_bin'],
        categories=['01_T1_10','02_T11_20','03_T21_30',
                    '04_T31_50','05_T51_100','06_T101plus'],
        ordered=True)
    pp_df['crisis_level'] = pd.Categorical(
        pp_df['crisis_level'],
        categories=CRISIS_LEVELS)

    model = smf.gee(
        'event ~ C(turn_bin) + C(crisis_level)',
        groups='session_id',
        data=pp_df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()
    )
    result = model.fit()
    print(result.summary())
    return result


# ── Marginal Gain Analysis ──────────────────────────────────
def compute_marginal_gains(kmf_dict: dict,
                            max_turn: int = 100) -> pd.DataFrame:
    """
    Compute turn-by-turn marginal gain in cumulative detection:
        gain(t) = [1-S(t+1)] - [1-S(t)] = S(t) - S(t+1)

    Used to identify diminishing-return thresholds per crisis level.
    """
    rows = []
    for level, kmf in kmf_dict.items():
        turns = list(range(1, max_turn + 1))
        sf = kmf.survival_function_at_times(turns).values
        for i, t in enumerate(turns[:-1]):
            gain = sf[i] - sf[i + 1]
            rows.append({
                'crisis_level': level,
                'turn': t,
                'marginal_gain': round(gain, 5)
            })
    return pd.DataFrame(rows)


# ── Visualization ───────────────────────────────────────────
def plot_km_trajectories(kmf_dict: dict,
                          save_path: str = 'figure_traj.pdf') -> None:
    """
    Plot crisis-stratified KM trajectories (1-S(t)) with
    confidence bands and annotated convergence/divergence zones.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for level, color in zip(CRISIS_LEVELS, COLORS):
        kmf = kmf_dict[level]
        t   = kmf.survival_function_.index.values
        s   = kmf.survival_function_['KM_estimate'].values
        ci_lower = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        ci_upper = kmf.confidence_interval_['KM_estimate_upper_0.95'].values

        lw = 2.4 if level in ['Emergency', 'Normal'] else 1.2
        ls = '-'  if level in ['Emergency', 'Normal'] else '--'
        ax.plot(t, 1 - s, color=color, lw=lw, ls=ls, label=level)
        ax.fill_between(t, 1 - ci_upper, 1 - ci_lower,
                        alpha=0.08, color=color)

    # Convergence zone
    ax.axvspan(9, 12, alpha=0.15, color='#F59E0B')
    ax.text(10.5, 0.25, 'MSW₅₀\n9–11t', fontsize=7,
            color='#92600A', ha='center', style='italic')

    # 25pp bracket at T50
    ax.axvline(50, color='#bbb', lw=0.7, ls=':')
    ax.annotate('', xy=(50, 0.97), xytext=(50, 0.72),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=0.9))
    ax.text(52, 0.848, '↕ 25pp', fontsize=9, color='#666', style='italic')

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Conversational turn', fontsize=10)
    ax.set_ylabel('Cumulative detection  1−S(t)', fontsize=10)
    ax.set_title('Same median, different trajectories', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.06)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"Saved: {save_path}")


# ── Main Pipeline ───────────────────────────────────────────
def main(data_path: str = 'data/sessions.csv'):
    print("=" * 60)
    print("Temporal Disclosure Trajectory Analysis")
    print("=" * 60)

    # 1. Load data
    df = load_data(data_path)

    # 2. Kaplan-Meier
    print("\n--- Kaplan-Meier Estimates ---")
    kmf_dict = fit_km_by_crisis(df)

    # 3. Detection table
    detection_table = compute_detection_table(kmf_dict)
    detection_table.to_csv('results/full_detection_table.csv', index=False)
    print("\n--- Cumulative Detection Table ---")
    print(detection_table.to_string(index=False))

    # 4. Log-rank test
    log_rank_test(df)

    # 5. Censoring sensitivity
    print("\n--- Censoring Sensitivity ---")
    sensitivity = censoring_sensitivity(df)
    print(sensitivity.to_string(index=False))

    # 6. Person-period expansion + GLM
    print("\n--- GLM Discrete-Time Hazard Model ---")
    print("Expanding to person-period format (may take a moment)...")
    pp_df = prepare_person_period(df)
    glm_result = fit_hazard_model(pp_df)

    # 7. Marginal gains
    mg_df = compute_marginal_gains(kmf_dict)
    mg_df.to_csv('results/marginal_gains.csv', index=False)

    # 8. Plot
    plot_km_trajectories(kmf_dict, save_path='results/figure_traj.pdf')

    print("\nDone. Results saved to results/")


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sessions.csv'
    main(data_path)
