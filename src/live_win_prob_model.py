"""
Live in-match win probability model.

Given a match's current state (score, possession, xG, cards, etc.) plus its
pre-match prior from prematch_prior.py, produces a live-updating win/draw/loss
probability and model-implied fair odds.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_possession_pct(row, spell_bounds_clean, window_seconds: int = 600):
    """
    Rolling possession percentage for the home and away team, over the trailing
    `window_seconds` (default 10 minutes) leading up to this snapshot's timestamp.

    Returns 0.5/0.5 if no possession spells exist yet in the window (e.g. minute 0).
    """
    window_start = row['snapshot_seconds'] - window_seconds
    window_end = row['snapshot_seconds']

    relevant_spells = spell_bounds_clean[
        (spell_bounds_clean['match_id'] == row['match_id']) &
        (spell_bounds_clean['spell_start'] >= window_start) &
        (spell_bounds_clean['spell_start'] < window_end)
    ]

    if len(relevant_spells) == 0:
        return pd.Series({'home_possession_pct': 0.5, 'away_possession_pct': 0.5})

    team_totals = relevant_spells.groupby('possession_team')['spell_duration'].sum()
    total_time = team_totals.sum()

    home_seconds = team_totals.get(row['home_team'], 0)
    away_seconds = team_totals.get(row['away_team'], 0)

    return pd.Series({
        'home_possession_pct': home_seconds / total_time,
        'away_possession_pct': away_seconds / total_time
    })


def compute_shot_momentum(row, momentum_events, window_seconds: int = 300):
    """
    Count of shots by each team in the trailing `window_seconds` (default 5 minutes)
    leading up to this snapshot's timestamp. Returns 0/0 if no shots occurred yet.
    """
    window_start = row['snapshot_seconds'] - window_seconds
    window_end = row['snapshot_seconds']

    relevant_shots = momentum_events[
        (momentum_events['match_id'] == row['match_id']) &
        (momentum_events['total_seconds'] >= window_start) &
        (momentum_events['total_seconds'] < window_end)
    ]

    if len(relevant_shots) == 0:
        return pd.Series({'home_shot_momentum': 0, 'away_shot_momentum': 0})

    home_shots = (relevant_shots['team'] == row['home_team']).sum()
    away_shots = (relevant_shots['team'] == row['away_team']).sum()

    return pd.Series({'home_shot_momentum': home_shots, 'away_shot_momentum': away_shots})


# ---------------------------------------------------------------------------
# Deliverable functions
# ---------------------------------------------------------------------------

def get_live_win_prob(match_state: dict, model, feature_cols: list) -> dict:
    """
    Given a single match's live state (a dict with the same keys as feature_cols)
    and a fitted model, return live win/draw/loss probabilities.

    match_state must include: home_possession_pct, away_possession_pct, score_diff,
    home_xg_cum, away_xg_cum, man_advantage, home_shot_momentum, away_shot_momentum,
    prob_home_win, prob_draw, prob_away_win (the last three coming from
    get_prematch_prior(), attached once per match and passed through unchanged).

    Assumes the model's encoded classes map 0=away_win, 1=draw, 2=home_win
    (confirmed against this project's LabelEncoder ordering).
    """
    X = pd.DataFrame([match_state])[feature_cols]
    probs = model.predict_proba(X)[0]

    return {
        'prob_away_win': probs[0],
        'prob_draw': probs[1],
        'prob_home_win': probs[2],
    }


def get_live_fair_odds(match_state: dict, model, feature_cols: list) -> dict:
    """
    Model-implied fair odds (no bookmaker margin) for the current live match state,
    derived from get_live_win_prob(). Decimal odds = 1 / probability.
    """
    probs = get_live_win_prob(match_state, model, feature_cols)
    return {
        'odds_home': 1 / probs['prob_home_win'],
        'odds_draw': 1 / probs['prob_draw'],
        'odds_away': 1 / probs['prob_away_win'],
    }