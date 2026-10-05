# src/prematch_prior.py

import pandas as pd

INITIAL_ELO = 1500
HOME_ADVANTAGE = 100 
TEAM_NAME_TO_ELO_NAME = {
    'AFC Bournemouth': 'Bournemouth',
    'AS Roma': 'Roma',
    'Atlético Madrid': 'Atletico Madrid',
    'Hellas Verona': 'Verona',
    'Inter Milan': 'Inter',
    'Leicester City': 'Leicester',
    'Levante UD': 'Levante',
    'Málaga': 'Malaga',
    'Norwich City': 'Norwich',
    'RC Deportivo La Coruña': 'Deportivo La Coruna',
    'Sporting Gijón': 'Sporting Gijon',
    'Stoke City': 'Stoke',
    'Swansea City': 'Swansea',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
}

def expected_score(rating_a, rating_b, home_advantage=0):
    return 1 / (1 + 10 ** ((rating_b - (rating_a + home_advantage)) / 400))


def get_empirical_draw_probability(expected_score, bin_edges, draw_rate_by_bin):
    new_bin = pd.cut([expected_score], bins=bin_edges)[0]
    if pd.isna(new_bin):
        return draw_rate_by_bin.mean()
    draw_probability = draw_rate_by_bin[new_bin]
    if pd.isna(draw_probability):
        return draw_rate_by_bin.mean()
    return draw_probability


def get_team_elo_as_of(team, date, team_elo_history, initial_elo=INITIAL_ELO):
    prior_matches = team_elo_history[
        (team_elo_history['team'] == team) &
        (team_elo_history['date'] < date)
    ]
    if len(prior_matches) == 0:
        return initial_elo
    return prior_matches.sort_values('date').iloc[-1]['elo_post']


def get_prematch_prior(home_team, away_team, date, team_elo_history, bin_edges, draw_rate_by_bin, home_advantage=HOME_ADVANTAGE):
    home_team_elo_name = TEAM_NAME_TO_ELO_NAME.get(home_team, home_team)
    away_team_elo_name = TEAM_NAME_TO_ELO_NAME.get(away_team, away_team)

    home_rating = get_team_elo_as_of(home_team_elo_name, date, team_elo_history)
    away_rating = get_team_elo_as_of(away_team_elo_name, date, team_elo_history)

    exp_home = expected_score(home_rating, away_rating, home_advantage=home_advantage)
    draw_prob = get_empirical_draw_probability(exp_home, bin_edges, draw_rate_by_bin)
    prob_home_win = (1 - draw_prob) * exp_home
    prob_away_win = (1 - draw_prob) * (1 - exp_home)

    return {
        'prob_home_win': prob_home_win,
        'prob_draw': draw_prob,
        'prob_away_win': prob_away_win
    }