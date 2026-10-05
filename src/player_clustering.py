"""
Player similarity engine, built on StatsBomb-derived per-90 stats across La Liga,
Premier League, Serie A, and Ligue 1 (2015/2016).

Given a player's standardized feature vector, finds the most stylistically similar
players in the dataset using cosine similarity -- deliberately chosen over Euclidean
distance, since cosine similarity compares the relative shape/emphasis of a player's
stat profile rather than raw magnitude, which better separates playing STYLE from
overall output level (a real distinction found during two-stage sub-cluster testing,
where overall quality dominated a magnitude-sensitive approach).
"""

from sklearn.metrics.pairwise import cosine_similarity


def find_similar_players(player_name: str, player_stats_df, X_scaled, n: int = 5):
    """
    Find the n most similar players to player_name, by cosine similarity on their
    standardized per-90 feature vectors.

    player_stats_df must have a 'player' column, and its row order must match X_scaled
    exactly (X_scaled is the output of StandardScaler().fit_transform() on
    player_stats_df's feature columns).
    """
    if player_name not in player_stats_df['player'].values:
        return f"'{player_name}' not found in dataset (check spelling, or they may not meet the minutes threshold)"

    idx = player_stats_df.index[player_stats_df['player'] == player_name][0]
    idx_pos = player_stats_df.index.get_loc(idx)

    target_vector = X_scaled[idx_pos].reshape(1, -1)
    similarities = cosine_similarity(target_vector, X_scaled)[0]

    result = player_stats_df[['player', 'position', 'team']].copy()
    result['similarity'] = similarities

    result = result[result['player'] != player_name].sort_values('similarity', ascending=False)
    return result.head(n)