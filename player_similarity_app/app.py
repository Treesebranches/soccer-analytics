"""
Player Similarity Explorer
---------------------------
Find stylistically similar soccer players across 4 top European leagues
(2015/2016 season), using per-90 stats derived from StatsBomb event data.

Expects these files in a `data/` folder next to this script:
  - player_stats_final.parquet   (from notebook 05's final save)
  - X_scaled.npy                 (standardized feature matrix, same row order)
  - feature_cols.pkl             (list of feature column names used)

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Player Similarity Explorer", page_icon="⚽", layout="wide")


@st.cache_data
def load_data():
    player_stats = pd.read_parquet("data/player_stats_final.parquet")
    X_scaled = np.load("data/X_scaled.npy")
    feature_cols = joblib.load("data/feature_cols.pkl")
    return player_stats, X_scaled, feature_cols


def find_similar_players(player_name, player_stats_df, X_scaled, n=10):
    idx = player_stats_df.index[player_stats_df["player"] == player_name][0]
    idx_pos = player_stats_df.index.get_loc(idx)

    target_vector = X_scaled[idx_pos].reshape(1, -1)
    similarities = cosine_similarity(target_vector, X_scaled)[0]

    result = player_stats_df[["player", "position", "team"]].copy()
    result["similarity"] = similarities
    result = result[result["player"] != player_name].sort_values("similarity", ascending=False)
    return result.head(n)


def make_radar_chart(player_stats_df, feature_cols, players, labels):
    # Show a compact, readable subset of stats on the radar -- all 21 features
    # would be unreadable at this scale.
    display_features = [
        "goals_p90", "assists_p90", "progressive_passes_p90", "progressive_carries_p90",
        "pressures_p90", "tackle_attempts_p90", "interceptions_p90", "aerial_pct",
    ]
    display_features = [f for f in display_features if f in feature_cols]

    fig = go.Figure()
    for player, label in zip(players, labels):
        row = player_stats_df[player_stats_df["player"] == player][display_features].iloc[0]
        # Normalize each stat to the 0-1 range across the full population for fair radar comparison
        normalized = []
        for f in display_features:
            col = player_stats_df[f]
            val = (row[f] - col.min()) / (col.max() - col.min() + 1e-9)
            normalized.append(val)
        fig.add_trace(go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=display_features + [display_features[0]],
            fill="toself",
            name=label,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
    )
    return fig


def main():
    st.title("⚽ Player Similarity Explorer")
    st.markdown(
        "Find stylistically similar players across the Premier League, La Liga, "
        "Serie A, and Ligue 1 (2015/2016 season) — regardless of club reputation "
        "or price tag. Built on 21 per-90 features derived from raw StatsBomb event "
        "data, including progressive passes/carries and average shot distance "
        "computed from pitch location rather than pulled pre-made from a provider."
    )

    try:
        player_stats, X_scaled, feature_cols = load_data()
    except FileNotFoundError as e:
        st.error(
            f"Couldn't find required data file: {e}. "
            "Make sure `data/player_stats_final.parquet`, `data/X_scaled.npy`, "
            "and `data/feature_cols.pkl` are present next to this app."
        )
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        player_list = sorted(player_stats["player"].unique())
        selected_player = st.selectbox("Choose a player", player_list, index=player_list.index("Harry Kane") if "Harry Kane" in player_list else 0)
        n_results = st.slider("Number of similar players", min_value=3, max_value=20, value=10)

        player_row = player_stats[player_stats["player"] == selected_player].iloc[0]
        st.markdown(f"**Position:** {player_row['position']}")
        st.markdown(f"**Club:** {player_row['team']}")
        st.markdown(f"**Minutes played:** {player_row['total_minutes']:.0f}")

    with col2:
        results = find_similar_players(selected_player, player_stats, X_scaled, n=n_results)
        st.subheader(f"Players most similar to {selected_player}")
        st.dataframe(
            results.rename(columns={
                "player": "Player", "position": "Position", "team": "Club", "similarity": "Similarity"
            }).style.format({"Similarity": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Stat profile comparison")
    compare_with = st.multiselect(
        "Compare the selected player against (pick up to 3 from the similar-players list)",
        options=results["player"].tolist(),
        default=results["player"].tolist()[:2] if len(results) >= 2 else results["player"].tolist(),
        max_selections=3,
    )

    if compare_with:
        players_to_plot = [selected_player] + compare_with
        labels_to_plot = [selected_player] + compare_with
        fig = make_radar_chart(player_stats, feature_cols, players_to_plot, labels_to_plot)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one player above to compare stat profiles.")

    st.divider()
    st.caption(
        "Built as part of a larger soccer analytics project covering pre-match and "
        "live in-match win probability modeling, benchmarked against real bookmaker "
        "odds. Data: StatsBomb open data, 2015/2016 season. Similarity is computed "
        "via cosine similarity on standardized per-90 stats, chosen specifically to "
        "compare playing style rather than overall output level."
    )


if __name__ == "__main__":
    main()
