"""
    Data Loading functions shared across notebooks
    
"""

from pathlib import Path
import pandas as pd
from statsbombpy import sb

# Project Root so paths work regardless of what notebook calls it
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# -----------------------------------------------------------------------------------------------------------------------------
# StatsBomb (live in-math event data- Tool 2)
def list_available_competitions() -> pd.DataFrame:
    # Return all StatsBomb open-data competitions available
    return sb.competitions()

def load_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Return the full match list for one competition/season
    Confirmed full-coverage club competitions (2015/2016 season, 380 matches each):
      - La Liga:          competition_id=11, season_id=27
      - Premier League:   competition_id=2,  season_id=27
      - Serie A:          competition_id=12, season_id=27
      
      Same season across all three leagues keeps event-tagging conventions and
    xG methodology consistent. (World Cup and Europa League were considered and
    dropped: international tournaments play differently enough to be considered for a separate project,
    and Europa League only has one old, sparsely-covered season with no modern
    event detail in StatsBomb's open data.)
    """
    return sb.matches(competition_id=competition_id, season_id=season_id)

def load_statsbomb_events(match_id: int) -> pd.DataFrame:
    # Return every evenmt (pass, shot, tackle) for a single match
    return sb.events(match_id=match_id)

def load_statsbomb_events_for_competition(competition_id: int, season_id: int, save: bool = True) -> pd.DataFrame:
    # Pull and concatenate events for every match in a competition
    cache_path = RAW_DIR/f"statsbomb_events_{competition_id}_{season_id}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    matches = load_statsbomb_matches(competition_id, season_id)
    all_events = []
    for match_id in matches ["match_id"]:
        events = load_statsbomb_events(match_id)
        events["match_id"] = match_id
        all_events.append(events)
    
    combined = pd.concat(all_events, ignore_index = True)
    combined = combined.drop_duplicates(subset=['id']).reset_index(drop=True)  # safety net against silent re-pull corruption
    if save:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path)
    return combined
    
# -----------------------------------------------------------------------------------------------------------------------------
# Understat (pre-match team strength data - Tool 1)

def load_understat_team_data(league: str, season: str) -> pd.DataFrame:
    #Pull season long team xG/xGA from Understat
    import soccerdata as sd
    understat = sd.Understat(leagues=league, seasons=season)
    return understat.read_team_match_stats()

def load_understat_team_data_multi(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    """
    Pull Understat team data across multiple leagues and seasons, concatenated
    into one dataframe with 'league' and 'season' columns for filtering later.
 
    Matches the three-league StatsBomb set by default:
        leagues = ['ENG-Premier League', 'ESP-La Liga', 'ITA-Serie A']
    """
    import soccerdata as sd
    frames = []
    for league in leagues:
        understat = sd.Understat(leagues=league, seasons=seasons)
        df = understat.read_team_match_stats()
        df["league"] = league
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
 
# -----------------------------------------------------------------------------------------------------------------------------
# FBref (player-season stats - part B clustering)

def load_fbref_player_stats(league: str, season: str, stat_type: str = "standard") -> pd.DataFrame:
    """
    Pull player-season stats from FBref via soccerdata.
    """
    import soccerdata as sd
    fbref = sd.FBref(leagues=league, seasons=season)
    return fbref.read_player_season_stats(stat_type=stat_type)
 
 
def _read_stat_type_with_retry(fbref, stat_type: str, max_retries: int = 3, delay_seconds: int = 5):
    """
    FBref scraping goes through a Selenium/Chrome browser session under the hood
    (not a plain HTTP request like Understat/football-data.co.uk), which occasionally
    hits transient connection hiccups on a cold-started browser. Retry with a short
    delay rather than failing the whole multi-league pull over one flaky call.
    """
    import time
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return fbref.read_player_season_stats(stat_type=stat_type)
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{max_retries} failed for stat_type='{stat_type}': {e}")
            if attempt < max_retries:
                time.sleep(delay_seconds)
    raise last_error
 
def _flatten_and_prefix_columns(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    """
    Flatten FBref's two-level column MultiIndex (group header + stat name) into
    single flat names, prefixed by stat_type.
 
    Using ONLY the second level (stat name) isn't safe: FBref's own tables reuse
    stat names across different group headers within a SINGLE stat_type table --
    e.g. the 'standard' page has both season totals and 'Per 90 Minutes' blocks,
    both containing a 'Gls' column under different group headers. Flattening to
    level 1 alone silently collapses these into duplicate columns (confirmed via
    a real ValueError: 'Duplicate column names found' listing 'standard_Gls' twice,
    'standard_Ast' twice, etc., plus several blank-named columns where the group
    header carried the only real information).
 
    Fix: combine BOTH levels into the column name, then guarantee uniqueness with
    a numeric suffix as a safety net regardless of cause.
    """
    new_cols = []
    seen_counts: dict[str, int] = {}
 
    for level0, level1 in df.columns:
        level0_clean = "" if str(level0).startswith("Unnamed") else str(level0).strip()
        level1_clean = str(level1).strip()
 
        if level0_clean and level1_clean:
            base = f"{stat_type}_{level0_clean}_{level1_clean}"
        elif level1_clean:
            base = f"{stat_type}_{level1_clean}"
        elif level0_clean:
            base = f"{stat_type}_{level0_clean}"
        else:
            base = f"{stat_type}_col"
 
        base = base.replace(" ", "_")
 
        if base in seen_counts:
            seen_counts[base] += 1
            base = f"{base}_{seen_counts[base]}"
        else:
            seen_counts[base] = 0
 
        new_cols.append(base)
 
    df = df.copy()
    df.columns = new_cols
    return df
 
def load_fbref_player_stats_all(season: str, stat_types: list[str] = None) -> pd.DataFrame:
    """
    Pull FBref player-season stats for all 'Big 5' European leagues in one call
    (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), joining several
    stat_types together per player into one wide table ready for clustering.
 
    This replaced an earlier per-league loop after 'ITA-Serie A' consistently
    failed on its own season lookup (read_seasons() returned nothing to
    concatenate, even after retrying season-string formats). Confirmed via
    soccerdata's own source (fbref.py) that 'Big 5 European Leagues Combined'
    is handled as its own code path -- one scrape covers all five leagues,
    each row tagged with a 'league' column via this exact mapping:
        {'Serie A': 'ITA-Serie A', 'Ligue 1': 'FRA-Ligue 1', 'La Liga': 'ESP-La Liga',
         'Premier League': 'ENG-Premier League', 'Fußball-Bundesliga': 'GER-Bundesliga'}
 
    Keeping Bundesliga and Ligue 1 in (rather than filtering down to 3 leagues)
    is a deliberate choice: more players for clustering, and Part B doesn't need
    to match Part A's exact league set the way Part A's own two tools need to
    match each other.
 
    One scrape instead of five also means one Selenium/Chrome browser session
    instead of five separate launches -- more robust against the connection
    flakiness seen with the per-league approach.
 
    Valid stat_types in this soccerdata version: 'standard', 'keeper', 'shooting',
    'playing_time', 'misc'. 'keeper' excluded by default (goalkeepers don't belong
    in outfield-player archetype clustering).
    """
    import soccerdata as sd
    if stat_types is None:
        stat_types = ["standard", "shooting", "playing_time", "misc"]
 
    fbref = sd.FBref(leagues="Big 5 European Leagues Combined", seasons=season)
 
    frames = []
    for stat_type in stat_types:
        df = _read_stat_type_with_retry(fbref, stat_type)
 
        # soccerdata sets league/season/team/player as a ROW index (not columns) on
        # every stat_type table -- confirmed via source: .set_index(["league","season","team","player"]).
        # That's why an earlier version of this function's ['league'] access failed:
        # 'league' was never a column. Since every stat_type table shares this exact
        # same row index, concatenating along columns (axis=1) aligns rows correctly
        # without needing a manual join key.
 
        df = _flatten_and_prefix_columns(df, stat_type)
        frames.append(df)
 
    merged = pd.concat(frames, axis=1)
    merged = merged.reset_index()  # brings league/season/team/player back as normal columns
 
    duplicated = merged.columns[merged.columns.duplicated()].tolist()
    if duplicated:
        print(f"WARNING: duplicate column names remain after flatten+prefix: {duplicated}. "
              f"This shouldn't happen given the numeric-suffix safety net -- inspect manually.")
 
    return merged

# -----------------------------------------------------------------------------------------------------------------------------
#football-data.co.uk - (Part A.3 market comparison)
def load_football_data_odds(league_code: str, season_code: str) -> pd.DataFrame:
    """
    Pull historical match results + odds directly from football-data.co.uk CSVs.
 
    league_code examples: 'E0' (Premier League), 'SP1' (La Liga), 'I1' (Serie A)
    season_code examples: '2425' (2024/2025 season), '2425' (2024/2025)
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    return pd.read_csv(url)
 
 
def load_football_data_odds_multi(league_codes: list[str], season_codes: list[str]) -> pd.DataFrame:
    """
    Pull football-data.co.uk odds across multiple leagues and seasons, concatenated
    into one dataframe tagged with 'league_code' and 'season_code' columns.
 
    Matches the three-league set:
        league_codes = ['E0', 'SP1', 'I1']   # Premier League, La Liga, Serie A
        season_codes  = ['2425', '2526']      # 2024/2025, 2025/2026
 
    """
    frames = []
    for league_code in league_codes:
        for season_code in season_codes:
            try:
                df = load_football_data_odds(league_code, season_code)
            except Exception as e:
                print(f"Skipping {league_code} {season_code}: {e}")
                continue
            df["league_code"] = league_code
            df["season_code"] = season_code
            frames.append(df)
 
    return pd.concat(frames, ignore_index=True)