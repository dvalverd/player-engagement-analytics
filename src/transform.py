
#Stage 2: ETL Pipeline & Data Warehouse
#Reads raw CSVs from data/raw/, cleans and transforms them,
#then loads into a DuckDB warehouse modeled as a star schema.




import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROCESSED_DIR / "warehouse.duckdb"



# Static fallback data (used when APIs failed)

STATIC_GAMES = pd.DataFrame([
    {"game_id":1,"name":"Grand Theft Auto V","released":"2013-09-17","rating":4.47,"ratings_count":6823,"metacritic":97,"playtime_avg_hrs":73,"genres":"Action, Adventure","platforms":"PC, PS5","esrb_rating":"Mature"},
    {"game_id":2,"name":"The Witcher 3","released":"2015-05-18","rating":4.67,"ratings_count":5921,"metacritic":93,"playtime_avg_hrs":102,"genres":"RPG, Adventure","platforms":"PC, PS5","esrb_rating":"Mature"},
    {"game_id":3,"name":"Cyberpunk 2077","released":"2020-12-10","rating":4.12,"ratings_count":4102,"metacritic":86,"playtime_avg_hrs":61,"genres":"RPG, Action","platforms":"PC, PS5","esrb_rating":"Mature"},
    {"game_id":4,"name":"Elden Ring","released":"2022-02-25","rating":4.65,"ratings_count":3841,"metacritic":96,"playtime_avg_hrs":58,"genres":"RPG, Action","platforms":"PC, PS5","esrb_rating":"Mature"},
    {"game_id":5,"name":"Hades","released":"2020-09-17","rating":4.54,"ratings_count":2901,"metacritic":93,"playtime_avg_hrs":29,"genres":"Action, Indie","platforms":"PC","esrb_rating":"Teen"},
    {"game_id":6,"name":"Stardew Valley","released":"2016-02-26","rating":4.56,"ratings_count":3712,"metacritic":89,"playtime_avg_hrs":58,"genres":"Simulation, RPG","platforms":"PC, PS5","esrb_rating":"Everyone"},
    {"game_id":7,"name":"Among Us","released":"2018-11-16","rating":3.84,"ratings_count":2145,"metacritic":85,"playtime_avg_hrs":12,"genres":"Casual, Indie","platforms":"PC","esrb_rating":"Everyone 10+"},
    {"game_id":8,"name":"Counter-Strike 2","released":"2012-08-21","rating":3.97,"ratings_count":5301,"metacritic":83,"playtime_avg_hrs":324,"genres":"Action, Shooter","platforms":"PC","esrb_rating":"Mature"},
    {"game_id":9,"name":"Apex Legends","released":"2019-02-04","rating":4.13,"ratings_count":3201,"metacritic":88,"playtime_avg_hrs":98,"genres":"Action, Shooter","platforms":"PC, PS5","esrb_rating":"Teen"},
    {"game_id":10,"name":"Dota 2","released":"2013-07-09","rating":3.95,"ratings_count":4100,"metacritic":90,"playtime_avg_hrs":400,"genres":"Strategy, Action","platforms":"PC","esrb_rating":"Teen"},
])

STATIC_STEAMSPY = pd.DataFrame([
    {"app_id":271590,"name":"Grand Theft Auto V","players_2weeks":85000,"players_forever":3200000,"average_2weeks_mins":312,"median_2weeks_mins":186,"average_forever_mins":4380,"median_forever_mins":1560,"positive_reviews":1042000,"negative_reviews":56000,"price_usd":29.99},
    {"app_id":292030,"name":"The Witcher 3","players_2weeks":42000,"players_forever":1800000,"average_2weeks_mins":480,"median_2weeks_mins":320,"average_forever_mins":6120,"median_forever_mins":3240,"positive_reviews":412000,"negative_reviews":8500,"price_usd":39.99},
    {"app_id":1091500,"name":"Cyberpunk 2077","players_2weeks":55000,"players_forever":2100000,"average_2weeks_mins":380,"median_2weeks_mins":240,"average_forever_mins":3660,"median_forever_mins":1980,"positive_reviews":390000,"negative_reviews":62000,"price_usd":59.99},
    {"app_id":1245620,"name":"Elden Ring","players_2weeks":38000,"players_forever":1400000,"average_2weeks_mins":360,"median_2weeks_mins":220,"average_forever_mins":3480,"median_forever_mins":1800,"positive_reviews":375000,"negative_reviews":21000,"price_usd":59.99},
    {"app_id":1145360,"name":"Hades","players_2weeks":18000,"players_forever":820000,"average_2weeks_mins":240,"median_2weeks_mins":160,"average_forever_mins":1740,"median_forever_mins":900,"positive_reviews":175000,"negative_reviews":3200,"price_usd":24.99},
    {"app_id":413150,"name":"Stardew Valley","players_2weeks":28000,"players_forever":1200000,"average_2weeks_mins":420,"median_2weeks_mins":280,"average_forever_mins":3480,"median_forever_mins":1800,"positive_reviews":410000,"negative_reviews":6200,"price_usd":14.99},
    {"app_id":945360,"name":"Among Us","players_2weeks":15000,"players_forever":900000,"average_2weeks_mins":90,"median_2weeks_mins":60,"average_forever_mins":720,"median_forever_mins":360,"positive_reviews":320000,"negative_reviews":25000,"price_usd":0.0},
    {"app_id":730,"name":"Counter-Strike 2","players_2weeks":950000,"players_forever":8200000,"average_2weeks_mins":780,"median_2weeks_mins":480,"average_forever_mins":19440,"median_forever_mins":8640,"positive_reviews":2100000,"negative_reviews":580000,"price_usd":0.0},
    {"app_id":1172620,"name":"Apex Legends","players_2weeks":320000,"players_forever":4200000,"average_2weeks_mins":480,"median_2weeks_mins":300,"average_forever_mins":5760,"median_forever_mins":2520,"positive_reviews":580000,"negative_reviews":95000,"price_usd":0.0},
    {"app_id":570,"name":"Dota 2","players_2weeks":620000,"players_forever":5800000,"average_2weeks_mins":900,"median_2weeks_mins":600,"average_forever_mins":23760,"median_forever_mins":10800,"positive_reviews":1400000,"negative_reviews":350000,"price_usd":0.0},
])



# Extract


def extract():
    print("Extracting raw data...")
    events = pd.read_csv(RAW_DIR / "player_events.csv", parse_dates=["timestamp"])

    try:
        games = pd.read_csv(RAW_DIR / "rawg_games.csv")
        if games.empty or len(games.columns) == 0:
            raise ValueError("empty")
        print(f"  games     -> {len(games)} rows (from CSV)")
    except Exception:
        print("  games     -> using static fallback")
        games = STATIC_GAMES.copy()

    try:
        steamspy = pd.read_csv(RAW_DIR / "steamspy_stats.csv")
        if steamspy.empty or len(steamspy.columns) == 0:
            raise ValueError("empty")
        print(f"  steamspy  -> {len(steamspy)} rows (from CSV)")
    except Exception:
        print("  steamspy  -> using static fallback")
        steamspy = STATIC_STEAMSPY.copy()

    print(f"  events    -> {len(events):,} rows\n")
    return events, games, steamspy


# Transform


def build_dim_games(games, steamspy):
    steamspy["name_clean"] = steamspy["name"].str.lower().str.strip()
    games["name_clean"]    = games["name"].str.lower().str.strip()
    merged = games.merge(steamspy, on="name_clean", how="left", suffixes=("_rawg", "_spy"))

    dim = pd.DataFrame({
        "game_key":         range(1, len(merged) + 1),
        "game_name":        merged["name_rawg"].fillna(merged.get("name_spy", "")),
        "genre":            merged["genres"].fillna("Unknown"),
        "platform_tags":    merged["platforms"].fillna("Unknown"),
        "esrb_rating":      merged["esrb_rating"].fillna("Not Rated"),
        "rawg_rating":      merged["rating"].fillna(0).round(2),
        "metacritic_score": merged["metacritic"].fillna(0).astype(int),
        "avg_playtime_hrs": merged["playtime_avg_hrs"].fillna(0).astype(int),
        "price_usd":        merged["price_usd"].fillna(0).round(2),
        "positive_reviews": merged["positive_reviews"].fillna(0).astype(int),
        "negative_reviews": merged["negative_reviews"].fillna(0).astype(int),
        "players_2weeks":   merged["players_2weeks"].fillna(0).astype(int),
        "released":         pd.to_datetime(merged["released"], errors="coerce"),
    })

    total = dim["positive_reviews"] + dim["negative_reviews"]
    dim["review_sentiment"] = np.where(
        total > 0, (dim["positive_reviews"] / total * 100).round(1), 0
    )
    dim["primary_genre"] = dim["genre"].str.split(",").str[0].str.strip()
    return dim


def build_dim_players(events):
    sessions  = events[events["event_type"] == "session_start"]
    purchases = events[events["event_type"] == "purchase"]
    churned   = events[events["event_type"] == "quit_game"]

    session_agg = sessions.groupby("player_id").agg(
        total_sessions=("event_type", "count"),
        total_playtime_mins=("session_duration_mins", "sum"),
        avg_session_mins=("session_duration_mins", "mean"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        favorite_game=("game", lambda x: x.value_counts().idxmax()),
        favorite_platform=("platform", lambda x: x.value_counts().idxmax()),
        region=("region", "first"),
        player_type=("player_type", "first"),
    ).reset_index()

    purchase_agg = purchases.groupby("player_id").agg(
        total_purchases=("event_type", "count"),
        total_spent_usd=("amount_usd", "sum"),
    ).reset_index()

    churn_flag = churned[["player_id"]].drop_duplicates()
    churn_flag["is_churned"] = True

    dim = session_agg.merge(purchase_agg, on="player_id", how="left")
    dim = dim.merge(churn_flag, on="player_id", how="left")

    dim["total_purchases"]   = dim["total_purchases"].fillna(0).astype(int)
    dim["total_spent_usd"]   = dim["total_spent_usd"].fillna(0).round(2)
    dim["is_churned"]        = dim["is_churned"].fillna(False)
    dim["avg_session_mins"]  = dim["avg_session_mins"].round(1)
    dim["total_playtime_hrs"]= (dim["total_playtime_mins"] / 60).round(1)
    dim["days_active"]       = (dim["last_seen"] - dim["first_seen"]).dt.days + 1
    dim.insert(0, "player_key", range(1, len(dim) + 1))
    return dim


def build_fact_sessions(events, dim_players, dim_games):
    sessions = events[events["event_type"] == "session_start"].copy()

    player_map = dim_players.set_index("player_id")["player_key"]
    game_map   = dim_games.set_index("game_name")["game_key"]

    sessions["player_key"] = sessions["player_id"].map(player_map)
    sessions["game_key"]   = sessions["game"].map(game_map)

    sessions["date"]        = sessions["timestamp"].dt.date
    sessions["hour"]        = sessions["timestamp"].dt.hour
    sessions["day_of_week"] = sessions["timestamp"].dt.day_name()
    sessions["week"]        = sessions["timestamp"].dt.isocalendar().week.astype(int)
    sessions["month"]       = sessions["timestamp"].dt.month
    sessions["is_weekend"]  = sessions["timestamp"].dt.dayofweek >= 5

    fact = sessions[[
        "player_key", "game_key", "timestamp", "date",
        "hour", "day_of_week", "week", "month", "is_weekend",
        "platform", "region", "player_type", "session_duration_mins",
    ]].copy()

    fact.insert(0, "session_key", range(1, len(fact) + 1))
    fact = fact.rename(columns={"session_duration_mins": "duration_mins"})
    return fact


def build_dim_date(fact):
    dates = pd.date_range(
        start=pd.to_datetime(fact["date"]).min(),
        end=pd.to_datetime(fact["date"]).max(),
        freq="D",
    )
    return pd.DataFrame({
        "date":        dates.date,
        "year":        dates.year,
        "month":       dates.month,
        "month_name":  dates.month_name(),
        "week":        dates.isocalendar().week.astype(int),
        "day_of_week": dates.day_name(),
        "is_weekend":  dates.dayofweek >= 5,
        "quarter":     dates.quarter,
    })



# Load into DuckDB


def load_to_warehouse(fact_sessions, dim_players, dim_games, dim_date):
    print("Loading into DuckDB warehouse...")
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    tables = {
        "dim_games":     dim_games,
        "dim_players":   dim_players,
        "dim_date":      dim_date,
        "fact_sessions": fact_sessions,
    }
    for name, df in tables.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        print(f"  + {name:20s} -> {len(df):>6,} rows")

    con.execute("CREATE INDEX idx_fact_player ON fact_sessions(player_key)")
    con.execute("CREATE INDEX idx_fact_game   ON fact_sessions(game_key)")

    total = con.execute("""
        SELECT COUNT(*) as sessions,
               ROUND(SUM(duration_mins)/60.0,1) as total_hrs,
               COUNT(DISTINCT player_key) as players,
               COUNT(DISTINCT game_key) as games
        FROM fact_sessions
    """).fetchdf()
    con.close()

    print(f"\n  Warehouse validated:")
    print(f"  {total.to_string(index=False)}")



# Save processed CSVs


def save_processed(fact, dim_players, dim_games, dim_date):
    fact.to_csv(PROCESSED_DIR / "fact_sessions.csv", index=False)
    dim_players.to_csv(PROCESSED_DIR / "dim_players.csv", index=False)
    dim_games.to_csv(PROCESSED_DIR / "dim_games.csv", index=False)
    dim_date.to_csv(PROCESSED_DIR / "dim_date.csv", index=False)
    print(f"\n  CSVs saved to {PROCESSED_DIR}")



# Main


def run_transform():
    print("=" * 55)
    print("  Player Engagement Analytics -- Stage 2: Transform")
    print("=" * 55 + "\n")

    events, games, steamspy = extract()

    print("Building star schema...")
    dim_games   = build_dim_games(games, steamspy)
    dim_players = build_dim_players(events)
    fact        = build_fact_sessions(events, dim_players, dim_games)
    dim_date    = build_dim_date(fact)

    print(f"  dim_games   -> {len(dim_games)} rows")
    print(f"  dim_players -> {len(dim_players)} rows")
    print(f"  fact        -> {len(fact):,} rows")
    print(f"  dim_date    -> {len(dim_date)} rows\n")

    load_to_warehouse(fact, dim_players, dim_games, dim_date)
    save_processed(fact, dim_players, dim_games, dim_date)

    print("\n" + "-" * 55)
    print("  Stage 2 complete.")
    print("-" * 55)

    return fact, dim_players, dim_games, dim_date


if __name__ == "__main__":
    run_transform()
