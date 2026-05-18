

import os
import time
import random
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAWG_API_KEY = os.getenv("RAWG_API_KEY", "")  # Free key at rawg.io — opt


RAWG_GAME_IDS = [
    "grand-theft-auto-v",
    "the-witcher-3-wild-hunt",
    "cyberpunk-2077",
    "elden-ring",
    "red-dead-redemption-2",
    "god-of-war",
    "hades",
    "hollow-knight",
    "stardew-valley",
    "among-us",
    "minecraft",
    "fortnite",
    "apex-legends",
    "valorant",
    "counter-strike-global-offensive",
    "league-of-legends",
    "dota-2",
    "overwatch",
    "destiny-2",
    "monster-hunter-world",
]


def fetch_rawg_games(game_slugs: list[str]) -> pd.DataFrame:

    print("Fetching RAWG game metadata...")
    records = []

    for slug in game_slugs:
        url = f"https://api.rawg.io/api/games/{slug}"
        params = {"key": RAWG_API_KEY} if RAWG_API_KEY else {}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                d = resp.json()
                records.append(
                    {
                        "game_id": d.get("id"),
                        "slug": d.get("slug"),
                        "name": d.get("name"),
                        "released": d.get("released"),
                        "rating": d.get("rating"),
                        "ratings_count": d.get("ratings_count"),
                        "metacritic": d.get("metacritic"),
                        "playtime_avg_hrs": d.get("playtime"),
                        "genres": ", ".join(g["name"] for g in d.get("genres", [])),
                        "platforms": ", ".join(
                            p["platform"]["name"] for p in d.get("platforms", [])
                        ),
                        "esrb_rating": (d.get("esrb_rating") or {}).get("name"),
                    }
                )
                print(f"  {d.get('name')}")
            else:
                print(f" {slug} — status {resp.status_code}")
        except Exception as e:
            print(f"   {slug} — error: {e}")
        time.sleep(0.5)  

    df = pd.DataFrame(records)
    out = RAW_DIR / "rawg_games.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} games → {out}\n")
    return df



STEAM_APP_IDS = {
    271590: "Grand Theft Auto V",
    292030: "The Witcher 3: Wild Hunt",
    1091500: "Cyberpunk 2077",
    1245620: "Elden Ring",
    1174180: "Red Dead Redemption 2",
    1593500: "God of War",
    1145360: "Hades",
    367520: "Hollow Knight",
    413150: "Stardew Valley",
    945360: "Among Us",
    730: "Counter-Strike: Global Offensive",
    570: "Dota 2",
    252950: "Rocket League",
    578080: "PUBG: Battlegrounds",
    1172620: "Apex Legends",
}


def fetch_steamspy_stats(app_ids: dict[int, str]) -> pd.DataFrame:
   
    print("Fetching SteamSpy player stats...")
    records = []

    for app_id, name in app_ids.items():
        url = f"https://steamspy.com/api.php?request=appdetails&appid={app_id}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                d = resp.json()
                records.append(
                    {
                        "app_id": app_id,
                        "name": d.get("name", name),
                        "owners_estimate": d.get("owners", "0 .. 0"),
                        "players_2weeks": d.get("players_2weeks", 0),
                        "players_forever": d.get("players_forever", 0),
                        "average_2weeks_mins": d.get("average_2weeks", 0),
                        "median_2weeks_mins": d.get("median_2weeks", 0),
                        "average_forever_mins": d.get("average_forever", 0),
                        "median_forever_mins": d.get("median_forever", 0),
                        "positive_reviews": d.get("positive", 0),
                        "negative_reviews": d.get("negative", 0),
                        "price_usd": round(d.get("price", 0) / 100, 2),
                    }
                )
                print(f"  {name}")
            else:
                print(f"  {name} — status {resp.status_code}")
        except Exception as e:
            print(f"  {name} — error: {e}")
        time.sleep(1)  # SteamSpy asks for 1s between requests

    df = pd.DataFrame(records)
    out = RAW_DIR / "steamspy_stats.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} games → {out}\n")
    return df




GAME_POOL = [
    "Grand Theft Auto V",
    "The Witcher 3",
    "Cyberpunk 2077",
    "Elden Ring",
    "Hades",
    "Stardew Valley",
    "Among Us",
    "Counter-Strike 2",
    "Dota 2",
    "Apex Legends",
]

EVENT_TYPES = ["session_start", "session_end", "purchase", "achievement", "quit_game"]

PLATFORMS = ["PC", "PlayStation 5", "Xbox Series X"]

REGIONS = ["North America", "Europe", "Asia", "South America", "Oceania"]


def simulate_player_events(
    n_players: int = 500,
    n_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
  #synthetic player logs
    print(f"Simulating {n_players} players over {n_days} days...")
    rng = np.random.default_rng(seed)
    random.seed(seed)

    start_date = datetime.now() - timedelta(days=n_days)


    player_types = rng.choice(
        ["casual", "regular", "hardcore"],
        size=n_players,
        p=[0.5, 0.35, 0.15],
    )
    sessions_per_week = {"casual": 2, "regular": 5, "hardcore": 12}
    session_duration = {
        "casual": (20, 60),
        "regular": (45, 120),
        "hardcore": (90, 300),
    }
    churn_prob = {"casual": 0.04, "regular": 0.015, "hardcore": 0.005}
    spend_prob = {"casual": 0.02, "regular": 0.05, "hardcore": 0.10}
    spend_amount = {
        "casual": (0.99, 9.99),
        "regular": (4.99, 29.99),
        "hardcore": (9.99, 59.99),
    }

    records = []
    player_ids = [f"P{str(i).zfill(5)}" for i in range(1, n_players + 1)]

    for pid, ptype in zip(player_ids, player_types):
        game = random.choice(GAME_POOL)
        platform = random.choice(PLATFORMS)
        region = rng.choice(
            REGIONS, p=[0.35, 0.30, 0.20, 0.10, 0.05]
        )
        signup_day = rng.integers(0, n_days // 3)  

        churned = False
        churn_day = None

        for day in range(signup_day, n_days):
            if churned:
                break


            time_factor = 1 + (day / n_days) * (0.5 if ptype == "casual" else 0.1)
            if rng.random() < churn_prob[ptype] * time_factor:
                churned = True
                churn_day = day
                records.append(
                    {
                        "player_id": pid,
                        "event_type": "quit_game",
                        "game": game,
                        "platform": platform,
                        "region": region,
                        "player_type": ptype,
                        "timestamp": start_date + timedelta(days=day, hours=int(rng.integers(8, 23))),
                        "session_duration_mins": None,
                        "amount_usd": None,
                        "is_churned": True,
                    }
                )
                break


            weekly_rate = sessions_per_week[ptype]
            n_sessions_today = rng.poisson(weekly_rate / 7)

            for _ in range(n_sessions_today):
                hour = int(rng.integers(8, 24))
                ts = start_date + timedelta(days=day, hours=hour, minutes=int(rng.integers(0, 60)))
                dur_min, dur_max = session_duration[ptype]
                duration = int(rng.integers(dur_min, dur_max))

      
                records.append(
                    {
                        "player_id": pid,
                        "event_type": "session_start",
                        "game": game,
                        "platform": platform,
                        "region": region,
                        "player_type": ptype,
                        "timestamp": ts,
                        "session_duration_mins": duration,
                        "amount_usd": None,
                        "is_churned": False,
                    }
                )

        
                if rng.random() < 0.15:
                    records.append(
                        {
                            "player_id": pid,
                            "event_type": "achievement",
                            "game": game,
                            "platform": platform,
                            "region": region,
                            "player_type": ptype,
                            "timestamp": ts + timedelta(minutes=int(duration * 0.6)),
                            "session_duration_mins": None,
                            "amount_usd": None,
                            "is_churned": False,
                        }
                    )

      
                if rng.random() < spend_prob[ptype]:
                    lo, hi = spend_amount[ptype]
                    amount = round(rng.uniform(lo, hi), 2)
                    records.append(
                        {
                            "player_id": pid,
                            "event_type": "purchase",
                            "game": game,
                            "platform": platform,
                            "region": region,
                            "player_type": ptype,
                            "timestamp": ts + timedelta(minutes=int(duration * 0.3)),
                            "session_duration_mins": None,
                            "amount_usd": amount,
                            "is_churned": False,
                        }
                    )

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    out = RAW_DIR / "player_events.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} events for {n_players} players → {out}\n")
    return df



def run_ingestion(
    fetch_apis: bool = True,
    n_players: int = 500,
    n_days: int = 90,
):
 
    print("=" * 55)
    print("  Player Engagement Analytics — Stage 1: Ingest")
    print("=" * 55 + "\n")

    results = {}

    if fetch_apis:
        results["rawg"] = fetch_rawg_games(RAWG_GAME_IDS)
        results["steamspy"] = fetch_steamspy_stats(STEAM_APP_IDS)
    else:
        print("Skipping live API calls (fetch_apis=False)\n")

    results["events"] = simulate_player_events(n_players=n_players, n_days=n_days)

    print("─" * 55)
    print("Ingestion complete. Summary:")
    for k, df in results.items():
        print(f"  {k:12s} → {len(df):>6,} rows, {len(df.columns)} columns")
    print("─" * 55)
    print(f"\nAll files saved to: {RAW_DIR}\n")
    return results





def generate_static_game_metadata() -> pd.DataFrame:
    
    data = [
        {"game_id": 1, "slug": "grand-theft-auto-v", "name": "Grand Theft Auto V", "released": "2013-09-17", "rating": 4.47, "ratings_count": 6823, "metacritic": 97, "playtime_avg_hrs": 73, "genres": "Action, Adventure", "platforms": "PC, PS5, Xbox Series X", "esrb_rating": "Mature"},
        {"game_id": 2, "slug": "the-witcher-3-wild-hunt", "name": "The Witcher 3: Wild Hunt", "released": "2015-05-18", "rating": 4.67, "ratings_count": 5921, "metacritic": 93, "playtime_avg_hrs": 102, "genres": "RPG, Adventure", "platforms": "PC, PS5, Xbox Series X, Nintendo Switch", "esrb_rating": "Mature"},
        {"game_id": 3, "slug": "cyberpunk-2077", "name": "Cyberpunk 2077", "released": "2020-12-10", "rating": 4.12, "ratings_count": 4102, "metacritic": 86, "playtime_avg_hrs": 61, "genres": "RPG, Action", "platforms": "PC, PS5, Xbox Series X", "esrb_rating": "Mature"},
        {"game_id": 4, "slug": "elden-ring", "name": "Elden Ring", "released": "2022-02-25", "rating": 4.65, "ratings_count": 3841, "metacritic": 96, "playtime_avg_hrs": 58, "genres": "RPG, Action", "platforms": "PC, PS5, Xbox Series X", "esrb_rating": "Mature"},
        {"game_id": 5, "slug": "red-dead-redemption-2", "name": "Red Dead Redemption 2", "released": "2019-11-05", "rating": 4.59, "ratings_count": 4512, "metacritic": 97, "playtime_avg_hrs": 75, "genres": "Action, Adventure", "platforms": "PC, PS5, Xbox Series X", "esrb_rating": "Mature"},
        {"game_id": 6, "slug": "hades", "name": "Hades", "released": "2020-09-17", "rating": 4.54, "ratings_count": 2901, "metacritic": 93, "playtime_avg_hrs": 29, "genres": "Action, Indie, RPG", "platforms": "PC, Nintendo Switch", "esrb_rating": "Teen"},
        {"game_id": 7, "slug": "stardew-valley", "name": "Stardew Valley", "released": "2016-02-26", "rating": 4.56, "ratings_count": 3712, "metacritic": 89, "playtime_avg_hrs": 58, "genres": "Simulation, RPG, Indie", "platforms": "PC, PS5, Xbox Series X, Nintendo Switch", "esrb_rating": "Everyone"},
        {"game_id": 8, "slug": "among-us", "name": "Among Us", "released": "2018-11-16", "rating": 3.84, "ratings_count": 2145, "metacritic": 85, "playtime_avg_hrs": 12, "genres": "Casual, Indie", "platforms": "PC, Nintendo Switch", "esrb_rating": "Everyone 10+"},
        {"game_id": 9, "slug": "counter-strike-global-offensive", "name": "Counter-Strike 2", "released": "2012-08-21", "rating": 3.97, "ratings_count": 5301, "metacritic": 83, "playtime_avg_hrs": 324, "genres": "Action, Shooter", "platforms": "PC", "esrb_rating": "Mature"},
        {"game_id": 10, "slug": "apex-legends", "name": "Apex Legends", "released": "2019-02-04", "rating": 4.13, "ratings_count": 3201, "metacritic": 88, "playtime_avg_hrs": 98, "genres": "Action, Shooter", "platforms": "PC, PS5, Xbox Series X", "esrb_rating": "Teen"},
    ]
    df = pd.DataFrame(data)
    out = RAW_DIR / "rawg_games.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} games (static) → {out}")
    return df


def generate_static_steamspy_stats() -> pd.DataFrame:
   
    data = [
        {"app_id": 271590, "name": "Grand Theft Auto V", "owners_estimate": "50,000,000 .. 100,000,000", "players_2weeks": 85000, "players_forever": 3200000, "average_2weeks_mins": 312, "median_2weeks_mins": 186, "average_forever_mins": 4380, "median_forever_mins": 1560, "positive_reviews": 1042000, "negative_reviews": 56000, "price_usd": 29.99},
        {"app_id": 292030, "name": "The Witcher 3: Wild Hunt", "owners_estimate": "20,000,000 .. 50,000,000", "players_2weeks": 42000, "players_forever": 1800000, "average_2weeks_mins": 480, "median_2weeks_mins": 320, "average_forever_mins": 6120, "median_forever_mins": 3240, "positive_reviews": 412000, "negative_reviews": 8500, "price_usd": 39.99},
        {"app_id": 1091500, "name": "Cyberpunk 2077", "owners_estimate": "20,000,000 .. 50,000,000", "players_2weeks": 55000, "players_forever": 2100000, "average_2weeks_mins": 380, "median_2weeks_mins": 240, "average_forever_mins": 3660, "median_forever_mins": 1980, "positive_reviews": 390000, "negative_reviews": 62000, "price_usd": 59.99},
        {"app_id": 1245620, "name": "Elden Ring", "owners_estimate": "10,000,000 .. 20,000,000", "players_2weeks": 38000, "players_forever": 1400000, "average_2weeks_mins": 360, "median_2weeks_mins": 220, "average_forever_mins": 3480, "median_forever_mins": 1800, "positive_reviews": 375000, "negative_reviews": 21000, "price_usd": 59.99},
        {"app_id": 1145360, "name": "Hades", "owners_estimate": "5,000,000 .. 10,000,000", "players_2weeks": 18000, "players_forever": 820000, "average_2weeks_mins": 240, "median_2weeks_mins": 160, "average_forever_mins": 1740, "median_forever_mins": 900, "positive_reviews": 175000, "negative_reviews": 3200, "price_usd": 24.99},
        {"app_id": 413150, "name": "Stardew Valley", "owners_estimate": "10,000,000 .. 20,000,000", "players_2weeks": 28000, "players_forever": 1200000, "average_2weeks_mins": 420, "median_2weeks_mins": 280, "average_forever_mins": 3480, "median_forever_mins": 1800, "positive_reviews": 410000, "negative_reviews": 6200, "price_usd": 14.99},
        {"app_id": 730, "name": "Counter-Strike 2", "owners_estimate": "100,000,000 ..", "players_2weeks": 950000, "players_forever": 8200000, "average_2weeks_mins": 780, "median_2weeks_mins": 480, "average_forever_mins": 19440, "median_forever_mins": 8640, "positive_reviews": 2100000, "negative_reviews": 580000, "price_usd": 0.00},
        {"app_id": 570, "name": "Dota 2", "owners_estimate": "100,000,000 ..", "players_2weeks": 620000, "players_forever": 5800000, "average_2weeks_mins": 900, "median_2weeks_mins": 600, "average_forever_mins": 23760, "median_forever_mins": 10800, "positive_reviews": 1400000, "negative_reviews": 350000, "price_usd": 0.00},
        {"app_id": 1172620, "name": "Apex Legends", "owners_estimate": "50,000,000 .. 100,000,000", "players_2weeks": 320000, "players_forever": 4200000, "average_2weeks_mins": 480, "median_2weeks_mins": 300, "average_forever_mins": 5760, "median_forever_mins": 2520, "positive_reviews": 580000, "negative_reviews": 95000, "price_usd": 0.00},
        {"app_id": 578080, "name": "PUBG: Battlegrounds", "owners_estimate": "50,000,000 .. 100,000,000", "players_2weeks": 280000, "players_forever": 3600000, "average_2weeks_mins": 540, "median_2weeks_mins": 360, "average_forever_mins": 6480, "median_forever_mins": 2880, "positive_reviews": 620000, "negative_reviews": 320000, "price_usd": 0.00},
    ]
    df = pd.DataFrame(data)
    out = RAW_DIR / "steamspy_stats.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} games (static) → {out}")
    return df


if __name__ == "__main__":
    run_ingestion(fetch_apis=True, n_players=500, n_days=90)
