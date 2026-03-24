#Stage 3 Analysis

"""
  1. EDA              — engagement trends, retention, playtime distributions
  2. A/B test sim     — hypothesis testing on a feature experiment
  3. Churn model      — logistic regression churn prediction
  4. Segmentation     — K-means player clustering

"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROCESSED_DIR / "warehouse.duckdb"


def get_con():
    return duckdb.connect(str(DB_PATH), read_only=True)



# Data Analysis


def run_eda() -> dict[str, pd.DataFrame]:
    print("\n── EDA")
    con = get_con()
    results = {}

    # Daily active players
    results["dau"] = con.execute("""
        SELECT date,
               COUNT(DISTINCT player_key) AS dau,
               COUNT(*) AS sessions,
               ROUND(AVG(duration_mins), 1) AS avg_duration_mins,
               ROUND(SUM(duration_mins) / 60.0, 1) AS total_hrs
        FROM fact_sessions
        GROUP BY date
        ORDER BY date
    """).fetchdf()

    # Engagement by game
    results["game_engagement"] = con.execute("""
        SELECT g.game_name,
               g.primary_genre,
               g.rawg_rating,
               COUNT(DISTINCT f.player_key) AS unique_players,
               COUNT(*) AS total_sessions,
               ROUND(AVG(f.duration_mins), 1) AS avg_session_mins,
               ROUND(SUM(f.duration_mins) / 60.0, 1) AS total_hrs_played
        FROM fact_sessions f
        JOIN dim_games g ON f.game_key = g.game_key
        GROUP BY g.game_name, g.primary_genre, g.rawg_rating
        ORDER BY unique_players DESC
    """).fetchdf()

    # Engagement by player type
    results["player_type_stats"] = con.execute("""
        SELECT player_type,
               COUNT(DISTINCT player_key) AS players,
               ROUND(AVG(duration_mins), 1) AS avg_session_mins,
               COUNT(*) AS total_sessions,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT player_key), 1) AS sessions_per_player
        FROM fact_sessions
        GROUP BY player_type
        ORDER BY avg_session_mins DESC
    """).fetchdf()

    # Sessions by hour of day
    results["hourly_pattern"] = con.execute("""
        SELECT hour,
               COUNT(*) AS sessions,
               ROUND(AVG(duration_mins), 1) AS avg_duration_mins
        FROM fact_sessions
        GROUP BY hour
        ORDER BY hour
    """).fetchdf()

    # Regional breakdown
    results["regional"] = con.execute("""
        SELECT region,
               COUNT(DISTINCT player_key) AS players,
               COUNT(*) AS sessions,
               ROUND(AVG(duration_mins), 1) AS avg_session_mins
        FROM fact_sessions
        GROUP BY region
        ORDER BY players DESC
    """).fetchdf()

    # Weekend vs weekday
    results["weekend_vs_weekday"] = con.execute("""
        SELECT is_weekend,
               COUNT(*) AS sessions,
               ROUND(AVG(duration_mins), 1) AS avg_duration_mins,
               COUNT(DISTINCT player_key) AS unique_players
        FROM fact_sessions
        GROUP BY is_weekend
    """).fetchdf()

    con.close()

    for k, df in results.items():
        path = ANALYSIS_DIR / f"eda_{k}.csv"
        df.to_csv(path, index=False)
        print(f"   {k:30s} → {len(df)} rows")

    print(f"\n  Top games by players:")
    print(results["game_engagement"][["game_name", "unique_players", "avg_session_mins"]].to_string(index=False))

    return results



# 2. A/B Test 


def run_ab_test() -> pd.DataFrame:
    
    print("\n── A/B Test ─")

    con = get_con()
    sessions = con.execute("""
        SELECT player_key, duration_mins, player_type
        FROM fact_sessions
    """).fetchdf()
    con.close()

    rng = np.random.default_rng(42)

    # Randomly assign players to groups (50/50 split)
    player_ids = sessions["player_key"].unique()
    treatment_ids = set(rng.choice(player_ids, size=len(player_ids) // 2, replace=False))
    sessions["group"] = sessions["player_key"].apply(
        lambda x: "treatment" if x in treatment_ids else "control"
    )

    # Treatment effect: Daily Rewards boosts session duration by ~12% with noise
    treatment_mask = sessions["group"] == "treatment"
    boost = rng.normal(loc=0.12, scale=0.04, size=treatment_mask.sum())
    sessions.loc[treatment_mask, "duration_mins"] = (
        sessions.loc[treatment_mask, "duration_mins"] * (1 + boost)
    ).round(1)

    control   = sessions[sessions["group"] == "control"]["duration_mins"]
    treatment = sessions[sessions["group"] == "treatment"]["duration_mins"]

    t_stat, p_value = stats.ttest_ind(treatment, control)

    # Cohen's d effect size
    pooled_std = np.sqrt((control.std() ** 2 + treatment.std() ** 2) / 2)
    cohens_d = (treatment.mean() - control.mean()) / pooled_std

    results = pd.DataFrame([{
        "metric":                  "session_duration_mins",
        "control_mean":            round(control.mean(), 2),
        "treatment_mean":          round(treatment.mean(), 2),
        "lift_pct":                round((treatment.mean() - control.mean()) / control.mean() * 100, 2),
        "control_n":               len(control),
        "treatment_n":             len(treatment),
        "t_statistic":             round(t_stat, 4),
        "p_value":                 round(p_value, 6),
        "statistically_significant": p_value < 0.05,
        "cohens_d":                round(cohens_d, 4),
        "effect_size_label":       "small" if abs(cohens_d) < 0.3 else "medium" if abs(cohens_d) < 0.5 else "large",
    }])

    results.to_csv(ANALYSIS_DIR / "ab_test_results.csv", index=False)

    print(f"  Control mean:   {results['control_mean'].iloc[0]} mins")
    print(f"  Treatment mean: {results['treatment_mean'].iloc[0]} mins")
    print(f"  Lift:           +{results['lift_pct'].iloc[0]}%")
    print(f"  p-value:        {results['p_value'].iloc[0]}")
    print(f"  Significant:    {results['statistically_significant'].iloc[0]}")
    print(f"  Cohen's d:      {results['cohens_d'].iloc[0]} ({results['effect_size_label'].iloc[0]})")

    # Also save group-level summary for visualization
    group_summary = sessions.groupby("group")["duration_mins"].agg(
        ["mean", "std", "count", "median"]
    ).reset_index()
    group_summary.to_csv(ANALYSIS_DIR / "ab_test_groups.csv", index=False)

    return results



#  Churn Prediction Model


def run_churn_model() -> dict:
    """
    Predict churn using logistic regression on player behavior features.
    Features: total sessions, avg session length, total spend, days active, player type.
    """
    print("\n Churn Model ")

    dim_players = pd.read_csv(PROCESSED_DIR / "dim_players.csv")

    features = [
        "total_sessions",
        "avg_session_mins",
        "total_playtime_hrs",
        "total_purchases",
        "total_spent_usd",
        "days_active",
    ]

    # Encode player type
    dim_players["player_type_enc"] = dim_players["player_type"].map(
        {"casual": 0, "regular": 1, "hardcore": 2}
    )
    features.append("player_type_enc")

    X = dim_players[features].fillna(0)
    y = dim_players["is_churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"  ROC-AUC: {auc:.3f}")
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=["retained", "churned"]))

    # Feature importance
    importance = pd.DataFrame({
        "feature":    features,
        "coefficient": model.coef_[0].round(4),
        "abs_importance": np.abs(model.coef_[0]).round(4),
    }).sort_values("abs_importance", ascending=False)
    importance.to_csv(ANALYSIS_DIR / "churn_feature_importance.csv", index=False)

    # Churn probability per player
    all_probs = model.predict_proba(scaler.transform(X.fillna(0)))[:, 1]
    dim_players["churn_probability"] = all_probs.round(4)
    dim_players["churn_risk"] = pd.cut(
        dim_players["churn_probability"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"]
    )
    dim_players[["player_id", "player_type", "is_churned", "churn_probability", "churn_risk"]].to_csv(
        ANALYSIS_DIR / "churn_scores.csv", index=False
    )

    metrics = pd.DataFrame([{"roc_auc": round(auc, 4), "n_train": len(X_train), "n_test": len(X_test)}])
    metrics.to_csv(ANALYSIS_DIR / "churn_metrics.csv", index=False)

    print(f"\n  Top churn predictors:")
    print(importance.head(4)[["feature", "coefficient"]].to_string(index=False))

    return {"auc": auc, "importance": importance}



#  (K-Means)


def run_segmentation() -> pd.DataFrame:

    print("\n── Segmentation ")

    dim_players = pd.read_csv(PROCESSED_DIR / "dim_players.csv")

    seg_features = [
        "total_sessions",
        "avg_session_mins",
        "total_playtime_hrs",
        "total_purchases",
        "total_spent_usd",
        "days_active",
    ]

    X = dim_players[seg_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k using inertia (elbow method)
    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append({"k": k, "inertia": round(km.inertia_, 2)})

    elbow_df = pd.DataFrame(inertias)
    elbow_df.to_csv(ANALYSIS_DIR / "segmentation_elbow.csv", index=False)

    # Fit with k=4 (good balance for player analytics)
    K = 4
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    dim_players["cluster"] = km.fit_predict(X_scaled)

    # Cluster profiles
    profile = dim_players.groupby("cluster")[seg_features + ["is_churned"]].agg({
        "total_sessions":    "mean",
        "avg_session_mins":  "mean",
        "total_playtime_hrs":"mean",
        "total_purchases":   "mean",
        "total_spent_usd":   "mean",
        "days_active":       "mean",
        "is_churned":        "mean",
    }).round(2).reset_index()

    # Label clusters meaningfully
    labels = {
        profile["total_spent_usd"].idxmax():       "High spenders",
        profile["total_sessions"].idxmax():        "Power players",
        profile["is_churned"].idxmax():            "At-risk players",
    }
    # Remaining cluster = casual browsers
    remaining = [i for i in range(K) if i not in labels]
    if remaining:
        labels[remaining[0]] = "Casual browsers"

    profile["segment_label"] = profile["cluster"].map(labels).fillna("Casual browsers")

    profile.to_csv(ANALYSIS_DIR / "segmentation_profiles.csv", index=False)

    # Save player-level cluster assignments
    dim_players["segment_label"] = dim_players["cluster"].map(labels).fillna("Casual browsers")
    dim_players[["player_id", "player_type", "cluster", "segment_label",
                 "total_sessions", "total_spent_usd", "is_churned"]].to_csv(
        ANALYSIS_DIR / "segmentation_players.csv", index=False
    )

    print(f"  Segments identified ({K} clusters):")
    print(profile[["cluster", "segment_label", "total_sessions",
                   "avg_session_mins", "total_spent_usd", "is_churned"]].to_string(index=False))

    return profile



# Main runner


def run_analysis():
    print("=" * 55)
    print("  Player Engagement Analytics — Stage 3: Analyze")
    print("=" * 55)

    eda_results = run_eda()
    ab_results  = run_ab_test()
    churn_results = run_churn_model()
    seg_results = run_segmentation()

    print("\n" + "─" * 55)
    print("  Stage 3 complete. All outputs in data/processed/analysis/")
    print("─" * 55)

    return eda_results, ab_results, churn_results, seg_results


if __name__ == "__main__":
    run_analysis()
