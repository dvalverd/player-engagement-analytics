# Player Engagement Analytics — Data Story

## Project Overview
End-to-end analytics pipeline tracking **440 players** across a PC/console gaming
platform over 90 days. Covers data ingestion, ETL, statistical testing, machine learning,
and interactive visualization.

---

## Key Findings

### 1. Engagement Trends
- **14,117 total sessions** recorded across the 90-day window.
- Peak DAU: **160** players. 7-day rolling average: **93** players/day.
- Average session length: **137 minutes** across all player types.
- Top game: **Elden Ring** with **52** unique players
  and **141.6** min average session.
- Sessions peak at **6PM**, consistent with after-work gaming behavior.
- **South America** players have the longest avg sessions at **149 min**,
  making it a high-value growth market despite having fewer total players.

### 2. A/B Test: Daily Rewards Feature
A simulated experiment testing whether a Daily Rewards feature increases session duration.

| Group | Avg Session | N |
|-------|------------|---|
| Control | 135.5 min | 7,009 |
| Treatment | 156.8 min | 7,108 |

- **Lift:** +15.7% increase in session duration
- **p-value:** 0.0 — statistically significant at α = 0.05
- **Cohen's d:** 0.26 (small effect size)
- **Recommendation:** Roll out Daily Rewards — the feature produces a meaningful,
  statistically significant improvement in player engagement.

### 3. Churn Prediction (Logistic Regression)
- **ROC-AUC: 0.979** — strong predictive performance.
- Overall churn rate across the cohort: **47%**.
- Top predictors of churn:
- **Days Active** (reduces churn, coef: -3.01)
- **Total Sessions** (reduces churn, coef: -0.782)
- **Total Purchases** (reduces churn, coef: -0.676)
- **Player Type Enc** (reduces churn, coef: -0.665)

- **Recommendation:** The first 12 days are critical. Players who build a daily habit
  early are dramatically less likely to churn. Prioritize onboarding and early engagement.

### 4. Player Segments (K-Means, k=4)
| Segment | Avg Sessions | Churn Rate |
|---------|-------------|------------|
| Power players | 152 | 0% |
| Casual browsers | 115 | 30% |
| Casual browsers | 40 | 59% |
| At-risk players | 6 | 100% |

- **Hardcore players** represent only **15%** of the player base but drive
  the majority of total playtime and have near-zero churn.
- **At-risk players** churn within **~12 days** on average — the intervention
  window is narrow.
- **Recommendation:** Focus retention efforts on casual browsers before they become
  at-risk. Reward power players with loyalty perks to sustain their engagement.

---

## Tech Stack
| Layer | Tools |
| Ingestion | Python, Requests, RAWG API, SteamSpy API |
| ETL | Pandas, DuckDB (star schema) |
| Analysis | NumPy, SciPy, Scikit-learn |
| Visualization | Chart.js, HTML/CSS |


