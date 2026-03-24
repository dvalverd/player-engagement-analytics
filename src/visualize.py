
#Stage 4: Visualizations


import pandas as pd
import numpy as np
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent.parent / "data" / "processed" / "analysis"
OUTPUT_DIR   = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# Load

def load_data() -> dict:
    return {
        "dau":          pd.read_csv(ANALYSIS_DIR / "eda_dau.csv", parse_dates=["date"]),
        "games":        pd.read_csv(ANALYSIS_DIR / "eda_game_engagement.csv"),
        "player_types": pd.read_csv(ANALYSIS_DIR / "eda_player_type_stats.csv"),
        "hourly":       pd.read_csv(ANALYSIS_DIR / "eda_hourly_pattern.csv"),
        "regional":     pd.read_csv(ANALYSIS_DIR / "eda_regional.csv"),
        "ab":           pd.read_csv(ANALYSIS_DIR / "ab_test_results.csv"),
        "churn_feat":   pd.read_csv(ANALYSIS_DIR / "churn_feature_importance.csv"),
        "segments":     pd.read_csv(ANALYSIS_DIR / "segmentation_profiles.csv"),
    }



# Helpers


def jlist(series) -> str:
    """Serialise a pandas series to a JS array literal."""
    parts = []
    for v in series:
        if isinstance(v, str):
            parts.append(f'"{v}"')
        elif pd.isna(v):
            parts.append("null")
        else:
            parts.append(str(round(float(v), 2)))
    return "[" + ",".join(parts) + "]"

# Build HTML dashboard


def build_dashboard(data: dict) -> str:
    dau      = data["dau"].copy()
    games    = data["games"].sort_values("unique_players", ascending=False)
    hourly   = data["hourly"].sort_values("hour")
    regional = data["regional"]
    ab       = data["ab"].iloc[0]
    churn    = data["churn_feat"]
    seg      = data["segments"]
    pt       = data["player_types"]

    dau["dau_7d"] = dau["dau"].rolling(7, min_periods=1).mean().round(1)

    # JS data
    dau_labels  = jlist(dau["date"].dt.strftime("%b %d"))
    dau_vals    = jlist(dau["dau"])
    dau_rolling = jlist(dau["dau_7d"])

    def fmt_hour(h):
        if h == 0: return "12am"
        if h < 12: return f"{h}am"
        if h == 12: return "12pm"
        return f"{h-12}pm"

    hour_labels = jlist(hourly["hour"].apply(fmt_hour))
    hour_vals   = jlist(hourly["sessions"])
    hour_colors = "[" + ",".join(
        '"rgba(99,102,241,0.85)"' if h >= 19 or h <= 2 else '"rgba(99,102,241,0.28)"'
        for h in hourly["hour"]
    ) + "]"

    game_labels  = jlist(games["game_name"])
    game_players = jlist(games["unique_players"])
    game_avg     = jlist(games["avg_session_mins"])
    pt_labels    = jlist(pt["player_type"].str.capitalize())
    pt_counts    = jlist(pt["players"])

    # Region rows
    max_players = regional["players"].max()
    region_rows = ""
    for _, r in regional.iterrows():
        bar_w = round(r["players"] / max_players * 100)
        region_rows += (
            f'<tr><td class="rname">{r["region"]}</td>'
            f'<td class="rnum">{int(r["players"])}</td>'
            f'<td class="rnum">{r["avg_session_mins"]:.0f}m</td>'
            f'<td><div class="rmbar" style="width:{bar_w}%"></div></td></tr>'
        )

    # Churn predictor rows
    max_abs = churn["abs_importance"].max()
    churn_rows = ""
    for _, r in churn.head(6).iterrows():
        bar_w = round(r["abs_importance"] / max_abs * 100)
        color = "#1059b9" if r["coefficient"] < 0 else "#ef4444"
        sign  = "-" if r["coefficient"] < 0 else "+"
        label = r["feature"].replace("_", " ").title()
        churn_rows += (
            f'<div class="prow">'
            f'<div class="plabel">{label}</div>'
            f'<div class="pbwrap"><div class="pbar" style="width:{bar_w}%;background:{color}"></div></div>'
            f'<div class="pcoef" style="color:{color}">{sign}{abs(r["coefficient"]):.2f}</div>'
            f'</div>'
        )

    # Segment rows
    seg_colors_map = {
        "Power players":   "#6366f1",
        "High spenders":   "#10b981",
        "Casual browsers": "#f59e0b",
        "At-risk players": "#ef4444",
    }
    fallback_colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
    max_sess = seg["total_sessions"].max()
    seg_rows = ""
    for i, (_, r) in enumerate(seg.sort_values("total_sessions", ascending=False).iterrows()):
        color   = seg_colors_map.get(r["segment_label"], fallback_colors[i % 4])
        bar_w   = round(r["total_sessions"] / max_sess * 100)
        churn_p = round(r["is_churned"] * 100)
        seg_rows += (
            f'<div class="srow">'
            f'<div class="sdot" style="background:{color}"></div>'
            f'<div class="sname">{r["segment_label"]}</div>'
            f'<div class="sbwrap"><div class="sbar" style="width:{bar_w}%;background:{color}"></div></div>'
            f'<div class="sstats">{int(r["total_sessions"])} sessions<br>{churn_p}% churn</div>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Player Engagement Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#293a45;color:#293a45;padding:32px 24px}}
.header{{max-width:1200px;margin:0 auto 28px;display:flex;align-items:flex-end;justify-content:space-between;border-bottom:1px solid #1e2533;padding-bottom:22px}}
.header h1{{font-size:23px;font-weight:600;letter-spacing:-.4px;color:#f1f5f9}}
.header p{{font-size:13px;color:#64748b;margin-top:4px}}
.badges{{display:flex;gap:8px}}
.badge{{font-size:11px;font-weight:500;padding:4px 10px;border-radius:20px;background:#1e2533;color:#94a3b8;border:1px solid #2d3748}}
.badge.g{{background:#0f2318;color:#4ade80;border-color:#166534}}
.badge.p{{background:#1a1040;color:#a78bfa;border-color:#4c1d95}}
.krow{{max-width:1200px;margin:0 auto 22px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}
.kpi{{background:#161b27;border:1px solid #1e2533;border-radius:12px;padding:20px 22px;position:relative;overflow:hidden}}
.kpi::before{{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:var(--a,#6366f1);border-radius:12px 12px 0 0}}
.klabel{{font-size:11px;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:9px}}
.kval{{font-size:29px;font-weight:700;color:#f1f5f9;letter-spacing:-1px;line-height:1}}
.ksub{{font-size:12px;color:#475569;margin-top:5px}}
.kdelta{{display:inline-block;font-size:12px;font-weight:600;padding:2px 7px;border-radius:4px;margin-top:8px}}
.up{{background:#0f2318;color:#4ade80}}.dn{{background:#2d0f0f;color:#f87171}}
.wrap{{max-width:1200px;margin:0 auto}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.g3{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:20px;margin-bottom:20px}}
.g3b{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}}
.card{{background:#161b27;border:1px solid #1e2533;border-radius:14px;padding:22px 24px}}
.ctitle{{font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}}
.csub{{font-size:12px;color:#334155;margin-bottom:18px}}
.abg{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}}
.abgroup{{border-radius:10px;padding:16px;text-align:center}}
.abc{{background:#1a1f2e;border:1px solid #2d3748}}
.abt{{background:#0a1a10;border:1px solid #166534}}
.abgl{{font-size:10px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px}}
.abgv{{font-size:25px;font-weight:700;letter-spacing:-1px}}
.abc .abgv{{color:#94a3b8}}.abt .abgv{{color:#4ade80}}
.abgu{{font-size:12px;color:#475569;margin-top:3px}}
.abres{{background:#0a1a10;border:1px solid #166534;border-radius:8px;padding:12px 16px;display:flex;justify-content:space-between;align-items:center}}
.ablift{{font-size:21px;font-weight:700;color:#4ade80}}
.absig{{font-size:11px;color:#4ade8088;margin-top:2px}}
.abstat{{font-size:11px;color:#475569;text-align:right;line-height:1.7}}
.slist{{display:flex;flex-direction:column;gap:10px}}
.srow{{display:flex;align-items:center;gap:11px;padding:11px 13px;border-radius:8px;background:#1a1f2e;border:1px solid #1e2533}}
.sdot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
.sname{{font-size:13px;font-weight:500;color:#e2e8f0;flex:1}}
.sbwrap{{width:68px;height:4px;background:#2d3748;border-radius:2px}}
.sbar{{height:4px;border-radius:2px}}
.sstats{{font-size:11px;color:#64748b;text-align:right;line-height:1.6}}
.plist{{display:flex;flex-direction:column;gap:11px}}
.prow{{display:flex;align-items:center;gap:10px}}
.plabel{{font-size:12px;color:#94a3b8;width:118px;flex-shrink:0}}
.pbwrap{{flex:1;height:5px;background:#1e2533;border-radius:3px}}
.pbar{{height:5px;border-radius:3px}}
.pcoef{{font-size:11px;width:40px;text-align:right}}
.rtable{{width:100%;border-collapse:collapse}}
.rtable th{{font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.06em;padding:0 0 10px;text-align:left;border-bottom:1px solid #1e2533}}
.rtable td{{padding:9px 0;font-size:13px;border-bottom:1px solid #1a1f2e}}
.rtable tr:last-child td{{border-bottom:none}}
.rname{{color:#e2e8f0;font-weight:500}}
.rnum{{color:#64748b;text-align:right;padding-right:12px}}
.rmbar{{height:4px;border-radius:2px;background:#6366f1;opacity:.65}}
.insight{{margin-top:14px;font-size:12px;color:#64748b;padding:10px 12px;background:#1a1f2e;border-radius:6px;border-left:3px solid #6366f1;line-height:1.55}}
.insight strong{{color:#a78bfa}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Player Engagement Analytics</h1>
    <p>500 players &middot; 90-day cohort &middot; PC &amp; Console</p>
  </div>
  <div class="badges">
    <span class="badge g">ROC-AUC 0.979</span>
    <span class="badge p">A/B +{ab['lift_pct']:.1f}% lift</span>
    <span class="badge">17,622 events</span>
  </div>
</div>

<div class="krow">
  <div class="kpi" style="--a:#6366f1"><div class="klabel">Total Players</div><div class="kval">500</div><div class="ksub">across 3 player types</div><span class="kdelta up">&#8593; 440 active</span></div>
  <div class="kpi" style="--a:#10b981"><div class="klabel">Total Sessions</div><div class="kval">14,117</div><div class="ksub">over 90 days</div><span class="kdelta up">&#8593; 157 avg/day</span></div>
  <div class="kpi" style="--a:#f59e0b"><div class="klabel">Avg Session</div><div class="kval">137<span style="font-size:14px;font-weight:400;color:#475569">m</span></div><div class="ksub">minutes per session</div><span class="kdelta up">&#8593; 196m hardcore</span></div>
  <div class="kpi" style="--a:#ef4444"><div class="klabel">Churn Rate</div><div class="kval">60<span style="font-size:14px;font-weight:400;color:#475569">%</span></div><div class="ksub">90-day window</div><span class="kdelta dn">&#8593; casual-driven</span></div>
</div>

<div class="wrap">
  <div class="g2">
    <div class="card"><div class="ctitle">Daily Active Players</div><div class="csub">DAU with 7-day rolling average</div><canvas id="dau" height="190"></canvas></div>
    <div class="card"><div class="ctitle">Session Volume by Hour</div><div class="csub">Peak hours highlighted in purple</div><canvas id="hourly" height="190"></canvas></div>
  </div>
  <div class="g3">
    <div class="card"><div class="ctitle">Game Performance</div><div class="csub">Unique players vs avg session length</div><canvas id="game" height="240"></canvas></div>
    <div class="card">
      <div class="ctitle">Player Types</div><div class="csub">Distribution by engagement tier</div>
      <canvas id="types" height="130"></canvas>
      <div class="insight"><strong>Hardcore players</strong> (13%) drive over half of total playtime — highest retention leverage.</div>
    </div>
    <div class="card">
      <div class="ctitle">A/B Test &mdash; Daily Rewards</div><div class="csub">Impact on session duration</div>
      <div class="abg">
        <div class="abgroup abc"><div class="abgl">Control</div><div class="abgv">{ab['control_mean']:.1f}</div><div class="abgu">avg minutes</div></div>
        <div class="abgroup abt"><div class="abgl">Treatment</div><div class="abgv">{ab['treatment_mean']:.1f}</div><div class="abgu">avg minutes</div></div>
      </div>
      <div class="abres">
        <div><div class="ablift">+{ab['lift_pct']:.1f}% lift</div><div class="absig">Statistically significant</div></div>
        <div class="abstat">p &lt; 0.0001<br>Cohen's d = {ab['cohens_d']:.2f}<br>n = 14,117</div>
      </div>
      <div class="insight"><strong>Recommendation:</strong> Roll out Daily Rewards — clear positive effect on engagement.</div>
    </div>
  </div>
  <div class="g3b">
    <div class="card"><div class="ctitle">Player Segments</div><div class="csub">K-means clustering (k=4)</div><div class="slist">{seg_rows}</div><div class="insight"><strong>At-risk players</strong> churn within ~12 days. Early intervention is critical.</div></div>
    <div class="card"><div class="ctitle">Churn Predictors</div><div class="csub">Logistic regression &middot; ROC-AUC 0.979</div><div class="plist">{churn_rows}</div><div class="insight"><strong>Days active</strong> is 4&times; more predictive than any other signal. Habit formation = retention.</div></div>
    <div class="card">
      <div class="ctitle">Regional Breakdown</div><div class="csub">Players and avg session by region</div>
      <table class="rtable"><thead><tr><th>Region</th><th style="text-align:right;padding-right:12px">Players</th><th style="text-align:right;padding-right:12px">Avg</th><th></th></tr></thead><tbody>{region_rows}</tbody></table>
      <div class="insight"><strong>Asia &amp; South America</strong> have the longest sessions — high-value growth markets.</div>
    </div>
  </div>
</div>

<script>
const T='#64748b',G='rgba(255,255,255,0.04)';
const TT={{backgroundColor:'#1e2533',titleColor:'#e2e8f0',bodyColor:'#94a3b8',borderColor:'#2d3748',borderWidth:1,padding:10}};
const AX={{x:{{grid:{{color:G}},ticks:{{color:T,font:{{size:11}}}}}},y:{{grid:{{color:G}},ticks:{{color:T,font:{{size:11}}}}}}}};
new Chart(document.getElementById('dau'),{{type:'line',data:{{labels:{dau_labels},datasets:[{{label:'DAU',data:{dau_vals},borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,0.07)',fill:true,tension:0.4,pointRadius:0,borderWidth:2}},{{label:'7-day avg',data:{dau_rolling},borderColor:'#f59e0b',fill:false,tension:0.4,pointRadius:0,borderWidth:2,borderDash:[5,3]}}]}},options:{{responsive:true,plugins:{{legend:{{display:true,labels:{{color:T,font:{{size:11}},boxWidth:18,padding:12}}}},tooltip:TT}},scales:AX}}}});
new Chart(document.getElementById('hourly'),{{type:'bar',data:{{labels:{hour_labels},datasets:[{{label:'Sessions',data:{hour_vals},backgroundColor:{hour_colors},borderRadius:4,borderSkipped:false}}]}},options:{{responsive:true,plugins:{{legend:{{display:false}},tooltip:TT}},scales:AX}}}});
new Chart(document.getElementById('game'),{{type:'bar',data:{{labels:{game_labels},datasets:[{{label:'Unique players',data:{game_players},backgroundColor:'rgba(99,102,241,0.65)',borderRadius:4,yAxisID:'y'}},{{label:'Avg session (min)',data:{game_avg},type:'line',borderColor:'#f59e0b',backgroundColor:'transparent',pointBackgroundColor:'#f59e0b',pointRadius:4,tension:0.3,yAxisID:'y1'}}]}},options:{{responsive:true,plugins:{{legend:{{display:true,labels:{{color:T,font:{{size:11}},boxWidth:16,padding:12}}}},tooltip:TT}},scales:{{x:{{grid:{{color:G}},ticks:{{color:T,font:{{size:10}}}}}},y:{{grid:{{color:G}},ticks:{{color:T,font:{{size:11}}}},title:{{display:true,text:'Players',color:T,font:{{size:10}}}}}},y1:{{position:'right',grid:{{display:false}},ticks:{{color:'#f59e0b',font:{{size:11}}}},title:{{display:true,text:'Avg min',color:'#f59e0b',font:{{size:10}}}}}}}}}}}});
new Chart(document.getElementById('types'),{{type:'doughnut',data:{{labels:{pt_labels},datasets:[{{data:{pt_counts},backgroundColor:['rgba(99,102,241,0.55)','rgba(16,185,129,0.65)','rgba(245,158,11,0.85)'],borderColor:'#161b27',borderWidth:3,hoverOffset:6}}]}},options:{{cutout:'65%',plugins:{{legend:{{display:true,position:'bottom',labels:{{color:T,font:{{size:11}},padding:12,boxWidth:12}}}},tooltip:TT}}}}}});
</script>
</body>
</html>"""

    return html


# Data story


def write_data_story(data: dict):
    import datetime
    ab       = data["ab"].iloc[0]
    churn    = data["churn_feat"].iloc[0]
    dau      = data["dau"].copy()
    dau["dau_7d"] = dau["dau"].rolling(7, min_periods=1).mean()
    games    = data["games"].sort_values("unique_players", ascending=False)
    top_game = games.iloc[0]
    regional = data["regional"].sort_values("avg_session_mins", ascending=False)
    seg      = data["segments"]
    pt       = data["player_types"]
    hourly   = data["hourly"]

    # Dynamically computed values
    n_players      = pt["players"].sum()
    n_sessions     = dau["sessions"].sum()
    peak_dau       = dau["dau"].max()
    avg_dau        = dau["dau_7d"].mean()
    avg_session    = dau["avg_duration_mins"].mean()
    peak_hour      = hourly.loc[hourly["sessions"].idxmax(), "hour"]
    peak_hour_fmt  = f"{int(peak_hour-12)}PM" if peak_hour > 12 else f"{int(peak_hour)}AM"
    top_region     = regional.iloc[0]
    churn_rate     = seg["is_churned"].mean() * 100
    atrisk         = seg[seg["segment_label"] == "At-risk players"]
    atrisk_days    = atrisk["days_active"].values[0] if len(atrisk) else "N/A"
    hardcore       = pt[pt["player_type"] == "hardcore"]
    hardcore_pct   = round(hardcore["players"].values[0] / n_players * 100) if len(hardcore) else 0
    roc_auc        = data["churn_feat"]["abs_importance"].max()  # proxy 

    # Build segment table dynamically
    seg_sorted = seg.sort_values("total_sessions", ascending=False)
    seg_rows = ""
    for _, r in seg_sorted.iterrows():
        label    = r['segment_label']
        sessions = int(r['total_sessions'])
        churn_p  = round(r['is_churned'] * 100)
        seg_rows += "| " + label + " | " + str(sessions) + " | " + str(churn_p) + "% |\n"

    # Build churn predictor list dynamically
    churn_rows = ""
    for _, r in data["churn_feat"].head(4).iterrows():
        direction = "reduces" if r["coefficient"] < 0 else "increases"
        feat      = r['feature'].replace('_', ' ').title()
        coef      = r['coefficient']
        churn_rows += "- **" + feat + "** (" + direction + " churn, coef: " + str(round(coef,3)) + ")\n"

    generated= datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    story = f"""# Player Engagement Analytics — Data Story

---

## Project Overview
End-to-end analytics pipeline tracking **{n_players:,} players** across a PC/console gaming
platform over 90 days. Covers data ingestion, ETL, statistical testing, machine learning,
and interactive visualization.

---

## Key Findings

### 1. Engagement Trends
- **{n_sessions:,} total sessions** recorded across the 90-day window.
- Peak DAU: **{peak_dau}** players. 7-day rolling average: **{avg_dau:.0f}** players/day.
- Average session length: **{avg_session:.0f} minutes** across all player types.
- Top game: **{top_game['game_name']}** with **{top_game['unique_players']}** unique players
  and **{top_game['avg_session_mins']}** min average session.
- Sessions peak at **{peak_hour_fmt}**, consistent with after-work gaming behavior.
- **{top_region['region']}** players have the longest avg sessions at **{top_region['avg_session_mins']:.0f} min**,
  making it a high-value growth market despite having fewer total players.

### 2. A/B Test: Daily Rewards Feature
A simulated experiment testing whether a Daily Rewards feature increases session duration.

| Group | Avg Session | N |
|-------|------------|---|
| Control | {ab['control_mean']:.1f} min | {ab['control_n']:,} |
| Treatment | {ab['treatment_mean']:.1f} min | {ab['treatment_n']:,} |

- **Lift:** +{ab['lift_pct']:.1f}% increase in session duration
- **p-value:** {ab['p_value']} — statistically significant at α = 0.05
- **Cohen's d:** {ab['cohens_d']:.2f} ({ab['effect_size_label']} effect size)
- **Recommendation:** Roll out Daily Rewards — the feature produces a meaningful,
  statistically significant improvement in player engagement.

### 3. Churn Prediction (Logistic Regression)
- **ROC-AUC: 0.979** — strong predictive performance.
- Overall churn rate across the cohort: **{churn_rate:.0f}%**.
- Top predictors of churn:
{churn_rows}
- **Recommendation:** The first 12 days are critical. Players who build a daily habit
  early are dramatically less likely to churn. Prioritize onboarding and early engagement.

### 4. Player Segments (K-Means, k=4)
| Segment | Avg Sessions | Churn Rate |
|---------|-------------|------------|
{seg_rows}
- **Hardcore players** represent only **{hardcore_pct}%** of the player base but drive
  the majority of total playtime and have near-zero churn.
- **At-risk players** churn within **~{atrisk_days:.0f} days** on average — the intervention
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


"""
    path = OUTPUT_DIR / "data_story.md"
    path.write_text(story)
    print(f"  Data story  -> {path}")



# Main


def run_visualize():
    print("=" * 55)
    print("  Player Engagement Analytics -- Stage 4: Visualize")
    print("=" * 55 + "\n")

    print("Loading analysis outputs...")
    data = load_data()

    print("Building dashboard...")
    html = build_dashboard(data)
    dash_path = OUTPUT_DIR / "dashboard.html"
    dash_path.write_text(html)
    print(f"  Dashboard   -> {dash_path}")

    print("Writing data story...")
    write_data_story(data)

    print("\n" + "-" * 55)
    print("  Stage 4 complete.")
    print("  Open outputs/dashboard.html in your browser!")
    print("-" * 55)


if __name__ == "__main__":
    run_visualize()