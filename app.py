import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from numpy.random import poisson

# --- Load Data ---
teams = pd.read_csv("data/epl_2015_16_xpts_clean.csv")
matches = pd.read_csv("data/epl_2015_16_matches_clean.csv")

avg_home_goals = matches["HomeGoals"].mean()
avg_away_goals = matches["AwayGoals"].mean()

# Strengths
teams["AttackStrength"] = teams["xG"] / teams["xG"].mean()
teams["DefenseStrength"] = teams["xGA"] / teams["xGA"].mean()
team_stats = teams.set_index("Team")[["AttackStrength","DefenseStrength"]].to_dict("index")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Controls")
n_sims = st.sidebar.slider("Number of simulations", 100, 5000, 1000, step=100)
leicester_boost = st.sidebar.slider("Leicester Attack Multiplier", 0.5, 2.0, 1.0, 0.1)

# --- Simulation Function ---
def simulate_season():
    points = defaultdict(int)
    for _, row in matches.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        # Apply Leicester boost if relevant
        ha_mult = leicester_boost if home == "Leicester" else 1.0
        aa_mult = leicester_boost if away == "Leicester" else 1.0

        λ_home = avg_home_goals * team_stats[home]["AttackStrength"] * ha_mult * team_stats[away]["DefenseStrength"]
        λ_away = avg_away_goals * team_stats[away]["AttackStrength"] * aa_mult * team_stats[home]["DefenseStrength"]

        g_home, g_away = poisson(λ_home), poisson(λ_away)

        if g_home > g_away:
            points[home] += 3
        elif g_home < g_away:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1
    return points

# --- Run Simulations ---
all_results = [simulate_season() for _ in range(n_sims)]
results_df = pd.DataFrame(all_results).fillna(0)

# --- Title ---
st.title("🏆 Leicester 2015/16: Miracle Season Simulator")

# --- Leicester Histogram ---
st.subheader("📊 Leicester Simulated Points Distribution")
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(results_df["Leicester"], bins=30, alpha=0.7, color="blue")
ax.axvline(81, color="red", linestyle="--", label="Actual Points (81)")
ax.set_title("Leicester Simulated Points")
ax.set_xlabel("Points"); ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# --- Title Probabilities ---
st.subheader("🏆 Title Probabilities")
title_probs = (results_df.idxmax(axis=1).value_counts() / n_sims) * 100
fig, ax = plt.subplots(figsize=(8,5))
title_probs.sort_values(ascending=False).head(10).plot(kind="bar", ax=ax)
ax.set_ylabel("Probability (%)")
st.pyplot(fig)

# --- Heatmap ---
st.subheader("🔥 Finish Position Heatmap")
positions = results_df.rank(axis=1, method="first", ascending=False).astype(int)
finish_probs = pd.DataFrame(0, index=results_df.columns, columns=range(1, len(results_df.columns)+1))
for team in results_df.columns:
    counts = positions[team].value_counts(normalize=True)
    for pos, prob in counts.items():
        finish_probs.loc[team, pos] = prob * 100

avg_finish = positions.mean().sort_values()
finish_probs = finish_probs.loc[avg_finish.index]

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(finish_probs, cmap="viridis", ax=ax)
ax.set_xlabel("Final League Position"); ax.set_ylabel("Team")
ax.set_title("EPL 2015/16 Finish Position Probabilities")
st.pyplot(fig)
