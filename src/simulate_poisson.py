import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import trange
from numpy.random import poisson
import matplotlib.pyplot as plt

# Load data
teams = pd.read_csv("data/epl_2015_16_xpts_clean.csv")
matches = pd.read_csv("data/epl_2015_16_matches_clean.csv")

# --- Step 1: League averages ---
league_avg_xG = teams["xG"].sum() / teams["Played"].sum() * 38  # total per season
league_avg_xGA = teams["xGA"].sum() / teams["Played"].sum() * 38

avg_home_goals = matches["HomeGoals"].mean()
avg_away_goals = matches["AwayGoals"].mean()

# --- Step 2: Strengths ---
teams["AttackStrength"] = teams["xG"] / teams["xG"].mean()
teams["DefenseStrength"] = teams["xGA"] / teams["xGA"].mean()

team_stats = teams.set_index("Team")[["AttackStrength", "DefenseStrength"]].to_dict("index")

def simulate_season():
    points = defaultdict(int)
    for _, row in matches.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        # Expected goals using attack/defense strengths
        λ_home = avg_home_goals * team_stats[home]["AttackStrength"] * team_stats[away]["DefenseStrength"]
        λ_away = avg_away_goals * team_stats[away]["AttackStrength"] * team_stats[home]["DefenseStrength"]

        # Poisson sampling
        g_home = poisson(λ_home)
        g_away = poisson(λ_away)

        # Points
        if g_home > g_away:
            points[home] += 3
        elif g_home < g_away:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

    return points

# --- Step 3: Run Monte Carlo ---
n_sims = 10000
all_results = []

for _ in trange(n_sims, desc="Simulating Seasons"):
    season_points = simulate_season()
    all_results.append(season_points)

# Convert to DataFrame
results_df = pd.DataFrame(all_results).fillna(0)


# --- Step 4: Summary stats ---
mean_points = results_df.mean().sort_values(ascending=False)
title_probs = (results_df.idxmax(axis=1).value_counts() / n_sims) * 100

print("📊 Average Points per Team:\n", mean_points)
print("\n🏆 Title Probabilities (%):\n", title_probs)

# --- Step 5: Leicester-specific plot ---
plt.figure(figsize=(10,6))
plt.hist(results_df["Leicester"], bins=30, alpha=0.7)
plt.axvline(81, color="red", linestyle="--", label="Actual Points (81)")
plt.title("Leicester Simulated Points Distribution (2015/16)(1)")
plt.xlabel("Points")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/leicester_points_distribution(1).png")
plt.close()



# --- Step 6: Title probabilities bar chart ---
top_titles = title_probs.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.bar(top_titles.index, top_titles.values, alpha=0.7)
plt.title("EPL 2015/16 Title Probabilities (Poisson Simulation)")
plt.ylabel("Probability (%)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/title_probabilities(1).png")
plt.close()

# --- Step 7: Finish position heatmap ---
import numpy as np

n_sims = results_df.shape[0]
teams = results_df.columns

# Rank each season (1 = champion, 20 = bottom)
positions = results_df.rank(axis=1, method="first", ascending=False).astype(int)

# Calculate finish probabilities
finish_probs = pd.DataFrame(0, index=teams, columns=range(1, len(teams)+1))
for team in teams:
    counts = positions[team].value_counts(normalize=True)
    for pos, prob in counts.items():
        finish_probs.loc[team, pos] = prob * 100

# Sort teams by average finishing position (nicer heatmap)
avg_finish = positions.mean().sort_values()
finish_probs = finish_probs.loc[avg_finish.index]

plt.figure(figsize=(14,8))
plt.imshow(finish_probs.values, aspect="auto", cmap="viridis")
plt.colorbar(label="Probability (%)")
plt.xticks(np.arange(len(finish_probs.columns)), finish_probs.columns)
plt.yticks(np.arange(len(finish_probs.index)), finish_probs.index)
plt.title("EPL 2015/16 Finish Position Probabilities (Poisson Simulation)")
plt.xlabel("Final League Position")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig("outputs/finish_position_heatmap(1).png")
plt.close()

