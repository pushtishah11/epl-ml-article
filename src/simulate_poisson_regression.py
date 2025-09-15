import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange
import statsmodels.api as sm
import os

# --- Setup ---
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

teams = pd.read_csv("data/epl_2015_16_xpts_clean.csv")
matches = pd.read_csv("data/epl_2015_16_matches_clean.csv")

# Strengths
teams["AttackStrength"] = teams["xG"] / teams["xG"].mean()
teams["DefenseStrength"] = teams["xGA"] / teams["xGA"].mean()
attack_strengths = teams.set_index("Team")["AttackStrength"].to_dict()
defense_strengths = teams.set_index("Team")["DefenseStrength"].to_dict()

# --- Step 1: Prepare data for regression ---
home_data = []
away_data = []
for _, row in matches.iterrows():
    home, away = row["HomeTeam"], row["AwayTeam"]
    home_data.append([row["HomeGoals"], attack_strengths[home], defense_strengths[away], 1.0])  # include home adv
    away_data.append([row["AwayGoals"], attack_strengths[away], defense_strengths[home]])

home_df = pd.DataFrame(home_data, columns=["Goals", "HomeAttack", "AwayDefense", "HomeAdvantage"])
away_df = pd.DataFrame(away_data, columns=["Goals", "AwayAttack", "HomeDefense"])

# --- Step 2: Fit Poisson regression ---
home_model = sm.GLM(home_df["Goals"], sm.add_constant(home_df[["HomeAttack", "AwayDefense", "HomeAdvantage"]]), family=sm.families.Poisson()).fit()
away_model = sm.GLM(away_df["Goals"], sm.add_constant(away_df[["AwayAttack", "HomeDefense"]]), family=sm.families.Poisson()).fit()

print(home_model.summary())
print(away_model.summary())

# --- Step 3: Expected goals function ---
def expected_goals(home, away):
    # home features (with intercept)
    home_features = np.array([1.0, attack_strengths[home], defense_strengths[away], 1.0])  
    home_features = home_features[:len(home_model.params)]
    λ_home = np.exp(home_features @ home_model.params.values)

    # away features (with intercept)
    away_features = np.array([1.0, attack_strengths[away], defense_strengths[home]])  
    away_features = away_features[:len(away_model.params)]
    λ_away = np.exp(away_features @ away_model.params.values)

    return λ_home, λ_away

# --- Step 4: Simulate one season ---
def simulate_season():
    points = defaultdict(int)
    for _, row in matches.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        λ_home, λ_away = expected_goals(home, away)

        g_home = np.random.poisson(λ_home)
        g_away = np.random.poisson(λ_away)

        if g_home > g_away:
            points[home] += 3
        elif g_home < g_away:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1
    return points

# --- Step 5: Monte Carlo ---
n_sims = 5000  # fast but reliable
all_results = []
for _ in trange(n_sims, desc="Simulating Seasons (Fast Regression)"):
    all_results.append(simulate_season())

results_df = pd.DataFrame(all_results).fillna(0)

# --- Step 6: Stats ---
mean_points = results_df.mean().sort_values(ascending=False)
title_probs = (results_df.idxmax(axis=1).value_counts() / n_sims) * 100

print("\n📊 Average Points per Team:\n", mean_points)
print("\n🏆 Title Probabilities (%):\n", title_probs)

# --- Step 7a: Leicester histogram ---
plt.figure(figsize=(10,6))
plt.hist(results_df["Leicester"], bins=30, alpha=0.7, color="blue")
plt.axvline(81, color="red", linestyle="--", label="Actual Points (81)")
plt.title("Leicester Simulated Points Distribution (2015/16, Regression)")
plt.xlabel("Points")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "leicester_points_distribution.png"))
plt.close()

# --- Step 7b: Title probabilities bar chart ---
plt.figure(figsize=(10,6))
title_probs.sort_values(ascending=False).plot(kind="bar")
plt.title("EPL 2015/16 Title Probabilities (Regression Simulation)")
plt.ylabel("Probability (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "title_probabilities.png"))
plt.close()

# --- Step 7c: Finish position heatmap ---
positions = results_df.rank(axis=1, method="first", ascending=False).astype(int)
finish_probs = pd.DataFrame(0.0, index=results_df.columns, columns=range(1, len(results_df.columns)+1))

for team in results_df.columns:
    counts = positions[team].value_counts(normalize=True)
    for pos, prob in counts.items():
        finish_probs.loc[team, pos] = prob * 100

avg_finish = positions.mean().sort_values()
finish_probs = finish_probs.loc[avg_finish.index]

plt.figure(figsize=(14,8))
plt.imshow(finish_probs.values, aspect="auto", cmap="viridis")
plt.colorbar(label="Probability (%)")
plt.xticks(np.arange(len(finish_probs.columns)), finish_probs.columns)
plt.yticks(np.arange(len(finish_probs.index)), finish_probs.index)
plt.title("EPL 2015/16 Finish Position Probabilities (Regression Simulation)")
plt.xlabel("Final League Position")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "finish_position_heatmap.png"))
plt.close()
