import pandas as pd
import statsmodels.api as sm
import pickle
import os

# --- Paths ---
DATA_DIR = "data"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

matches = pd.read_csv(os.path.join(DATA_DIR, "epl_2015_16_matches_clean.csv"))
teams = pd.read_csv(os.path.join(DATA_DIR, "epl_2015_16_xpts_clean.csv"))

# Compute team strengths
teams["AttackStrength"] = teams["xG"] / teams["xG"].mean()
teams["DefenseStrength"] = teams["xGA"] / teams["xGA"].mean()
team_stats = teams.set_index("Team")[["AttackStrength", "DefenseStrength"]]

# Merge features into matches
def add_features(df):
    df = df.copy()
    df["HomeAttack"] = df["HomeTeam"].map(team_stats["AttackStrength"])
    df["HomeDefense"] = df["HomeTeam"].map(team_stats["DefenseStrength"])
    df["AwayAttack"] = df["AwayTeam"].map(team_stats["AttackStrength"])
    df["AwayDefense"] = df["AwayTeam"].map(team_stats["DefenseStrength"])
    df["HomeAdvantage"] = 1
    return df

matches_feat = add_features(matches)

# --- Model for Home Goals ---
X_home = matches_feat[["HomeAttack", "AwayDefense", "HomeAdvantage"]]
y_home = matches_feat["HomeGoals"]

X_home = sm.add_constant(X_home)
home_model = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()

# --- Model for Away Goals ---
X_away = matches_feat[["AwayAttack", "HomeDefense"]]
y_away = matches_feat["AwayGoals"]

X_away = sm.add_constant(X_away)
away_model = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()

# Save models
with open(os.path.join(OUTPUT_DIR, "home_model.pkl"), "wb") as f:
    pickle.dump(home_model, f)

with open(os.path.join(OUTPUT_DIR, "away_model.pkl"), "wb") as f:
    pickle.dump(away_model, f)

print("✅ Poisson regression models trained and saved in 'models/'")
print(home_model.summary())
print(away_model.summary())
