'''import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Load data ---
teams = pd.read_csv("data/epl_2015_16_xpts_clean.csv")

# --- 2. True league table (based on actual points) ---
teams["Points"] = teams["Wins"]*3 + teams["Draws"]
teams = teams.sort_values("Points", ascending=False).reset_index(drop=True)
teams["Rank"] = teams.index + 1

# --- 3. Define features & target ---
features = ["xG", "xGA", "xPTS", "Wins", "Draws", "Losses"]
X = teams[features]
y = teams["Rank"]

# Simplify target into buckets (Top 4, Mid 5–17, Relegation 18–20)
def categorize(rank):
    if rank <= 4:
        return "Top 4"
    elif rank <= 17:
        return "Mid"
    else:
        return "Relegation"

y_cat = teams["Rank"].apply(categorize)

# --- 4. Train model ---
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

# --- 5. Evaluate ---
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Top 4", "Mid", "Relegation"])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Top 4", "Mid", "Relegation"], yticklabels=["Top 4", "Mid", "Relegation"])
plt.title("Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/classification_confusion_matrix.png")
plt.close()

# --- 6. Leicester Prediction ---
leicester = teams[teams["Team"] == "Leicester"][features]
pred = model.predict(leicester)[0]
print(f"Leicester Actual Rank: 1 → Model Predicted: {pred}")
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
teams = pd.read_csv("data/epl_2015_16_xpts_clean.csv")

# If no Rank column → compute from actual table (sort by Wins/Draws/Points)
teams["Points"] = teams["Wins"]*3 + teams["Draws"]  # reconstruct actual points
teams["Rank"] = teams["Points"].rank(ascending=False, method="first").astype(int)

# --- Define categories ---
def categorize_position(row):
    if row["Rank"] <= 4:
        return "Top 4"
    elif row["Rank"] >= 18:
        return "Relegation"
    else:
        return "Mid"

teams["Category"] = teams.apply(categorize_position, axis=1)

# --- Features ---
X = teams[["xG", "xGA", "xPTS"]]
y = teams["Category"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- Model ---
clf = RandomForestClassifier(random_state=42, n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- Results ---
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Feature Importance ---
importances = clf.feature_importances_
feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=True)

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance in Predicting League Category")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# --- Leicester check ---
leicester_cat = teams.loc[teams["Team"] == "Leicester", "Category"].values[0]
leicester_pred = clf.predict(teams.loc[teams["Team"] == "Leicester", ["xG","xGA","xPTS"]])[0]
print(f"Leicester Actual Rank: 1 → Model Predicted: {leicester_pred} (Actual Category: {leicester_cat})")
