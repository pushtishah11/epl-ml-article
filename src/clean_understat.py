import pandas as pd
import ast

# -------------------------------
# 1. Leicester xG match data
# -------------------------------
def clean_leicester():
    df = pd.read_csv("leicester_xg.csv")
    clean = []

    for _, row in df.iterrows():
        h = ast.literal_eval(row["h"])
        a = ast.literal_eval(row["a"])
        goals = ast.literal_eval(row["goals"])
        xg = ast.literal_eval(row["xG"])
        forecast = ast.literal_eval(row["forecast"])

        clean.append({
            "MatchID": row["id"],
            "Date": row["datetime"],
            "Opponent": a["title"] if row["side"] == "h" else h["title"],
            "Venue": "Home" if row["side"] == "h" else "Away",
            "LeicesterGoals": int(goals[row["side"]]),
            "OpponentGoals": int(goals["a" if row["side"] == "h" else "h"]),
            "Leicester_xG": float(xg[row["side"]]),
            "Opponent_xG": float(xg["a" if row["side"] == "h" else "h"]),
            "Result": row["result"],
            "Forecast_Win": forecast["w"],
            "Forecast_Draw": forecast["d"],
            "Forecast_Loss": forecast["l"],
        })

    df_clean = pd.DataFrame(clean)
    df_clean.to_csv("leicester_xg_clean.csv", index=False)
    print("✅ Saved leicester_xg_clean.csv")
    return df_clean


# -------------------------------
# 2. Full EPL matches
# -------------------------------
def clean_matches():
    df = pd.read_csv("epl_2015_16_matches.csv")
    clean = []

    for _, row in df.iterrows():
        h = ast.literal_eval(row["h"])
        a = ast.literal_eval(row["a"])
        goals = ast.literal_eval(row["goals"])
        xg = ast.literal_eval(row["xG"])
        forecast = ast.literal_eval(row["forecast"])

        clean.append({
            "MatchID": row["id"],
            "Date": row["datetime"],
            "HomeTeam": h["title"],
            "AwayTeam": a["title"],
            "HomeGoals": int(goals["h"]),
            "AwayGoals": int(goals["a"]),
            "Home_xG": float(xg["h"]),
            "Away_xG": float(xg["a"]),
            "Forecast_HomeWin": float(forecast["w"]),
            "Forecast_Draw": float(forecast["d"]),
            "Forecast_AwayWin": float(forecast["l"]),
        })

    df_clean = pd.DataFrame(clean)
    df_clean.to_csv("epl_2015_16_matches_clean.csv", index=False)
    print("✅ Saved epl_2015_16_matches_clean.csv")
    return df_clean


# -------------------------------
# 3. League table (xPTS etc.)
# -------------------------------
def clean_league_table():
    df = pd.read_csv("epl_2015_16_xpts.csv")
    teams = df.iloc[1].tolist()
    history = df.iloc[2].tolist()

    clean = []
    for team, hist in zip(teams, history):
        games = ast.literal_eval(hist)
        xg = sum(float(g["xG"]) for g in games)
        xga = sum(float(g["xGA"]) for g in games)
        xpts = sum(float(g["xpts"]) for g in games)
        wins = sum(1 for g in games if g.get("result") == "w")
        draws = sum(1 for g in games if g.get("result") == "d")
        losses = sum(1 for g in games if g.get("result") == "l")

        clean.append({
            "Team": team,
            "xG": round(xg, 2),
            "xGA": round(xga, 2),
            "xPTS": round(xpts, 1),
            "Wins": wins,
            "Draws": draws,
            "Losses": losses,
            "Played": len(games)
        })

    df_clean = pd.DataFrame(clean)
    df_clean.to_csv("epl_2015_16_xpts_clean.csv", index=False)
    print("✅ Saved epl_2015_16_xpts_clean.csv")
    return df_clean


# -------------------------------
# Run all
# -------------------------------
if __name__ == "__main__":
    leicester = clean_leicester()
    matches = clean_matches()
    league_table = clean_league_table()

    print("\n🎯 Samples:")
    print(leicester.head(), "\n")
    print(matches.head(), "\n")
    print(league_table.head())
