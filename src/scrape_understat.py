'''import pandas as pd
from understatapi import UnderstatClient

with UnderstatClient() as understat:
    # 1. Leicester 2015/16 matches
    leicester_data = understat.team("Leicester").get_match_data(season="2015")
    leicester_df = pd.DataFrame(leicester_data)
    leicester_df.to_csv("leicester_xg.csv", index=False)
    print("✅ leicester_xg.csv saved")
    print(leicester_df.head())

    # 2. EPL 2015/16 league totals (use get_team_data)
    league_data = understat.league("EPL").get_team_data(season="2015")
    league_df = pd.DataFrame(league_data)
    league_df.to_csv("epl_2015_16_xpts.csv", index=False)
    print("✅ epl_2015_16_xpts.csv saved")
    print(league_df.head())

    # 3. EPL 2015/16 all matches (380 fixtures)
    matches_data = understat.league("EPL").get_match_data(season="2015")
    matches_df = pd.DataFrame(matches_data)
    matches_df.to_csv("epl_2015_16_matches.csv", index=False)
    print("✅ epl_2015_16_matches.csv saved")
    print(matches_df.head())
'''
import pandas as pd
from understatapi import UnderstatClient

with UnderstatClient() as understat:
    # 1. Leicester 2015/16 matches (xG only)
    leicester = understat.team("Leicester").get_match_data(season="2015")
    leicester_clean = []
    for m in leicester:
        leicester_clean.append({
            "Date": m["datetime"][:10],
            "is_home": m["side"] == "h",
            "Opponent": m["a"]["title"] if m["side"] == "h" else m["h"]["title"],
            "xG": float(m["h"]["xG"]) if m["side"] == "h" else float(m["a"]["xG"]),
            "xGA": float(m["a"]["xG"]) if m["side"] == "h" else float(m["h"]["xG"]),
            "Result": m["result"]
        })
    leicester_df = pd.DataFrame(leicester_clean)
    leicester_df.to_csv("leicester_xg.csv", index=False)
    print("✅ leicester_xg.csv saved")
    print(leicester_df.head())

    # 2. EPL 2015/16 league totals
    league = understat.league("EPL").get_team_data(season="2015")
    league_clean = []
    for team in league:
        history = team["history"]
        league_clean.append({
            "Team": team["title"],
            "xG": sum(float(h["xG"]) for h in history),
            "xGA": sum(float(h["xGA"]) for h in history),
            "xPTS": sum(float(h["xpts"]) for h in history),
        })
    league_df = pd.DataFrame(league_clean)
    league_df.to_csv("epl_2015_16_xpts.csv", index=False)
    print("✅ epl_2015_16_xpts.csv saved")
    print(league_df.head())

    # 3. EPL 2015/16 matches (only xG data)
    matches = understat.league("EPL").get_match_data(season="2015")
    matches_clean = []
    for m in matches:
        matches_clean.append({
            "Date": m["datetime"][:10],
            "Home": m["h"]["title"],
            "Away": m["a"]["title"],
            "Home_xG": float(m["h"]["xG"]),
            "Away_xG": float(m["a"]["xG"]),
            "Result": m["result"]
        })
    matches_df = pd.DataFrame(matches_clean)
    matches_df.to_csv("epl_2015_16_matches.csv", index=False)
    print("✅ epl_2015_16_matches.csv saved")
    print(matches_df.head())
