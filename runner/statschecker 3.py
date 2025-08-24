import requests
import pandas as pd
import math
import numpy as np

# ---------------- CONFIG ---------------- #
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\runner\apikeys.env")  # Loads .env file

FACEIT_API_KEY = os.getenv("FACEIT_API_KEY")
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

FACEIT_BASE_URL = "https://open.faceit.com/data/v4"
STEAM_BASE_URL = "https://api.steampowered.com"

HEADERS = {"Authorization": f"Bearer {FACEIT_API_KEY}"}

pd.set_option("display.max_columns", None)




# ---------------- FACEIT FUNCTIONS ---------------- #

def get_player_data(nickname):
    url = f"{FACEIT_BASE_URL}/players"
    resp = requests.get(url, headers=HEADERS, params={"nickname": nickname})
    resp.raise_for_status()
    return resp.json()

def get_lifetime_stats(player_id):
    url = f"{FACEIT_BASE_URL}/players/{player_id}/stats/cs2"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        data = resp.json()
        return {k.strip(): v for k, v in data.get("lifetime", {}).items()}
    return None

def normalize_per_round(stats):
    if "Rounds" not in stats:
        return stats

    try:
        rounds_played = float(stats["Rounds"])
    except (ValueError, TypeError):
        return stats

    new_stats = {}
    for k, v in stats.items():
        try:
            num = float(v)
        except (ValueError, TypeError):
            new_stats[k] = v
            continue

        key_lower = k.lower()
        if "%" in k or "ratio" in key_lower or "streak" in key_lower or "matches" in key_lower:
            new_stats[k] = v
            continue

        per_round_value = round(num / rounds_played, 3) if rounds_played > 0 else 0
        new_stats[f"{k} per round"] = per_round_value

    return new_stats

# ---------------- STEAM FUNCTIONS ---------------- #

def get_steam_level(steam_id64):
    url = f"{STEAM_BASE_URL}/IPlayerService/GetSteamLevel/v1/"
    params = {"key": STEAM_API_KEY, "steamid": steam_id64}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("response", {}).get("player_level", None)

def get_cs2_hours_and_privacy(steam_id64):
    url = f"{STEAM_BASE_URL}/IPlayerService/GetOwnedGames/v1/"
    params = {
        "key": STEAM_API_KEY,
        "steamid": steam_id64,
        "include_appinfo": True,
        "include_played_free_games": True
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    response_data = resp.json().get("response", {})

    games = response_data.get("games", [])
    if not games:
        return math.nan, 1  # Private profile → NaN hours, flag=1

    for game in games:
        if game.get("appid") == 730:  # CS2
            return round(game.get("playtime_forever", 0) / 60, 1), 0  # hours, flag=0
    return 0, 0  # Public profile but no CS2 found

# ---------------- MAIN ---------------- #

def stats_lifetime(nickfile, output_excel):
    wanted_stats = {
        "Average K/D Ratio": "Average KD ratio",
        "ADR": "ADR",
        "Average Headshots %": "HS %",
        "Win Rate %": "WR %",
        "Matches": "Matches Played",
        "Utility Damage per Round": "Utility Damage per Round",
        "1v1 Win Rate": "1v1 Win Rate",
        "1v2 Win Rate": "1v2 Win Rate",
        "Entry Rate": "Entry Rate",
        "Entry Success Rate": "Entry Success"
    }

    with open(nickfile, "r", encoding="utf-8") as f:
        nicknames = [line.strip() for line in f if line.strip()]

    results = []
    skipped_count = 0

    for nick in nicknames:
        print(f"→ Processing {nick} …")
        try:
            player_data = get_player_data(nick)
            player_id = player_data.get("player_id")
            steam_id64 = player_data.get("steam_id_64")

            stats = get_lifetime_stats(player_id)
            if not stats:
                print(f"⚠️ No lifetime stats found for {nick}")
                continue

            try:
                total_rounds = float(stats.get("Total Rounds with extended stats", 0))
            except (ValueError, TypeError):
                total_rounds = 0

            if total_rounds == 0:
                print(f"⚠️ Skipping {nick} — 0 rounds played (stats likely wiped).")
                skipped_count += 1
                continue

            stats = normalize_per_round(stats)

            # Steam extras
            steam_level, cs2_hours, profile_private = None, math.nan, None
            cs2_hours_hidden_flag, steam_level_hidden_flag = 0, 0

            if steam_id64:
                try:
                    steam_level = get_steam_level(steam_id64)
                    cs2_hours, profile_private = get_cs2_hours_and_privacy(steam_id64)

                    # Handle hidden CS2 hours
                    faceit_matches = int(stats.get("Matches", 0))
                    if cs2_hours == 0 and faceit_matches > 0:
                        cs2_hours = np.nan
                        cs2_hours_hidden_flag = 1

                    # Handle hidden Steam level
                    if steam_level is None and faceit_matches > 0:
                        steam_level_hidden_flag = 1

                except Exception as e:
                    print(f"⚠️ Could not fetch Steam data for {nick}: {e}")

        except Exception as e:
            print(f"❌ Failed for {nick}: {e}")
            continue

        row = {
            "Nickname": nick,
            "Steam Level": steam_level,
            "Steam Level Hidden Flag": steam_level_hidden_flag,
            "CS2 Hours": cs2_hours,
            "CS2 Hours Hidden Flag": cs2_hours_hidden_flag,
            "Steam Profile Private": profile_private
        }

        for api_key, nice_name in wanted_stats.items():
            row[nice_name] = stats.get(api_key, None)

        try:
            total_kills = float(stats.get("Total Kills with extended stats", 0))
            row["Kills Per Round"] = round(total_kills / total_rounds, 2) if total_rounds > 0 else None
        except (ValueError, TypeError):
            row["Kills Per Round"] = None

        results.append(row)

    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_excel, index=False)
        print(f"✅ Lifetime stats saved to '{output_excel}'")
        print(f"ℹ️ Skipped {skipped_count} players with 0 rounds played.")
        print(df)
    else:
        print("⚠️ No data to save.")


if __name__ == "__main__":
    stats_lifetime(
        r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\runner\match_players.txt",
        r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\runner\match_players.xlsx"
    )
