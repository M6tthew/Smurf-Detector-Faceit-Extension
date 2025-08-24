import requests
import sys

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\runner\apikeys.env")  # Loads .env file


# ==== CONFIGURE BASE URL AND HEADERS ====
API_KEY = os.getenv("FACEIT_API_KEY")
BASE_URL = "https://open.faceit.com/data/v4"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}



print("API_KEY:", API_KEY)

def get_match_players(match_id):
    """Get all players in a match."""
    url = f"{BASE_URL}/matches/{match_id}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    players = []
    for team in data["teams"].values():
        for player in team["roster"]:
            players.append({
                "player_id": player["player_id"],
                "nickname": player["nickname"]
            })
    return players

def save_players_to_file(match_id, filename=r"C:\Users\Matthew\OneDrive\Documents\Faceit 2\runner\match_players.txt"):
    players = get_match_players(match_id)
    with open(filename, "w", encoding="utf-8") as f:
        for player in players:
            f.write(f"{player['nickname']}\n")
    print(f"[+] Saved {len(players)} players to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python matchstater.py <match_id>")
        sys.exit(1)

    match_id = sys.argv[1]  # <-- grab match_id from command line
    save_players_to_file(match_id)
