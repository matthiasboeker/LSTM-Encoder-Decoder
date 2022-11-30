from pathlib import Path
import pickle
from typing import List

from preprocessing.data_loader import SoccerPlayer
from preprocessing.read_in_data import initialise_players


def generate_player_pickles(players: List[SoccerPlayer], path_to_save_pickle: Path, name: str) -> None:
    player_dfs = [player.to_dataframe() for player in players]
    with open(path_to_save_pickle / name, "wb") as f:
        pickle.dump(player_dfs, f)

def main():
    players = initialise_players(Path(__file__).parent /"data"/"features")
    path_to_store = Path(__file__).parent /"data"
    generate_player_pickles(players, path_to_store, "players.pkl")

if __name__ == "__main__":
    main()