from pathlib import Path
from typing import Any, Dict, List, Union
import pickle
import json
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sp
from scipy.special import expit


def u_shaped_func(acwr_x: float) -> float:
    if acwr_x < 1:
        return expit(-3.4+2*(1-acwr_x)**2)
    elif 1 <= acwr_x < 1.7:
        return expit(-3.4 + (1-acwr_x) ** 2)
    else:
        return expit(1.5*acwr_x -5.4)


def s_shaped_func(acwr_x: float) -> float:
    if acwr_x < 1.3:
        return 0.02
    else:
        return expit(3.5*(acwr_x-1.3)-3.89)


def load_in_injuries(path_to_injuries: Path):
    return pd.read_csv(path_to_injuries, engine="python")


def flatten_list(some_list: List[Any]) -> List[Any]:
    return [item for sublist in some_list for item in sublist]


def load_in_names(path_to_file_folder: Path) -> List[str]:
    return pd.read_csv(path_to_file_folder / "acwr.csv", engine="python").columns[1:]


def strip_data(input_df: pd.DataFrame) -> pd.DataFrame:
    mask = input_df.replace(0, np.nan)
    f_idx = mask.first_valid_index()
    l_idx = mask.last_valid_index()
    return input_df.loc[f_idx:l_idx, :]


def correct_acwr(players: Dict[str, pd.DataFrame]):
    return_dict = {}
    for name, player_df in players.items():
        player_df["acwr"] = player_df["atl"]/player_df["ctl28"]
        player_df["acwr"] = player_df["acwr"].fillna(0)
        return_dict[name] = player_df
    return return_dict


def clean_type(injuries: pd.DataFrame) -> pd.DataFrame:
    injury_df = injuries.copy()
    injury_type = injury_df["type"]
    injury_keys = [json.loads(injury_type).keys() for injury_type in injury_type.tolist()]
    injury_values = [json.loads(injury_type).values() for injury_type in injury_type.tolist()]
    injury_df["type"] = injury_keys
    injury_df["type_location"] = injury_values
    injury_df["timestamp"] = pd.to_datetime(injury_df["timestamp"], infer_datetime_format=True)
    return injury_df.explode(["type", "type_location"]).dropna()


def rnd_sample_acwr(player: pd.DataFrame, injury_times: List[pd.DatetimeIndex]):
    nr_injuries = len(injury_times)
    extended_datetimes = flatten_list([[datetime+timedelta(i) for i in range(1, 14)] for datetime in injury_times])
    no_injuries = strip_data(player.drop([injury_times+extended_datetimes][0])).acwr
    return no_injuries.sample(nr_injuries, random_state=0)


def map_acwr_injury(players: Dict[str, pd.DataFrame], injury_df: pd.DataFrame, filter_by: Union[str, None]):
    mapped_acwr_injury = {}
    if filter_by:
        injury_df = filter_severity(injury_df, filter_by)
    grouped_injuries = injury_df.groupby("player_name")
    for name, group in grouped_injuries:
        mapped_acwr_injury[name] = {}
        unique_timestamps = list(set(group["timestamp"]))
        mapped_acwr_injury[name]["injuries"] = players[name].acwr.loc[unique_timestamps].reset_index(drop=False).copy()
        mapped_acwr_injury[name]["non-injuries"] = rnd_sample_acwr(players[name], unique_timestamps).reset_index(drop=False)
    return mapped_acwr_injury


def filter_severity(injury_df: pd.DataFrame, filter_by: str) -> pd.DataFrame:
    return injury_df.loc[injury_df["type_location"] == filter_by]


def load_in(path_to_players: Path, names: List[str]):
    with open(path_to_players, "rb") as f:
        players = pickle.load(f)
    players_dict = {name: player for name, player in zip(names, players)}
    return correct_acwr(players_dict)


def unfold_injuries_per_player(mapped_injury_acwr):
    all_players_injuries = pd.concat([item["injuries"] for item in mapped_injury_acwr.values()])
    all_players_injuries["label"] = np.ones(len(all_players_injuries))
    all_players_control = pd.concat([item["non-injuries"] for item in mapped_injury_acwr.values()])
    all_players_control["label"] = np.zeros(len(all_players_control))
    return pd.concat([all_players_injuries, all_players_control])


def plot_binary_dist(injuries):
    fig = plt.figure(figsize=(15, 10))
    plt.scatter(injuries["acwr"], injuries["label"] + 1, marker="x", color="black")
    plt.xlabel("ACWR")
    plt.ylabel("Control vs Injury")
    plt.show()


if __name__ == "__main__":
    path_to_injuries = Path(__file__).parent.parent / "data" / "features" / "injuries.csv"
    player_names = load_in_names(Path(__file__).parent.parent / "data" / "features")

    players = load_in(Path(__file__).parent.parent / "data" / "players.pkl", player_names)
    injury_df = clean_type(load_in_injuries(path_to_injuries)).copy()
    injuries2acwr = map_acwr_injury(players, injury_df, "minor")
    injuries = unfold_injuries_per_player(injuries2acwr)
    design_matrix = sp.add_costant(injuries["acwr"])
    logit_mod = sp.Logit(endog=injuries["label"], exog=design_matrix)
    logit_res = logit_mod.fit()
    print(logit_res.summary())

    pred_input = np.linspace(injuries["acwr"].min(), injuries["acwr"].max(), len(injuries["acwr"]))
    pred_input = sp.add_constant(pred_input)

    predictions = logit_mod.predict(params=logit_res.params, exog=pred_input)
    plt.scatter(injuries["acwr"], injuries["label"], marker="x")
    plt.axhline(y=0.5)
    plt.plot(pred_input[:,1], predictions, c='red')
    plt.show()
    player_risk = logit_mod.predict(params=logit_res.params, exog=sm.add_constant(list(players.values())[0].acwr))
    plt.scatter(list(players.values())[0].acwr, player_risk, marker="x", color="b")
    plt.show()
    #eg_player = list(players.values())[0].acwr
    s_risk_eg_player = list(players.values())[0].acwr.apply(lambda x: s_shaped_func(x))
    u_risk_eg_player = list(players.values())[0].acwr.apply(lambda x: u_shaped_func(x))
    #plt.plot(eg_player)
    #plt.plot(s_risk_eg_player.loc["2020-10-01":])
    #plt.plot(u_risk_eg_player.loc["2020-10-01":])
    #plt.scatter(s_risk_eg_player, player_risk, marker="x", color="b")
    #plt.show()
    #plt.plot(list(players.values())[0].readiness.reset_index(drop=True)/10)
    #plt.show()




