from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot_ts_result_eval(ground_truth: np.array, predictions: np.array, path_to_save_figures: Path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(ground_truth, label="Ground Truth")
    ax.plot(predictions, label="Predictions")
    ax.set_title("Prediction vs Ground Truth")
    ax.set_ylabel("Score")
    ax.set_xlabel("Time Steps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_to_save_figures)


def plot_training_loss(training_loss: np.array, validation_loss: np.array, path_to_save_figures: Path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(training_loss, label="Training Loss")
    ax.plot(validation_loss, label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_to_save_figures)


def plot_feature_distributions(summary_module, selected_features: List[str]):
    nr_features = len(selected_features)
    fig, axs = plt.subplots(nrows=int(nr_features/2), ncols=2)
    for ax in axs.ravel():
        ax.plot()
