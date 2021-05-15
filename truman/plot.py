"""Plotting env histories."""
from typing import List, Union
from typing_extensions import Literal

import logging

import pandas as pd


logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.info("Importing matplotlib failed. Plotting will not work.")


def plot(
    df: pd.DataFrame,
    alpha: float = 0.7,
    use_cols: Union[List[str], Literal["all"]] = "all",
    ax=None,
):
    """Plot the history of an environment from its history dataframe.

    Plot a scatter plot for the actions, and line plots for each other 1D series (the rewards,
    each component of the observations, any data included in the environment info).

    This function will fail if matplotlib is not installed.

    alpha: alpha of the plotted lines, float between 0 and 1
    use_cols: default "all", otherwise a list of the columns of the history.to_df() dataframe to
        plot
    ax: an optional matplotlib ax to plot onto
    """
    ax1 = ax or plt.gca()

    ax1.set_xlabel("step")
    ax1.set_yticks([])

    if use_cols == "all":
        use_cols = df.drop("done", axis=1).columns
    else:
        use_cols = list(set(use_cols).intersection(set(df.columns)))

    for i, column in enumerate(use_cols):
        ax = ax1.twinx()
        ax.set_ylabel(column, color=f"C{i}")
        ax.tick_params(axis="y", labelcolor=f"C{i}")
        if column == "action":
            plotter = ax.scatter
        else:
            plotter = ax.plot
        plotter(df.index, df[column], color=f"C{i}", alpha=alpha)
        ax.yaxis.tick_right()
        ax.spines["right"].set_position(("outward", 60 * i))
