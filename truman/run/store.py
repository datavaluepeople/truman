"""Utilities for summarising and storing results of simulation runs."""
import os.path

import pandas as pd


def summarise(history: pd.DataFrame, elapsed_time: float, env_id: str, run_params: dict) -> dict:
    """Summarise the history into a single row."""
    return {
        "avg_reward": history["reward"].mean(),
        "num_steps": len(history) - 1,
        "time_seconds": elapsed_time,
        "env_id": env_id,
    }


def write(history: pd.DataFrame, summary: dict, env_id: str, run_params: dict):
    """Write the history and summary to individual files in the run_params output directory."""
    write_base_fp = os.path.join(run_params["output_directory"], env_id)
    with open(f"{write_base_fp}.parquet", "wb") as history_fh:
        history.to_parquet(history_fh)
    with open(f"{write_base_fp}_summary.csv", "w") as summary_fh:
        # Convert the summary dict into a single row CSV
        pd.Series(summary).to_frame().T.to_csv(summary_fh, index=False)
