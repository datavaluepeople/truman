import pandas as pd

from truman.run import store


def test_summarise():
    history = pd.DataFrame({"reward": [1, 2, 3, 4]})
    summary = store.summarise(history, elapsed_time=10, env_id="test_env", run_params=None)

    assert len(summary) == 4
    assert summary["avg_reward"] == 2.5
    assert summary["time_seconds"] == 10
    assert summary["env_id"] == "test_env"


def test_write(tmpdir):
    run_params = {"output_directory": str(tmpdir)}
    history = pd.DataFrame({"a": [1, 2, 3]})
    summary = {"something": 1}
    env_id = "test_env"
    store.write(history, summary, env_id, run_params)

    pd.testing.assert_frame_equal(history, pd.read_parquet(tmpdir / "test_env.parquet"))
    read_summary = pd.read_csv(tmpdir / "test_env_summary.csv")
    assert len(read_summary) == 1
    assert read_summary["something"].iloc[0] == 1
