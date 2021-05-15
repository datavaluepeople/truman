import pandas as pd
import pytest

from truman import plot


@pytest.mark.parametrize(
    "use_cols, expected_plots, expected_scatters",
    [
        ("all", 1, 1),
        (["reward"], 1, 0),
        (["action"], 0, 1),
    ],
)
def test_plot(mocker, use_cols, expected_plots, expected_scatters):
    mocker.patch.object(plot, "plt")
    ax = mocker.MagicMock()

    history = pd.DataFrame({"done": [1], "action": [1], "reward": [1]})
    plot.plot(history, use_cols=use_cols, ax=ax)

    assert ax.twinx().plot.call_count == expected_plots
    assert ax.twinx().scatter.call_count == expected_scatters
