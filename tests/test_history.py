import pytest

from truman import history


@pytest.fixture
def hist():
    hist = history.History()
    hist.append(None, (10, 5), None, None, None, None)
    hist.append(0, (10, 5), 5, False, {"info_1": 1, "info_2": 2}, {"agent_foo": 0})
    hist.append(1, (10, 5), 5, True, {"info_1": 0, "info_2": 0}, {"agent_foo": 1})
    return hist


def test_observable(hist):
    observables = hist.observable()
    assert observables[0] == [None, 0, 1]
    assert observables[1] == [(10, 5), (10, 5), (10, 5)]
    assert observables[2] == [None, 5, 5]
    assert observables[3] == [None, False, True]
    assert len(observables) == 4
    _all = hist.all()
    assert _all[:4] == observables[:4]
    assert _all[4] == [None, {"info_1": 1, "info_2": 2}, {"info_1": 0, "info_2": 0}]
    assert _all[5] == [None, {"agent_foo": 0}, {"agent_foo": 1}]
    assert len(_all) == 6


def test_to_df(hist):
    df = hist.to_df()
    assert df.columns.to_list() == [
        "action",
        "observation_0",
        "observation_1",
        "reward",
        "done",
        "info_1",
        "info_2",
        "agent_agent_foo",
    ]
    assert (df["action"].iloc[:1].isnull()).all()
    assert (df["action"].iloc[1:] == [0, 1]).all()
    assert (df["observation_0"] == [10, 10, 10]).all()
