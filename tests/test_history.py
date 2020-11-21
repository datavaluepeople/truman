from truman import history


def test_history():
    hist = history.History()
    hist.append(0, (10, 5), 5, False, {"info_1": 1, "info_2": 2})
    hist.append(1, (10, 5), 5, True, {"info_1": 0, "info_2": 0})
    observables = hist.observable()
    assert observables[0] == [0, 1]
    assert observables[1] == [(10, 5), (10, 5)]
    assert observables[2] == [5, 5]
    assert observables[3] == [False, True]
    assert len(observables) == 4
    _all = hist.all()
    assert _all[:4] == observables[:4]
    assert _all[4] == [{"info_1": 1, "info_2": 2}, {"info_1": 0, "info_2": 0}]
    assert len(_all) == 5
