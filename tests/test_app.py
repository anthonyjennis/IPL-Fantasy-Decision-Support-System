"""Headless smoke tests for streamlit_app.py, including the v1 crash cases."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def run(setup=lambda a: None):
    at = AppTest.from_file(APP, default_timeout=60); at.run(); setup(at)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_loads():
    run()


def test_optimize_default_is_valid_xi():
    at = run(lambda a: a.button[0].click().run())
    xi = at.dataframe[-1].value
    assert len(xi) == 11 and xi["Role"].nunique() == 4


def test_optimize_with_too_few_players_warns_instead_of_crashing():   # v1: ValueError
    at = run(lambda a: (a.sidebar.slider[0].set_value(45).run(), a.button[0].click().run()))
    assert any("Loosen the filters" in w.value for w in at.warning)


def test_two_team_match_respects_team_cap():
    def s(a):
        a.sidebar.multiselect[0].select("CSK").select("MI").run()
        a.sidebar.number_input[0].set_value(7).run(); a.button[0].click().run()
    xi = run(s).dataframe[-1].value
    assert xi["Team_2025"].value_counts().max() <= 7
