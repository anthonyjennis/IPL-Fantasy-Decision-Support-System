"""Hand-checked expectations for pipeline/build_from_cricsheet.py on a synthetic match."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import build_from_cricsheet as b  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def rows():
    df = b.build(FIX)
    return df, {(r.match_id, r.player): r for r in df.itertuples()}


def test_shape_and_meta():
    df, _ = rows()
    assert df["match_id"].nunique() == 2
    assert len(df) == 44                       # 22 named players x 2 matches
    assert set(df["season"]) == {2026}
    assert df.loc[df.match_id == "match_2_no_result", "no_result"].all()


def test_batting_excludes_wides_and_super_over():
    _, r = rows()
    a1 = r[("match_1", "A1")]
    # faced: the 4, the 6, the no-ball, the wicket ball (the wide is not a ball faced); super-over 6 ignored
    assert (a1.runs, a1.balls_faced, a1.fours, a1.sixes) == (11, 4, 1, 1)
    assert a1.dismissed and a1.dismissal_kind == "caught"                     # super-over 'bowled' ignored
    assert a1.batting_position == 1


def test_bowling_counts_and_maiden():
    _, r = rows()
    b1 = r[("match_1", "B1")]
    # conceded: 4 + 6 + 1 wide + (1 + 1 no-ball) + 2; the leg-bye is not the bowler's
    assert (b1.balls_bowled, b1.runs_conceded, b1.wickets) == (5, 15, 1)
    b2 = r[("match_1", "B2")]
    assert (b2.balls_bowled, b2.runs_conceded, b2.wickets, b2.maidens) == (6, 0, 1, 1)
    a9 = r[("match_1", "A9")]
    assert (a9.wickets, a9.lbw_bowled_wickets, a9.catches, a9.maidens) == (2, 1, 1, 0)  # run out not credited; 1 run conceded


def test_fielding():
    _, r = rows()
    assert r[("match_1", "B5")].catches == 1
    assert r[("match_1", "B9")].stumpings == 1
    assert r[("match_1", "A5")].run_outs == 1 and r[("match_1", "A1")].run_outs == 1
    assert r[("match_1", "A1")].won and not r[("match_1", "B1")].won
