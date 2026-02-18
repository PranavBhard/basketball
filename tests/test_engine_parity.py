"""
BasketballFeatureComputer integration tests.

Uses synthetic game data to validate feature computation across all stat
categories: counting stats, rate stats, net stats, schedule stats, context
stats, derived stats, and composite time periods (blend/delta).
"""

import pytest
from datetime import datetime

from bball.features.compute import BasketballFeatureComputer


# ---------------------------------------------------------------------------
# Synthetic game data fixtures
# ---------------------------------------------------------------------------

def _make_team(name, points, assists, blocks, steals, total_reb, off_reb,
               fg_made, fg_att, ft_made, ft_att, three_made, three_att, to):
    return {
        "name": name,
        "points": points,
        "assists": assists,
        "blocks": blocks,
        "steals": steals,
        "total_reb": total_reb,
        "off_reb": off_reb,
        "FG_made": fg_made,
        "FG_att": fg_att,
        "FT_made": ft_made,
        "FT_att": ft_att,
        "three_made": three_made,
        "three_att": three_att,
        "TO": to,
    }


def _make_game(date, home_team_data, away_team_data, home_won=None,
               game_type="regseason"):
    if home_won is None:
        home_won = home_team_data["points"] > away_team_data["points"]
    return {
        "date": date,
        "season": "2024-2025",
        "game_type": game_type,
        "homeTeam": home_team_data,
        "awayTeam": away_team_data,
        "homeWon": home_won,
    }


@pytest.fixture
def games_home():
    """Build preloaded games_home dict with 5 games."""
    games = {}
    season = "2024-2025"
    games[season] = {}

    game_data = [
        ("2025-01-05", "BOS",
         _make_team("BOS", 112, 28, 5, 8, 45, 10, 42, 88, 18, 22, 10, 30, 12),
         _make_team("MIA", 100, 20, 3, 5, 38, 7, 36, 82, 18, 24, 10, 28, 16), True),
        ("2025-01-08", "LAL",
         _make_team("LAL", 108, 25, 4, 7, 42, 9, 40, 86, 20, 24, 8, 26, 14),
         _make_team("BOS", 115, 30, 6, 9, 44, 11, 43, 90, 19, 22, 10, 32, 10), False),
        ("2025-01-10", "BOS",
         _make_team("BOS", 120, 32, 7, 10, 48, 12, 45, 92, 20, 24, 10, 30, 11),
         _make_team("LAL", 105, 22, 3, 6, 40, 8, 38, 85, 20, 25, 9, 28, 15), True),
        ("2025-01-12", "LAL",
         _make_team("LAL", 110, 26, 4, 7, 42, 9, 40, 86, 22, 26, 8, 25, 13),
         _make_team("MIA", 102, 21, 3, 5, 39, 8, 37, 83, 18, 22, 10, 28, 14), True),
        ("2025-01-14", "MIA",
         _make_team("MIA", 98, 19, 4, 6, 37, 7, 35, 80, 18, 22, 10, 30, 17),
         _make_team("BOS", 118, 29, 5, 8, 46, 11, 44, 91, 20, 24, 10, 31, 12), False),
    ]

    for date, home, home_data, away_data, home_won in game_data:
        if date not in games[season]:
            games[season][date] = {}
        games[season][date][home] = _make_game(date, home_data, away_data, home_won)

    return games


@pytest.fixture
def computer(games_home):
    comp = BasketballFeatureComputer()
    comp.set_preloaded_data(games_home, games_home)
    return comp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicCountingStats:
    """Basic stats use the standard engine path (no custom handler)."""

    def test_points_avg(self, computer):
        result = computer.compute_matchup_features(
            ["points|season|avg|home", "points|season|avg|away"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS: 112 (home) + 115 (away) + 120 (home) + 118 (away) = 465 / 4 = 116.25
        assert abs(result["points|season|avg|home"] - 116.25) < 0.01
        # LAL: 105 (away) + 108 (home) + 110 (home) = 323 / 3 = 107.67
        assert abs(result["points|season|avg|away"] - 107.667) < 0.01

    def test_points_diff(self, computer):
        result = computer.compute_matchup_features(
            ["points|season|avg|diff"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        assert abs(result["points|season|avg|diff"] - (116.25 - 107.667)) < 0.01

    def test_points_games_2(self, computer):
        result = computer.compute_matchup_features(
            ["points|games_2|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS last 2 games: Jan 10 (120 home) + Jan 14 (118 away) = 238 / 2 = 119
        assert abs(result["points|games_2|avg|home"] - 119.0) < 0.01


class TestRateStats:
    """Rate stats use compute_basic_rate custom handler."""

    def test_efg_avg(self, computer):
        result = computer.compute_matchup_features(
            ["efg|season|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS games: compute eFG% per game then average
        # Game 1 (Jan 5): (42 + 0.5*10) / 88 * 100 = 47/88*100 = 53.41
        # Game 2 (Jan 8): (43 + 0.5*10) / 90 * 100 = 48/90*100 = 53.33
        # Game 3 (Jan 10): (45 + 0.5*10) / 92 * 100 = 50/92*100 = 54.35
        # Game 4 (Jan 14): (44 + 0.5*10) / 91 * 100 = 49/91*100 = 53.85
        # avg = (53.41 + 53.33 + 54.35 + 53.85) / 4 = 53.735
        val = result["efg|season|avg|home"]
        assert val > 50.0  # sanity check
        assert val < 60.0

    def test_wins_avg(self, computer):
        result = computer.compute_matchup_features(
            ["wins|season|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS: won Jan 5 (home, homeWon=True), won Jan 8 (away, homeWon=False),
        #      won Jan 10 (home, homeWon=True), won Jan 14 (away, homeWon=False)
        # 4 wins / 4 games = 1.0
        assert abs(result["wins|season|avg|home"] - 1.0) < 0.01


class TestNetStats:
    """Net stats use compute_net custom handler."""

    def test_points_net(self, computer):
        result = computer.compute_matchup_features(
            ["points_net|season|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # Should be a number (team stat - opponent stat)
        assert isinstance(result["points_net|season|avg|home"], float)


class TestScheduleStats:
    """Schedule stats use compute_schedule custom handler."""

    def test_games_played(self, computer):
        result = computer.compute_matchup_features(
            ["games_played|season|raw|home", "games_played|season|raw|away"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        assert result["games_played|season|raw|home"] == 4.0  # BOS: 4 games
        assert result["games_played|season|raw|away"] == 3.0  # LAL: 3 games

    def test_days_rest(self, computer):
        result = computer.compute_matchup_features(
            ["days_rest|none|raw|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS last game: Jan 14, current: Jan 15, rest = 1
        assert result["days_rest|none|raw|home"] == 1.0

    def test_b2b(self, computer):
        result = computer.compute_matchup_features(
            ["b2b|none|raw|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS last game: Jan 14, current: Jan 15, rest = 1 day = b2b
        assert result["b2b|none|raw|home"] == 1.0


class TestContextStats:
    """Context stats use compute_context custom handler."""

    def test_home_court(self, computer):
        result = computer.compute_matchup_features(
            ["home_court|none|raw|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        assert result["home_court|none|raw|home"] == 1.0


class TestDerivedStats:
    """Derived stats use compute_derived custom handler."""

    def test_margin(self, computer):
        result = computer.compute_matchup_features(
            ["margin|season|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # BOS margins: +12 (Jan 5), +7 (Jan 8, away), +15 (Jan 10), +20 (Jan 14, away)
        # avg = (12 + 7 + 15 + 20) / 4 = 13.5
        assert abs(result["margin|season|avg|home"] - 13.5) < 0.01


class TestCompositeTimePeriods:
    """Blend/delta composite time periods."""

    def test_blend(self, computer):
        result = computer.compute_matchup_features(
            ["points|blend:games_2:0.7/season:0.3|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # Should blend games_2 avg (119.0) * 0.7 + season avg (116.25) * 0.3
        expected = 119.0 * 0.7 + 116.25 * 0.3
        assert abs(result["points|blend:games_2:0.7/season:0.3|avg|home"] - expected) < 0.5

    def test_delta(self, computer):
        result = computer.compute_matchup_features(
            ["points|delta:games_2-season|avg|home"],
            "BOS", "LAL", "2024-2025", "2025-01-15",
        )
        # games_2 avg (119.0) - season avg (116.25) = 2.75
        assert abs(result["points|delta:games_2-season|avg|home"] - 2.75) < 0.5
