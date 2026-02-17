#!/usr/bin/env python3
"""
Test that player and injury features are correctly calculated.

This test verifies that:
1. The feature registry returns feature names that match what calculators generate
2. Features are calculated with non-zero values for games with injuries
3. The training data service returns the correct feature list

This test was created after a major bug was found where all player_ and inj_
features were calculating to 0 because the feature registry was generating
theoretical feature names that didn't match what calculators actually return.

Run with:
    PYTHONPATH=/Users/pranav/Documents/NBA pytest tests/test_player_injury_features.py -v
"""

import pytest
from datetime import date


class TestPlayerInjuryFeatures:
    """Test that player and injury features calculate correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        from bball_app.core.mongo import Mongo
        from bball_app.core.league_config import load_league_config

        self.mongo = Mongo()
        self.db = self.mongo.db
        self.league = load_league_config('nba')

    def test_registry_player_features_match_calculator(self):
        """Verify registry player features match what PERCalculator returns."""
        from bball_app.core.features.registry import FeatureGroups
        from bball_app.core.stats.per_calculator import PERCalculator

        # Get features from registry
        registry_features = set(FeatureGroups.get_features_for_group('player_talent'))

        # Get features from calculator
        per_calc = PERCalculator(self.db, preload=False, league=self.league)
        result = per_calc.get_game_per_features(
            home_team='ATL',
            away_team='BOS',
            season='2023-2024',
            game_date='2023-10-27'
        )

        if result is None:
            pytest.skip("No PER data available for test game")

        actual_features = {k for k in result.keys() if k.startswith('player_')}

        # All registry features should be in actual features
        missing = registry_features - actual_features
        assert len(missing) == 0, f"Registry features not found in calculator output: {missing}"

    def test_registry_injury_features_match_calculator(self):
        """Verify registry injury features match what StatHandlerV2 returns."""
        from bball_app.core.features.registry import FeatureGroups
        from bball_app.core.stats.handler import StatHandlerV2
        from bball_app.core.stats.per_calculator import PERCalculator

        # Get features from registry
        registry_features = set(FeatureGroups.get_features_for_group('injuries'))

        # Get a game with injuries
        game = self.db.stats_nba.find_one({
            'homeTeam.injured_players.0': {'$exists': True}
        })

        if not game:
            pytest.skip("No game with injuries found")

        date_parts = game['date'].split('-')
        year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])

        # Get features from calculator
        stat_handler = StatHandlerV2(
            statistics=[],
            use_exponential_weighting=False,
            db=self.db,
            lazy_load=True,
            league=self.league
        )
        per_calc = PERCalculator(self.db, preload=False, league=self.league)

        result = stat_handler.get_injury_features(
            game['homeTeam']['name'],
            game['awayTeam']['name'],
            game['season'],
            year, month, day,
            game_doc=game,
            per_calculator=per_calc
        )

        actual_features = {k for k in result.keys() if k.startswith('inj_')}

        # All registry features should be in actual features
        missing = registry_features - actual_features
        assert len(missing) == 0, f"Registry features not found in calculator output: {missing}"

    def test_player_features_have_nonzero_values(self):
        """Verify player features calculate to non-zero values."""
        from bball_app.core.stats.per_calculator import PERCalculator

        per_calc = PERCalculator(self.db, preload=False, league=self.league)
        result = per_calc.get_game_per_features(
            home_team='ATL',
            away_team='BOS',
            season='2023-2024',
            game_date='2023-10-27'
        )

        if result is None:
            pytest.skip("No PER data available for test game")

        player_features = {k: v for k, v in result.items() if k.startswith('player_')}
        nonzero_count = sum(1 for v in player_features.values() if v != 0)

        # Most player features should be non-zero
        assert nonzero_count > len(player_features) * 0.8, \
            f"Too many zero player features: {len(player_features) - nonzero_count}/{len(player_features)}"

    def test_injury_features_have_nonzero_values(self):
        """Verify injury features calculate to non-zero values for games with injuries."""
        from bball_app.core.stats.handler import StatHandlerV2
        from bball_app.core.stats.per_calculator import PERCalculator

        # Get a game with injuries
        game = self.db.stats_nba.find_one({
            'homeTeam.injured_players.0': {'$exists': True}
        })

        if not game:
            pytest.skip("No game with injuries found")

        date_parts = game['date'].split('-')
        year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])

        stat_handler = StatHandlerV2(
            statistics=[],
            use_exponential_weighting=False,
            db=self.db,
            lazy_load=True,
            league=self.league
        )
        per_calc = PERCalculator(self.db, preload=False, league=self.league)

        result = stat_handler.get_injury_features(
            game['homeTeam']['name'],
            game['awayTeam']['name'],
            game['season'],
            year, month, day,
            game_doc=game,
            per_calculator=per_calc
        )

        inj_features = {k: v for k, v in result.items() if k.startswith('inj_')}
        nonzero_count = sum(1 for v in inj_features.values() if v != 0)

        # At least half of injury features should be non-zero for a game with injuries
        assert nonzero_count > len(inj_features) * 0.5, \
            f"Too many zero injury features: {len(inj_features) - nonzero_count}/{len(inj_features)}"

    def test_training_data_service_returns_correct_features(self):
        """Verify get_all_possible_features returns features that calculators generate."""
        from bball_app.core.services.training_data import get_all_possible_features
        from bball_app.core.stats.per_calculator import PERCalculator
        from bball_app.core.stats.handler import StatHandlerV2

        all_features = get_all_possible_features(no_player=False)
        requested_player = [f for f in all_features if f.startswith('player_')]
        requested_inj = [f for f in all_features if f.startswith('inj_')]

        # Get a game with injuries
        game = self.db.stats_nba.find_one({
            'homeTeam.injured_players.0': {'$exists': True}
        })

        if not game:
            pytest.skip("No game with injuries found")

        date_parts = game['date'].split('-')
        year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
        home = game['homeTeam']['name']
        away = game['awayTeam']['name']
        season = game['season']

        # Get actual features from calculators
        per_calc = PERCalculator(self.db, preload=False, league=self.league)
        stat_handler = StatHandlerV2(
            statistics=[],
            use_exponential_weighting=False,
            db=self.db,
            lazy_load=True,
            league=self.league
        )

        per_result = per_calc.get_game_per_features(home, away, season, game['date'])
        inj_result = stat_handler.get_injury_features(
            home, away, season, year, month, day,
            game_doc=game,
            per_calculator=per_calc
        )

        if per_result is None:
            pytest.skip("No PER data available for test game")

        # Check all requested player features exist in results
        actual_player = set(per_result.keys())
        missing_player = [f for f in requested_player if f not in actual_player]
        assert len(missing_player) == 0, \
            f"Requested player features not returned by calculator: {missing_player}"

        # Check all requested injury features exist in results
        actual_inj = set(inj_result.keys())
        missing_inj = [f for f in requested_inj if f not in actual_inj]
        assert len(missing_inj) == 0, \
            f"Requested injury features not returned by calculator: {missing_inj}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
