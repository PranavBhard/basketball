"""
Test that vegas_* features are properly loaded in the prediction workflow.

This verifies that PredictionContext preloads vegas/pregame_lines data
so that StatHandlerV2._get_vegas_feature() can access them without
falling back to per-feature DB queries.
"""

import pytest
from unittest.mock import MagicMock, patch
from collections import defaultdict


class TestVegasPredictionFlow:
    """Tests for vegas features in prediction workflow."""

    def test_prediction_context_preloads_vegas_fields(self):
        """Verify PredictionContext._preload_games() includes vegas/pregame_lines in projection."""
        from bball_app.core.services.prediction import PredictionContext

        # Mock DB
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        mock_collection.find = MagicMock(return_value=[])

        # Create context (will call _preload)
        with patch.object(PredictionContext, '_preload_player_stats'):
            with patch.object(PredictionContext, '_preload_venues'):
                ctx = PredictionContext(mock_db, season='2024-2025')

        # Check that find was called with vegas fields in projection
        find_calls = mock_collection.find.call_args_list
        assert len(find_calls) >= 1, "Expected at least one find call for games"

        # Get the projection from the games query
        games_call = find_calls[0]
        projection = games_call[0][1] if len(games_call[0]) > 1 else games_call[1].get('projection', {})

        assert 'vegas' in projection, "Projection should include 'vegas' field"
        assert 'pregame_lines' in projection, "Projection should include 'pregame_lines' field"
        assert projection['vegas'] == 1, "'vegas' should be projected (value=1)"
        assert projection['pregame_lines'] == 1, "'pregame_lines' should be projected (value=1)"

    def test_stat_handler_uses_preloaded_vegas_data(self):
        """Verify StatHandlerV2._get_vegas_feature() uses preloaded games_home data."""
        from bball_app.core.stats.handler import StatHandlerV2

        # Create handler with preloaded games data containing vegas
        handler = StatHandlerV2(statistics=[], db=None, league=None)
        handler.games_home = {
            '2024-2025': {
                '2025-01-15': {
                    'BOS': {
                        'homeTeam': {'name': 'BOS'},
                        'awayTeam': {'name': 'LAL'},
                        'vegas': {
                            'home_ML': -150,
                            'away_ML': 130,
                            'home_spread': -4.5,
                            'away_spread': 4.5,
                            'OU': 219.5
                        }
                    }
                }
            }
        }

        # Test vegas_ML
        home_ml = handler._get_vegas_feature('vegas_ML', 'BOS', 'LAL', 2025, 1, 15, 'home')
        assert home_ml == -150.0, f"Expected home_ML=-150, got {home_ml}"

        away_ml = handler._get_vegas_feature('vegas_ML', 'BOS', 'LAL', 2025, 1, 15, 'away')
        assert away_ml == 130.0, f"Expected away_ML=130, got {away_ml}"

        # Test vegas_spread
        home_spread = handler._get_vegas_feature('vegas_spread', 'BOS', 'LAL', 2025, 1, 15, 'home')
        assert home_spread == -4.5, f"Expected home_spread=-4.5, got {home_spread}"

        away_spread = handler._get_vegas_feature('vegas_spread', 'BOS', 'LAL', 2025, 1, 15, 'away')
        assert away_spread == 4.5, f"Expected away_spread=4.5, got {away_spread}"

        # Test vegas_ou
        ou = handler._get_vegas_feature('vegas_ou', 'BOS', 'LAL', 2025, 1, 15, 'none')
        assert ou == 219.5, f"Expected OU=219.5, got {ou}"

    def test_stat_handler_falls_back_to_pregame_lines(self):
        """Verify fallback to pregame_lines when vegas field is missing."""
        from bball_app.core.stats.handler import StatHandlerV2

        handler = StatHandlerV2(statistics=[], db=None, league=None)
        handler.games_home = {
            '2024-2025': {
                '2025-01-15': {
                    'BOS': {
                        'homeTeam': {'name': 'BOS'},
                        'awayTeam': {'name': 'LAL'},
                        # No 'vegas' field - use pregame_lines fallback
                        'pregame_lines': {
                            'home_ml': -140,
                            'away_ml': 120,
                            'spread': -3.5,
                            'over_under': 215.0
                        }
                    }
                }
            }
        }

        # Test fallback to pregame_lines
        home_ml = handler._get_vegas_feature('vegas_ML', 'BOS', 'LAL', 2025, 1, 15, 'home')
        assert home_ml == -140.0, f"Expected home_ml=-140 from pregame_lines, got {home_ml}"

        home_spread = handler._get_vegas_feature('vegas_spread', 'BOS', 'LAL', 2025, 1, 15, 'home')
        assert home_spread == -3.5, f"Expected spread=-3.5 from pregame_lines, got {home_spread}"

        # Away spread should be negated from pregame_lines.spread
        away_spread = handler._get_vegas_feature('vegas_spread', 'BOS', 'LAL', 2025, 1, 15, 'away')
        assert away_spread == 3.5, f"Expected away_spread=3.5 (negated), got {away_spread}"

        ou = handler._get_vegas_feature('vegas_ou', 'BOS', 'LAL', 2025, 1, 15, 'none')
        assert ou == 215.0, f"Expected over_under=215.0 from pregame_lines, got {ou}"

    def test_vegas_implied_prob_calculation(self):
        """Verify vegas_implied_prob formula: if ML<0: -ML/(-ML+100), else: 100/(ML+100)."""
        from bball_app.core.stats.handler import StatHandlerV2

        handler = StatHandlerV2(statistics=[], db=None, league=None)
        handler.games_home = {
            '2024-2025': {
                '2025-01-15': {
                    'BOS': {
                        'homeTeam': {'name': 'BOS'},
                        'awayTeam': {'name': 'LAL'},
                        'vegas': {
                            'home_ML': -150,  # Favorite: (-(-150)) / (-(-150) + 100) = 150/250 = 0.6
                            'away_ML': 130,   # Underdog: 100 / (130 + 100) = 100/230 ≈ 0.4348
                        }
                    }
                }
            }
        }

        home_prob = handler._get_vegas_feature('vegas_implied_prob', 'BOS', 'LAL', 2025, 1, 15, 'home')
        expected_home = 150 / 250  # 0.6
        assert abs(home_prob - expected_home) < 0.001, f"Expected home_prob={expected_home}, got {home_prob}"

        away_prob = handler._get_vegas_feature('vegas_implied_prob', 'BOS', 'LAL', 2025, 1, 15, 'away')
        expected_away = 100 / 230  # ≈ 0.4348
        assert abs(away_prob - expected_away) < 0.001, f"Expected away_prob≈{expected_away}, got {away_prob}"

    def test_missing_vegas_returns_zero(self):
        """Verify that missing vegas data returns 0.0 gracefully."""
        from bball_app.core.stats.handler import StatHandlerV2

        handler = StatHandlerV2(statistics=[], db=None, league=None)
        handler.games_home = {
            '2024-2025': {
                '2025-01-15': {
                    'BOS': {
                        'homeTeam': {'name': 'BOS'},
                        'awayTeam': {'name': 'LAL'},
                        # No vegas or pregame_lines data
                    }
                }
            }
        }

        # Should return 0.0 when no vegas data available
        result = handler._get_vegas_feature('vegas_ML', 'BOS', 'LAL', 2025, 1, 15, 'home')
        assert result == 0.0, f"Expected 0.0 for missing vegas data, got {result}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
