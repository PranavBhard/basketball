"""
Basketball Ensemble Predictor.

Subclass of sportscore's BaseEnsemblePredictor with basketball-specific:
- Feature generation (SharedFeatureGenerator, player_filters)
- Result formatting (odds, winner determination)
- Player list tracking for UI display
"""

import re
from typing import Dict, List, Optional

from sportscore.models.base_ensemble import BaseEnsemblePredictor


class EnsemblePredictor(BaseEnsemblePredictor):
    """Basketball ensemble prediction using shared feature generation and cached models."""

    def __init__(self, db, ensemble_config: Dict, league=None):
        # Cache for tracking player lists (for UI display) - set before super().__init__
        # because _create_feature_generator is called during super().__init__
        self._per_player_lists: Dict = {}
        self._injury_player_lists: Dict = {}
        super().__init__(db, ensemble_config, league=league)

        # Also generate home/away companions for diff meta-features
        # so the UI can show the per-team breakdown alongside the diff value
        companions = []
        for feat in (self.meta_feature_names or []):
            parts = feat.split('|')
            if len(parts) >= 4 and parts[3] == 'diff':
                companions.append(feat.replace('|diff', '|home', 1))
                companions.append(feat.replace('|diff', '|away', 1))
        if companions:
            self.all_features = self._collect_unique_features([self.all_features, companions])

    def _get_model_config_collection_name(self) -> str:
        if self.league:
            return self.league.collections.get('model_config_classifier', 'nba_model_config')
        return 'nba_model_config'

    def _create_feature_generator(self):
        from bball.features.generator import SharedFeatureGenerator
        return SharedFeatureGenerator(self.db, preload_venues=True, league=self.league)

    def _generate_game_features(self, **kwargs) -> Dict:
        """Generate basketball game features using SharedFeatureGenerator."""
        # Validate player_filters (basketball-specific requirement)
        player_filters = kwargs.get('player_filters')
        home_team = kwargs.get('home_team')
        away_team = kwargs.get('away_team')
        if not player_filters:
            raise ValueError("player_filters is required for ensemble prediction")
        if home_team not in player_filters or 'playing' not in player_filters[home_team]:
            raise ValueError(f"player_filters must include '{home_team}' with 'playing' list")
        if away_team not in player_filters or 'playing' not in player_filters[away_team]:
            raise ValueError(f"player_filters must include '{away_team}' with 'playing' list")

        all_feature_dict = self.feature_generator.generate_features(
            feature_names=self.all_features, **kwargs
        )

        # Store player lists for UI
        self._per_player_lists = self.feature_generator._per_player_lists.copy()
        self._injury_player_lists = self.feature_generator._injury_player_lists.copy()

        return all_feature_dict

    def _format_prediction_result(self, ensemble_home_prob: float, home_team: str, away_team: str,
                                   base_model_breakdowns: List[Dict], meta_info: Dict,
                                   all_feature_dict: Dict,
                                   ensemble_draw_prob: float = None,
                                   ensemble_away_prob: float = None) -> Dict:
        """Format basketball prediction result with odds, winner, and breakdown."""
        meta_values = meta_info['meta_values']
        meta_feature_cols = meta_info['meta_feature_cols']
        meta_normalized_values = meta_info['meta_normalized_values']
        stacking_mode = meta_info['stacking_mode']
        use_disagree = meta_info['use_disagree']
        use_conf = meta_info['use_conf']
        use_logit = meta_info.get('use_logit', False)

        # Determine winner
        pred = 1 if ensemble_home_prob >= 0.5 else 0
        if pred == 1:
            winner = home_team
            winner_prob = ensemble_home_prob
        else:
            winner = away_team
            winner_prob = 1 - ensemble_home_prob

        # Convert probability to American odds
        if winner_prob >= 0.5:
            odds = int(-100 * winner_prob / (1 - winner_prob))
        else:
            odds = int(100 * (1 - winner_prob) / winner_prob)

        # Extract home/away companion values for diff meta-features (for UI breakdown)
        meta_companions = {}
        for feat in (self.meta_feature_names or []):
            parts = feat.split('|')
            if len(parts) >= 4 and parts[3] == 'diff':
                home_key = feat.replace('|diff', '|home', 1)
                away_key = feat.replace('|diff', '|away', 1)
                if home_key in all_feature_dict:
                    meta_companions[home_key] = float(all_feature_dict[home_key])
                if away_key in all_feature_dict:
                    meta_companions[away_key] = float(all_feature_dict[away_key])

        # Build features_dict with ensemble breakdown nested inside
        features_dict_with_breakdown = {
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in meta_values.items()},
            '_meta_companions': meta_companions,
            '_meta_feature_cols': meta_feature_cols,
            '_ensemble_run_id': self.ensemble_config.get('ensemble_run_id'),
            '_base_model_ids': [str(x) for x in self.base_model_ids],
            '_ensemble_breakdown': {
                'stacking_mode': stacking_mode,
                'use_disagree': use_disagree,
                'use_conf': use_conf,
                'use_logit': use_logit,
                'base_models': base_model_breakdowns,
                'meta_feature_cols': meta_feature_cols,
                'meta_feature_values': {col: float(meta_values.get(col, 0.0)) for col in meta_feature_cols},
                'meta_normalized_values': meta_normalized_values
            }
        }

        return {
            'predicted_winner': winner,
            'home_win_prob': round(100 * ensemble_home_prob, 1),
            'home_pts': None,
            'away_pts': None,
            'odds': odds,
            'home_games_played': None,
            'away_games_played': None,
            'features_dict': features_dict_with_breakdown,
            'base_model_breakdowns': base_model_breakdowns,
            'meta_features': meta_values
        }

    def get_player_lists(self) -> Dict:
        """Get player lists from the most recent prediction (for UI display)."""
        result = {}
        if self._per_player_lists:
            result.update(self._per_player_lists)
        if self._injury_player_lists:
            result.update(self._injury_player_lists)
        return result


def create_ensemble_predictor(db, ensemble_config: Dict, league=None) -> Optional[EnsemblePredictor]:
    """
    Factory function to create an EnsemblePredictor.

    Args:
        db: MongoDB database connection
        ensemble_config: Ensemble configuration from model config collection
        league: Optional LeagueConfig for league-aware collection routing

    Returns:
        EnsemblePredictor instance, or None if creation fails
    """
    try:
        return EnsemblePredictor(db, ensemble_config, league=league)
    except Exception as e:
        print(f"Failed to create EnsemblePredictor: {e}")
        import traceback
        traceback.print_exc()
        return None
