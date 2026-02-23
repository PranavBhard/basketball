"""Service factory for the basketball web plugin.

Returns the ``services`` dict that shared_routes.py expects on ``g.services``.
Heavy objects (ConfigManager, TrainingService) are cached per league.
"""

import logging
import os
import subprocess
import sys
from typing import Dict

from sportscore.web.service_helpers import (
    get_available_features,
    make_jobs_service,
    make_market_helpers,
)
from sportscore.web.services import (
    SportServices, ModelServices, MarketServices,
    FeatureServices, EloServices, DataServices,
)

logger = logging.getLogger(__name__)

# Process-level caches (keyed by league_id)
_config_managers: Dict[str, object] = {}
_training_services: Dict[str, object] = {}


def _get_config_manager(db, league):
    lid = getattr(league, "league_id", "default")
    if lid not in _config_managers:
        from bball.services.config_manager import ModelConfigManager
        _config_managers[lid] = ModelConfigManager(db)
    return _config_managers[lid]


def _get_training_service(db, league):
    lid = getattr(league, "league_id", "default")
    if lid not in _training_services:
        from bball.services.training_service import TrainingService
        _training_services[lid] = TrainingService(league=league, db=db)
    return _training_services[lid]



# ---------------------------------------------------------------------------
# Market: sport-specific price getter (uses bball.market.kalshi)
# ---------------------------------------------------------------------------


def _market_prices_getter(date_str, db, league):
    """Fetch public Kalshi market prices for all games on a date."""
    from datetime import datetime as _dt
    from bball.market.kalshi import get_game_market_data

    try:
        game_date = _dt.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"success": False, "error": "Invalid date format"}

    lid = getattr(league, "league_id", "nba")
    games_coll = league.collections.get("games", "stats_nba")
    games = list(db[games_coll].find(
        {"date": date_str},
        {"game_id": 1, "homeTeam.name": 1, "awayTeam.name": 1},
    ))
    if not games:
        return {"success": True, "markets": {}, "message": "No games for this date"}

    markets = {}
    for game in games:
        gid = game.get("game_id")
        ht = game.get("homeTeam", {}).get("name", "")
        at = game.get("awayTeam", {}).get("name", "")
        if not ht or not at:
            continue
        try:
            md = get_game_market_data(
                game_date=game_date, away_team=at, home_team=ht,
                league_id=lid, use_cache=True, cache_ttl=30,
            )
            if md:
                markets[gid] = md.to_dict()
        except Exception:
            continue
    return {"success": True, "markets": markets}



# ---------------------------------------------------------------------------
# Feature / master training helpers
# ---------------------------------------------------------------------------

def _feature_dependency_resolver(feature_substrings, match_mode="OR"):
    """Resolve feature dependencies from substrings."""
    try:
        from bball.features.sets import find_features_by_substrings
        from bball.features.dependencies import resolve_dependencies, categorize_features
        matched = find_features_by_substrings(feature_substrings, match_mode=match_mode)
        deps = resolve_dependencies(matched)
        categories = categorize_features(deps)
        return {"success": True, "features": list(deps), "categories": categories, "matched": list(matched)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _feature_regenerator(data, league=None):
    """Regenerate features in master training CSV."""
    try:
        from bball.features.sets import find_features_by_substrings
        from bball.features.dependencies import resolve_dependencies
        feature_substrings = data.get("feature_substrings", [])
        match_mode = data.get("match_mode", "OR")
        matched = find_features_by_substrings(feature_substrings, match_mode=match_mode)
        features = resolve_dependencies(matched)
        seasons = data.get("seasons")

        # Spawn subprocess to regenerate
        cmd = [
            sys.executable, "-m", "bball.pipeline.training_pipeline",
            getattr(league, "league_id", "nba"),
            "--features", ",".join(features),
        ]
        if seasons:
            cmd += ["--seasons", seasons]
        cmd.append("--add")

        subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        return {"success": True, "message": f"Regenerating {len(features)} features", "features": list(features)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _possible_features_getter():
    """Get all possible features that could be computed."""
    try:
        from bball.services.training_data import get_all_possible_features
        features = get_all_possible_features(no_player=False)
        return features
    except Exception as e:
        logger.error(f"Error getting possible features: {e}")
        return []


def _feature_column_adder(data, league=None):
    """Add feature columns to master training CSV."""
    try:
        from bball.features.parser import validate_feature_name
        from bball.services.training_data import get_all_possible_features

        features = data.get("features", [])
        if not features:
            return {"success": False, "error": "No features specified"}

        possible = get_all_possible_features(no_player=False)
        invalid = [f for f in features if f not in possible and not validate_feature_name(f)]
        if invalid:
            return {"success": False, "error": f"Invalid features: {invalid}"}

        cmd = [
            sys.executable, "-m", "bball.pipeline.training_pipeline",
            getattr(league, "league_id", "nba"),
            "--features", ",".join(features),
            "--add",
        ]
        subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        return {"success": True, "message": f"Adding {len(features)} columns", "features": features}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _available_seasons_getter_factory(league):
    """Factory that returns a callable to get available seasons."""
    def _getter():
        try:
            from bball.services.training_data import TrainingDataService
            svc = TrainingDataService(league=league)
            return svc.get_available_seasons()
        except Exception as e:
            logger.error(f"Error getting available seasons: {e}")
            return []
    return _getter


def _season_regenerator(data, league=None):
    """Regenerate master training for specific seasons."""
    try:
        seasons = data.get("seasons", [])
        if not seasons:
            return {"success": False, "error": "No seasons specified"}
        cmd = [
            sys.executable, "-m", "bball.pipeline.training_pipeline",
            getattr(league, "league_id", "nba"),
            "--seasons", ",".join(seasons),
            "--add",
        ]
        subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        return {"success": True, "message": f"Regenerating {len(seasons)} seasons"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _full_regenerator(data, league=None):
    """Full regeneration of master training CSV."""
    try:
        cmd = [
            sys.executable, "-m", "bball.pipeline.training_pipeline",
            getattr(league, "league_id", "nba"),
        ]
        min_season = data.get("min_season")
        if min_season:
            cmd += ["--min-season", min_season]
        subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        return {"success": True, "message": "Full regeneration started"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Elo / Cache helpers
# ---------------------------------------------------------------------------

def _elo_stats_getter_factory(db, league):
    def _getter():
        try:
            from bball.stats.elo_cache import get_elo_cache
            cache = get_elo_cache(db)
            stats = cache.get_stats(league=league)
            return {"success": True, **stats}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return _getter


def _elo_runner(data, league=None):
    try:
        from sportscore.services.jobs import create_job, complete_job, fail_job
        import threading

        job_id = create_job("elo_cache", league=league)
        seasons = data.get("seasons")

        def _run():
            try:
                from bball.stats.elo_cache import get_elo_cache
                from bball.mongo import Mongo
                _db = Mongo().db
                cache = get_elo_cache(_db)
                cache.compute(seasons=seasons, league=league)
                complete_job(job_id, "Elo cache computed successfully", league=league)
            except Exception as e:
                fail_job(job_id, str(e), league=league)

        threading.Thread(target=_run, daemon=True).start()
        return {"success": True, "job_id": job_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _elo_clearer(league=None):
    try:
        from bball.stats.elo_cache import get_elo_cache
        from bball.mongo import Mongo
        db = Mongo().db
        cache = get_elo_cache(db)
        cache.clear(league=league)
        return {"success": True, "message": "Elo cache cleared"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _cached_league_stats_getter_factory(db, league):
    def _getter():
        try:
            coll = league.collections.get("cached_league_stats", "cached_league_stats")
            docs = list(db[coll].find({}).sort("season", -1))
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            return {"success": True, "stats": docs}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return _getter


def _league_stats_cacher(data, league=None):
    try:
        from sportscore.services.jobs import create_job, complete_job, fail_job
        import threading

        season = data.get("season")
        job_id = create_job("cache_league_stats", league=league)

        def _run():
            try:
                from sportscore.cli.generic_commands import CacheLeagueStatsCommand
                import argparse
                args = argparse.Namespace(league=getattr(league, "league_id", "nba"), season=season)
                CacheLeagueStatsCommand().run(args)
                complete_job(job_id, "League stats cached", league=league)
            except Exception as e:
                fail_job(job_id, str(e), league=league)

        threading.Thread(target=_run, daemon=True).start()
        return {"success": True, "job_id": job_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _espn_db_auditor(data, league=None):
    try:
        from bball.data.espn_client import ESPNClient
        from sportscore.services.jobs import create_job, complete_job, fail_job
        import threading

        season = data.get("season")
        job_id = create_job("espn_audit", league=league)

        def _run():
            try:
                client = ESPNClient(league=league)
                result = client.audit_games(season=season)
                complete_job(job_id, f"Audit complete: {result}", league=league)
            except Exception as e:
                fail_job(job_id, str(e), league=league)

        threading.Thread(target=_run, daemon=True).start()
        return {"success": True, "job_id": job_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _espn_db_puller(data, league=None):
    try:
        from sportscore.services.jobs import create_job, complete_job, fail_job
        import threading

        game_ids = data.get("game_ids", [])
        job_id = create_job("espn_pull", league=league)

        def _run():
            try:
                from bball.data.espn_client import ESPNClient
                client = ESPNClient(league=league)
                for gid in game_ids:
                    client.pull_game(gid)
                complete_job(job_id, f"Pulled {len(game_ids)} games", league=league)
            except Exception as e:
                fail_job(job_id, str(e), league=league)

        threading.Thread(target=_run, daemon=True).start()
        return {"success": True, "job_id": job_id}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

def build_services(db, league) -> SportServices:
    """Build the SportServices container expected by sportscore shared_routes.

    Called once per league (cached by context.py).
    """
    from bball.features.sets import FEATURE_SETS, FEATURE_SET_DESCRIPTIONS

    config_manager = _get_config_manager(db, league)
    training_service = _get_training_service(db, league)
    market = make_market_helpers()
    market.prices_getter = _market_prices_getter

    return SportServices(
        model=ModelServices(
            config_manager=config_manager,
            training_service=training_service,
            points_trainer=_get_points_trainer(db, league),
            default_model_types=["LogisticRegression"],
            c_supported_models=["LogisticRegression", "SVC"],
            default_c_values=["0.1"],
        ),
        market=market,
        features=FeatureServices(
            feature_sets=FEATURE_SETS,
            feature_set_descriptions=FEATURE_SET_DESCRIPTIONS,
            available_features=get_available_features(league),
            master_training_csv=getattr(league, "master_training_csv", None),
            dependency_resolver=_feature_dependency_resolver,
            regenerator=_feature_regenerator,
            possible_getter=_possible_features_getter,
            column_adder=_feature_column_adder,
            available_seasons_getter=_available_seasons_getter_factory(league),
            season_regenerator=_season_regenerator,
            full_regenerator=_full_regenerator,
        ),
        elo=EloServices(
            stats_getter=_elo_stats_getter_factory(db, league),
            runner=_elo_runner,
            clearer=_elo_clearer,
            cached_league_stats_getter=_cached_league_stats_getter_factory(db, league),
            league_stats_cacher=_league_stats_cacher,
        ),
        data=DataServices(
            espn_db_auditor=_espn_db_auditor,
            espn_db_puller=_espn_db_puller,
        ),
        jobs=make_jobs_service(),
    )


def _get_points_trainer(db, league):
    """Get PointsRegressionTrainer (lazy import)."""
    try:
        from bball.models.points_regression import PointsRegressionTrainer
        return PointsRegressionTrainer(db=db)
    except Exception:
        return None
