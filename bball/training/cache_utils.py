"""Cache utilities — delegates to sportscore with basketball-specific defaults."""
from sportscore.training.cache_utils import (
    get_best_config,
    read_csv_safe,
    load_model_cache as _load_model_cache,
    save_model_cache as _save_model_cache,
    get_latest_training_csv as _get_latest_training_csv,
)
from bball.training.constants import MODEL_CACHE_FILE, MODEL_CACHE_FILE_NO_PER, OUTPUTS_DIR


def load_model_cache(no_per: bool = False) -> dict:
    """Load cached model configurations."""
    cache_file = MODEL_CACHE_FILE_NO_PER if no_per else MODEL_CACHE_FILE
    return _load_model_cache(cache_file)


def save_model_cache(cache: dict, no_per: bool = False):
    """Save model configurations to cache."""
    cache_file = MODEL_CACHE_FILE_NO_PER if no_per else MODEL_CACHE_FILE
    _save_model_cache(cache, cache_file)


def get_latest_training_csv(output_dir: str = OUTPUTS_DIR, no_per: bool = False) -> str:
    """Find the most recent classifier training CSV file."""
    return _get_latest_training_csv(output_dir, no_per=no_per)


__all__ = ['load_model_cache', 'save_model_cache', 'get_best_config', 'get_latest_training_csv', 'read_csv_safe']
