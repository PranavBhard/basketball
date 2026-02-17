"""
Cache Utilities

Functions for managing model configuration caches and training data files.
"""

import os
import glob
import json
import csv

import pandas as pd

from bball_app.core.training.constants import (
    MODEL_CACHE_FILE,
    MODEL_CACHE_FILE_NO_PER,
    OUTPUTS_DIR,
)


def get_latest_training_csv(output_dir: str = OUTPUTS_DIR, no_per: bool = False) -> str:
    """
    Find the most recent classifier training CSV file.

    Args:
        output_dir: Directory to search for CSV files
        no_per: If True, look for files with 'no_per' in the name
                If False, look for files without 'no_per' in the name

    Returns:
        Path to the most recent training CSV file, or None if not found
    """
    pattern = os.path.join(output_dir, 'classifier_training_*.csv')
    csv_files = glob.glob(pattern)

    # Filter out standardized versions
    csv_files = [f for f in csv_files if '_standardized' not in f]

    # Filter based on no_per flag
    if no_per:
        csv_files = [f for f in csv_files if 'no_per' in f]
    else:
        csv_files = [f for f in csv_files if 'no_per' not in f]

    if not csv_files:
        return None

    # Sort by modification time, most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]


def load_model_cache(no_per: bool = False) -> dict:
    """
    Load cached model configurations.

    Args:
        no_per: If True, load cache without PER features

    Returns:
        Dict with 'configs', 'best', 'timestamp' keys
    """
    cache_file = MODEL_CACHE_FILE_NO_PER if no_per else MODEL_CACHE_FILE
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {'configs': [], 'best': None, 'timestamp': None}


def save_model_cache(cache: dict, no_per: bool = False):
    """
    Save model configurations to cache.

    Args:
        cache: Dict with model configurations
        no_per: If True, save to no-PER cache file
    """
    cache_file = MODEL_CACHE_FILE_NO_PER if no_per else MODEL_CACHE_FILE
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def get_best_config(cache: dict) -> dict:
    """
    Get the best model configuration from cache based on accuracy.

    Args:
        cache: Dict with 'configs' list

    Returns:
        Best config dict, or None if no configs
    """
    if not cache.get('configs'):
        return None

    # Sort by accuracy, then by lower log_loss as tiebreaker
    sorted_configs = sorted(
        cache['configs'],
        key=lambda x: (-x['accuracy_mean'], x['log_loss_mean'])
    )
    return sorted_configs[0]


def read_csv_safe(csv_path: str) -> pd.DataFrame:
    """
    Read CSV file with error handling for trailing empty columns.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with trailing empty columns removed
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Remove trailing empty columns from header
        while header and (not header[-1] or header[-1].strip() == ''):
            header.pop()
        expected_cols = len(header)

        for row_num, row in enumerate(reader, start=2):
            # Trim row to expected length (remove trailing empty columns)
            if len(row) > expected_cols:
                # Remove trailing empty columns
                row = row[:expected_cols]
            elif len(row) < expected_cols:
                # Pad with empty strings if row is too short
                row.extend([''] * (expected_cols - len(row)))

            # Only add rows that match expected column count
            if len(row) == expected_cols:
                rows.append(row)
            else:
                # Log warning for rows that still don't match (shouldn't happen after trimming)
                print(f"Warning: Row {row_num} has {len(row)} columns, expected {expected_cols}. Skipping.")

    # Create DataFrame
    df = pd.DataFrame(rows, columns=header)

    # Convert numeric columns
    for col in df.columns:
        if col not in ['Home', 'Away']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    # Remove any trailing empty columns (Unnamed columns)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')

    return df
