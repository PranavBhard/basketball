"""
Model Evaluation

Functions for evaluating classifier models with cross-validation
and time-based calibration.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from bball_app.core.training.model_factory import create_model_with_c


def evaluate_model_combo(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    c_value: float = None,
    n_splits: int = 5
) -> dict:
    """
    Evaluate a model/C-value combination using time-series cross-validation.

    Args:
        X: Feature matrix (scaled)
        y: Target vector
        model_type: Model type (e.g., 'LogisticRegression')
        c_value: C-value for regularization (optional)
        n_splits: Number of CV splits

    Returns:
        Dict with accuracy, std, log_loss, brier, etc., including per-fold metrics
    """
    model = create_model_with_c(model_type, c_value)

    # Time-series split for CV
    tscv = TimeSeriesSplit(n_splits=n_splits)

    accuracies = []
    log_losses = []
    briers = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_copy = create_model_with_c(model_type, c_value)
        model_copy.fit(X_train, y_train)

        # Predictions
        y_pred = model_copy.predict(X_val)
        y_proba = model_copy.predict_proba(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred) * 100
        ll = log_loss(y_val, y_proba)
        brier = brier_score_loss(y_val, y_proba[:, 1])

        accuracies.append(acc)
        log_losses.append(ll)
        briers.append(brier)

    # Always include fold-level metrics
    result = {
        'model_type': model_type,
        'c_value': c_value,
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'accuracy_folds': accuracies,
        'log_loss_mean': float(np.mean(log_losses)),
        'log_loss_std': float(np.std(log_losses)),
        'log_loss_folds': log_losses,
        'brier_mean': float(np.mean(briers)),
        'brier_std': float(np.std(briers)),
        'brier_folds': briers,
        'n_folds': n_splits
    }

    return result


def evaluate_model_combo_with_calibration(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    y: np.ndarray,
    model_type: str,
    c_value: float = None,
    calibration_method: str = 'isotonic',
    calibration_years: list = None,
    evaluation_year: int = None,
    logger=None
) -> dict:
    """
    Evaluate a model with time-based calibration using year-based temporal splits.

    Time-based calibration splits data chronologically by year:
    - Train: All data before first calibration_year (e.g., < 2022)
    - Calibrate: Data from calibration_years (e.g., 2022, 2023) - combined
    - Evaluate: Data from evaluation_year (e.g., 2024)

    This is different from the non-calibrated version which uses TimeSeriesSplit CV.
    The key difference is that calibration learns to adjust probabilities on recent
    but not latest data, then evaluates on the most recent data.

    Args:
        df: DataFrame with Year, Month, Day columns for sorting
        X_scaled: Scaled feature matrix (must match df rows)
        y: Target vector (must match df rows)
        model_type: Model type (e.g., 'LogisticRegression')
        c_value: C-value for regularization (optional)
        calibration_method: 'isotonic' or 'sigmoid'
        calibration_years: List of years to use for calibration (e.g., [2022, 2023])
        evaluation_year: Year to use for evaluation (e.g., 2024)
        logger: Logger instance for logging

    Returns:
        Dict with accuracy, std, log_loss, brier, etc.
    """
    from sklearn.calibration import IsotonicRegression
    from sklearn.linear_model import LogisticRegression as LR

    # Handle backward compatibility: if single calibration_year passed, convert to list
    if calibration_years is None:
        raise ValueError("calibration_years must be specified for time-based calibration")

    # Ensure calibration_years is a list
    if not isinstance(calibration_years, list):
        calibration_years = [calibration_years]

    if logger:
        logger.info(f"Using time-based calibration (method: {calibration_method})")
        logger.info(f"Calibration years: {calibration_years}, Evaluation year: {evaluation_year}")

    # Create date column for sorting and derive SeasonStartYear (e.g., 2010-2011 -> 2010)
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy[['Year', 'Month', 'Day']])
    df_copy['SeasonStartYear'] = np.where(df_copy['Month'] >= 10, df_copy['Year'], df_copy['Year'] - 1)

    if evaluation_year is None:
        raise ValueError("evaluation_year must be specified for time-based calibration")

    # Interpret provided years as season start years
    cal_seasons = [int(y) for y in calibration_years]
    eval_season = int(evaluation_year)
    # Get the earliest calibration season to determine training cutoff
    earliest_cal_year = min(cal_seasons)

    # Split by year boundaries:
    # Train: All data before earliest calibration_year (Year < earliest_cal_year)
    # Calibrate: Data from all calibration_years (Year in calibration_years) - combined
    # Evaluate: Data from evaluation_year (Year == evaluation_year)

    # Training set: all data before earliest calibration season
    train_mask = df_copy['SeasonStartYear'] < earliest_cal_year
    X_train = X_scaled[train_mask]
    y_train = y[train_mask]

    # Calibration set: data from all calibration seasons (combined)
    cal_mask = df_copy['SeasonStartYear'].isin(cal_seasons)
    X_cal = X_scaled[cal_mask]
    y_cal = y[cal_mask]

    # Evaluation set: data from evaluation season
    eval_mask = df_copy['SeasonStartYear'] == eval_season
    X_eval = X_scaled[eval_mask]
    y_eval = y[eval_mask]

    if logger:
        logger.info(f"Train set: {len(X_train)} games (SeasonStartYear < {earliest_cal_year})")
        logger.info(f"Calibration set: {len(X_cal)} games (Seasons in {cal_seasons})")
        logger.info(f"Evaluation set: {len(X_eval)} games (SeasonStartYear == {eval_season})")

        if len(X_train) == 0:
            logger.warning(f"Training set is empty! Check begin_year and calibration_years. Earliest cal year: {earliest_cal_year}")
        if len(X_cal) == 0:
            logger.warning(f"Calibration set is empty! Check calibration_years: {calibration_years}")
        if len(X_eval) == 0:
            logger.warning(f"Evaluation set is empty! Check evaluation_year: {evaluation_year}")

    # Train model on training set
    model = create_model_with_c(model_type, c_value)
    model.fit(X_train, y_train)

    # Get raw predictions on calibration set
    y_proba_raw_cal = model.predict_proba(X_cal)

    # Fit calibrator on calibration set
    if calibration_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_proba_raw_cal[:, 1], y_cal)
        if logger:
            logger.info("Fitted isotonic calibrator on calibration set")
    elif calibration_method == 'sigmoid':
        # Use sigmoid (Platt scaling) - create wrapper to avoid cv='prefit' compatibility issues
        sigmoid_calibrator = LR()
        sigmoid_calibrator.fit(y_proba_raw_cal[:, 1].reshape(-1, 1), y_cal)

        class SigmoidCalibratedModel:
            def __init__(self, base_model, calibrator):
                self.base_model = base_model
                self.calibrator = calibrator

            def predict(self, X):
                return self.base_model.predict(X)

            def predict_proba(self, X):
                raw_proba = self.base_model.predict_proba(X)
                calibrated_1 = self.calibrator.predict_proba(raw_proba[:, 1].reshape(-1, 1))[:, 1]
                # Ensure probabilities sum to 1
                calibrated_1 = np.clip(calibrated_1, 0.0, 1.0)
                return np.column_stack([1 - calibrated_1, calibrated_1])

        calibrator = SigmoidCalibratedModel(model, sigmoid_calibrator)
        if logger:
            logger.info("Fitted sigmoid calibrator on calibration set")
    else:
        if logger:
            logger.warning(f"Unknown calibration method: {calibration_method}, using raw probabilities")
        calibrator = None

    # Get raw predictions on evaluation set
    y_proba_raw_eval = model.predict_proba(X_eval)

    # Apply calibration to evaluation set
    if calibrator is not None:
        if calibration_method == 'isotonic':
            y_proba_calibrated = np.column_stack([
                1 - calibrator.predict(y_proba_raw_eval[:, 1]),
                calibrator.predict(y_proba_raw_eval[:, 1])
            ])
        else:  # sigmoid
            y_proba_calibrated = calibrator.predict_proba(X_eval)
    else:
        y_proba_calibrated = y_proba_raw_eval

    # Calculate predictions and metrics on evaluation set
    y_pred_calibrated = (y_proba_calibrated[:, 1] >= 0.5).astype(int)

    acc = accuracy_score(y_eval, y_pred_calibrated) * 100
    ll = log_loss(y_eval, y_proba_calibrated)
    brier = brier_score_loss(y_eval, y_proba_calibrated[:, 1])

    if logger:
        logger.info(f"Calibrated results on season starting {eval_season} - Accuracy: {acc:.2f}%, Log Loss: {ll:.4f}, Brier: {brier:.4f}")

    # Return in same format as evaluate_model_combo (single evaluation, so std=0)
    return {
        'model_type': model_type,
        'c_value': c_value,
        'accuracy_mean': float(acc),
        'accuracy_std': 0.0,  # Single evaluation, no std
        'accuracy_folds': [acc],
        'log_loss_mean': float(ll),
        'log_loss_std': 0.0,
        'log_loss_folds': [ll],
        'brier_mean': float(brier),
        'brier_std': 0.0,
        'brier_folds': [brier],
        'n_folds': 1,  # For compatibility, but this is a single split, not cross-validation
        'split_type': 'time_based_calibration',  # Indicates this is time-based split, not CV folds
        'calibration_years': calibration_years,
        'evaluation_year': evaluation_year,
        'train_set_size': len(X_train),
        'calibrate_set_size': len(X_cal),
        'eval_set_size': len(X_eval),
    }


def compute_feature_importance(model, feature_names: list, model_type: str) -> dict:
    """
    Compute feature importance for a trained model.

    Args:
        model: Trained sklearn model
        feature_names: List of feature names
        model_type: Type of model

    Returns:
        Dict mapping feature names to importance scores
    """
    importance = {}

    if model_type == 'GradientBoosting':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, imp in zip(feature_names, importances):
                importance[name] = float(imp)
    elif model_type == 'RandomForest':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, imp in zip(feature_names, importances):
                importance[name] = float(imp)
    elif model_type == 'LogisticRegression':
        # Use absolute coefficient values as importance proxy
        if hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_[0])
            for name, coef in zip(feature_names, coefs):
                importance[name] = float(coef)

    return importance
