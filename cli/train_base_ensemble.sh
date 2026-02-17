#!/bin/bash
#
# Train Base Ensemble CLI
#
# Trains an ensemble (stacking) model that combines multiple base models.
#
# IMPORTANT: Temporal parameters (begin_year, calibration_years, evaluation_year)
# are DERIVED from the base models. All base models must have identical temporal
# configurations. This ensures platform-wide consistency in how time-calibration
# is handled for meta-model training.
#
# Usage:
#   ./cli/train_base_ensemble.sh <league> [options]
#
# Examples:
#   ./cli/train_base_ensemble.sh nba --model LR --c-value 0.1 \
#       --models "model1_name,model2_name" --extra-features "pred_margin"
#
#   ./cli/train_base_ensemble.sh nba --model LR --models "model1,model2,model3" \
#       --use-disagree --use-conf
#
# Options:
#   --model TYPE            Meta-model type: LR, GB, SVM (required)
#   --c-value FLOAT         Regularization parameter for meta-model (default: 0.1)
#   --models LIST           Comma-separated base model names or IDs (required)
#   --extra-features LIST   Comma-separated additional features (e.g., pred_margin)
#   --stacking-mode MODE    Stacking mode: naive, informed (default: informed)
#   --use-disagree          Include pairwise disagreement features
#   --use-conf              Include confidence features
#
# Model Types:
#   LR  = LogisticRegression
#   GB  = GradientBoosting
#   SVM = Support Vector Machine
#
# Stacking Modes:
#   naive    = Use only base model predictions
#   informed = Include additional features (disagree, conf, meta-features)
#
# Temporal Configuration:
#   The meta-model trains on calibration_years using OOF predictions from base
#   models, and evaluates on evaluation_year. These values are derived from the
#   base models' configs - no CLI override is allowed to ensure consistency.

set -e

LEAGUE=${1:-nba}
shift 2>/dev/null || true

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment
source ./setup.sh

# Set Python path
export PYTHONPATH=/Users/pranav/Documents/basketball

# Run the ensemble training script
python -m bball_app.cli.scripts.train_base_ensemble "$LEAGUE" "$@"
