#!/bin/bash
#
# Train Base Model CLI
#
# Trains a base classifier model (LogisticRegression, GradientBoosting, etc.)
#
# Usage:
#   ./cli/train_base_model.sh <league> [options]
#
# Examples:
#   ./cli/train_base_model.sh nba --model LR --c-value 0.1 \
#       --train-seasons 2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \
#       --calibration-seasons 2023 --evaluation-seasons 2024 \
#       --features "points|season|avg|diff,assists|season|avg|diff" \
#       --name "test_lr_model"
#
# Options:
#   --model TYPE            Model type: LR, GB, SVM (required)
#   --c-value FLOAT         Regularization parameter (default: 0.1)
#   --time-method METHOD    Calibration method: sigmoid, isotonic (default: sigmoid)
#   --train-seasons YEARS   Comma-separated training seasons (required)
#   --calibration-seasons YEARS  Comma-separated calibration seasons (required)
#   --evaluation-seasons YEAR    Evaluation season year (required)
#   --features LIST         Comma-separated feature names (required)
#   --name NAME             Model name (auto-generated if not provided)
#   --min-games N           Minimum games played filter (default: 20)
#   --include-injuries      Include injury features
#   --no-master             Do not use master training CSV
#
# Model Types:
#   LR  = LogisticRegression
#   GB  = GradientBoosting
#   SVM = Support Vector Machine

set -e

LEAGUE=${1:-nba}
shift 2>/dev/null || true

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment
source ./setup.sh

# Set Python path
export PYTHONPATH=/Users/pranav/Documents/basketball

# Run the training script
python -m bball_app.cli.scripts.train_base_model "$LEAGUE" "$@"
