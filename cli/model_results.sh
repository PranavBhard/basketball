#!/bin/bash
#
# Model Results CLI
#
# View model details, metrics, and feature importances.
#
# Usage:
#   ./cli/model_results.sh <league> --model <name_or_id>
#   ./cli/model_results.sh <league> --list
#
# Examples:
#   # View model by name
#   ./cli/model_results.sh nba --model "test_lr_model"
#
#   # View model by ID
#   ./cli/model_results.sh nba --model "67a1b2c3d4e5f6g7h8i9j0k1"
#
#   # List all models
#   ./cli/model_results.sh nba --list
#
#   # List trained models only
#   ./cli/model_results.sh nba --list --trained-only
#
#   # List ensemble models
#   ./cli/model_results.sh nba --list --ensemble-only
#
#   # Show all feature importances
#   ./cli/model_results.sh nba --model "test_lr_model" --all-features
#
# Options:
#   --model NAME_OR_ID      Model name or MongoDB _id
#   --list                  List available models
#   --trained-only          Only show trained models (with --list)
#   --ensemble-only         Only show ensemble models (with --list)
#   --top-features N        Number of top features to show (default: 20)
#   --all-features          Show all feature importances
#
# Output:
#   - Base models: Config details, metrics, feature importances (f-scores)
#   - Ensemble models: Meta-model details + base model names with their
#     individual feature importances

set -e

LEAGUE=${1:-nba}
shift 2>/dev/null || true

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment
source ./setup.sh

# Set Python path
export PYTHONPATH=/Users/pranav/Documents/basketball

# Run the model results script
python -m bball_app.cli.scripts.model_results "$LEAGUE" "$@"
