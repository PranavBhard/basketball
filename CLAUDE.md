# Project Instructions for Claude Code

## ⚠️ CRITICAL: Core Architecture Principles (READ FIRST)

This codebase uses a **strict layered architecture**. Violating these principles creates bugs, inconsistencies, and technical debt.

### The `core/` Layer is the Single Source of Truth (SSoT)

All core functionality lives in `bball_app/core/`:
- **Feature registry, calculations, and generation** (`core/features/`)
- **Training infrastructure** (`core/training/`) - ExperimentRunner, StackingTrainer, DatasetBuilder, RunTracker, schemas
- **Model training and prediction workflows** (`core/models/`, `core/services/prediction.py`)
- **Database access** (`core/mongo.py`, `core/data/`)
- **Shared utilities** (`core/stats/`, `core/utils/`)

### Consumer Layers (DO NOT put core logic here)

These layers **consume** `core/` - they do NOT implement core functionality themselves:
- `web/` - Web UI layer
- `agents/` - AI agent tooling layer
- `cli/` - Command-line interface layer

### Rules

1. **NEVER extend or modify legacy code outside `core/`** - Legacy files may exist in `cli/` or elsewhere. Do not touch them.
2. **NEVER duplicate core logic** - If functionality exists in `core/`, import and use it. Do not rewrite it elsewhere.
3. **All new core functionality goes in `core/`** - New features, calculations, or workflows belong in the core layer.
4. **Consumer layers are thin wrappers** - `web/`, `agents/`, and `cli/` should only handle I/O, formatting, and orchestration. The actual logic lives in `core/`.
5. **ALL cli scripts and web app views/endpoints are thin wrappers of core layer code** - They NEVER rewrite shared core code.
6. **League-driven architecture** - Assume the first argument for ALL `cli/scripts` is `<league>` and the script uses league config YAMLs (`leagues/`) to run dynamically for different leagues. Assume every web app view/page/core logic update uses the league YAML config.
### ⚠️ HARD STOPS - Do NOT Proceed If:

- **Writing sklearn/model training logic in `web/` or `cli/`**: Stop. This belongs in `core/services/` or `core/training/`.
- **Writing feature calculation logic outside `core/features/`**: Stop. Use the feature registry.
- **Writing database queries outside `core/` or repository classes**: Stop. Use repository patterns.

### What Belongs Where

| Logic Type | Belongs In | NOT In |
|------------|-----------|--------|
| Model training/evaluation | `core/services/training_service.py` | `web/app.py`, `cli/` |
| Model creation/instantiation | `core/training/` | Views, CLI scripts |
| Feature calculations | `core/features/`, `core/stats/` | Anywhere else |
| Database access | `core/data/` repositories | Views, CLI inline |
| Prediction workflow | `core/services/prediction.py` | Views |
| Config management | `core/services/config_manager.py` | Views |

### Before Writing Code, Ask:
- Does this functionality already exist in `core/`? → **Use it**
- Am I about to modify a file outside `core/` that has a `core/` equivalent? → **Stop. Use the `core/` version**
- Is this new core logic? → **Add it to `core/`, then call it from consumer layers**
- Am I writing training/fitting logic in a view? → **Stop. Put it in `core/services/`**

### Known Technical Debt (To Be Fixed)

These are known violations that need refactoring:

1. **Training logic inline in `web/app.py`** (around line 7300+)
   - StandardScaler, SelectKBest, train_test_split inline → Should call `TrainingService`
   - This is a future refactoring to move evaluation loops to TrainingService

2. **See `FUTURE_PLANS.md`** for planned improvements to player feature calculations (cross-team stats for traded players, training/prediction parity).

---

## Python Environment
- Always run `source venv/bin/activate` before executing Python commands
- Use `python` (not `python3`) after activating the venv
- Set `PYTHONPATH=/Users/pranav/Documents/basketball` when running scripts that import `bball_app`

## MongoDB Access
- run `. ./setup.sh` but DO NOT read that file EVER (note: this also activates the venv)
- Use `from bball_app.core.mongo import Mongo` to connect to the database
- The Mongo wrapper handles connection details automatically

## ⚠️ Data Filtering Rules

**CRITICAL: All stat calculations and training data MUST exclude certain game types (preseason, allstar, etc.).**

### Single Source of Truth: League Config YAML

The excluded game types are defined in each league's YAML config file (`leagues/nba.yaml`, `leagues/cbb.yaml`):

```yaml
# leagues/nba.yaml
season:
  exclude_game_types:
    - preseason
    - allstar
```

**NEVER hardcode `['preseason', 'allstar']`** - always read from the league config.

### How to Access

**In `core/` classes** (StatHandler, PERCalculator, etc.):
```python
# Use the _exclude_game_types property
query['game_type'] = {'$nin': self._exclude_game_types}
```

**In `web/app.py`** (Flask context):
```python
league_config = getattr(g, "league", None)
exclude_game_types = league_config.exclude_game_types if league_config else ['preseason', 'allstar']
query['game_type'] = {'$nin': exclude_game_types}
```

### When This Filter is Required

- Feature calculations
- Training data generation
- Player stat aggregations (PPG, APG, RPG, PER, etc.)
- Team record calculations
- Any statistical analysis

### Examples

```python
# CORRECT - using league config in web layer
league_config = getattr(g, "league", None)
exclude_game_types = league_config.exclude_game_types if league_config else ['preseason', 'allstar']
player_games = db[player_stats_collection].find({
    'player_id': player_id,
    'season': season,
    'game_type': {'$nin': exclude_game_types}
})

# CORRECT - using _exclude_game_types in core classes
query['game_type'] = {'$nin': self._exclude_game_types}

# WRONG - hardcoded list
query['game_type'] = {'$nin': ['preseason', 'allstar']}

# WRONG - missing game_type filter entirely
player_games = db.nba_player_stats.find({
    'player_id': player_id,
    'season': season
})
```

## Common Commands
```bash
# Activate venv and run Python
source venv/bin/activate && python script.py

# Run with proper PYTHONPATH
source venv/bin/activate && PYTHONPATH=/Users/pranav/Documents/basketball python script.py
```
