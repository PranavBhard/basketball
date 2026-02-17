# League-Aware Refactoring Plan

## Overview

Make the entire platform (core, web, cli, agents) fully league-aware by:
1. Renaming NBA-specific classes to generic sport-level names
2. Removing hardcoded NBA defaults throughout
3. Propagating league context through all code paths

## Current State

**Good news:** Repositories already support league-aware patterns via constructor parameters.
**Blockers:** Class names encode "NBA", many functions default to NBA when no league provided.

---

## Phase 1: Core Model Rename (Critical Path)

### 1.1 Rename `NBAModel` → `BballModel`

**File:** `core/models/nba_model.py` → `core/models/bball_model.py`

Changes:
- Rename class `NBAModel` to `BballModel`
- Rename file to `bball_model.py`
- Add `league` parameter to `__init__()`
- Pass league to repositories and stat handlers
- Update `import_collection()` call to use league-aware collection name

**Impact:** 22 files import this class - all need updated imports

### 1.2 Update All Importers

Files that import `NBAModel`:

**CLI Layer:**
- `cli/feature_selection_by_importance.py`
- `cli/populate_master_training_cols.py`
- `cli/train.py`
- `cli/master_training_data.py`

**Web Layer:**
- `web/app.py`

**Agents Layer:**
- `agents/tools/matchup_predict.py`
- `agents/tools/dataset_builder.py`
- `agents/tools/support_tools.py`
- `agents/tools/blend_experimenter.py`

**Tests:**
- `tests/test_mil_den_prediction_debug.py`
- `tests/test_prediction_workflow.py`
- `tests/test_prediction_with_player.py`
- `tests/test_feature_generation.py`
- `tests/test_training_workflow.py`
- `tests/test_master_training_workflow.py`

---

## Phase 2: Remove Hardcoded Defaults

### 2.1 StatHandlerV2 (`core/stats/handler.py`)

**Line 108:** `self.all_games = import_collection('stats_nba')`

Change to:
```python
def __init__(self, ..., league: LeagueConfig = None):
    self.league = league
    coll_name = league.collections["games"] if league else "stats_nba"
    self.all_games = import_collection(coll_name)
```

### 2.2 League Cache (`core/stats/league_cache.py`)

**7 functions** default to `load_league_config("nba")`:
- `get_cached_league_stats()`
- `cache_league_stats()`
- `get_league_stats_for_season()`
- `get_stats_for_date()`
- `get_all_cached_seasons()`
- `clear_cache_for_season()`
- `get_cache_stats()`

Change: Make `league` parameter required OR propagate from caller context.

### 2.3 Prediction Service (`core/services/prediction.py`)

**Lines 145, 191:** Remove fallback defaults:
```python
# Before
games_coll = self.league.collections["games"] if self.league else "stats_nba"

# After
games_coll = self.league.collections["games"]  # Require league
```

### 2.4 Model Factory (`core/models/factory.py`)

**Line 133:** `db.model_config_nba` hardcoded

Change to use league-aware collection access.

---

## Phase 3: Update import_collection Utility

**File:** `core/utils/collection.py`

Option A: Add league parameter
```python
def import_collection(coll_or_league, query=..., league: LeagueConfig = None):
    if league:
        coll = league.collections.get(coll_or_league, coll_or_league)
    else:
        coll = coll_or_league
    # ... rest of function
```

Option B: Create new function `import_league_collection(league, key, query)`

---

## Phase 4: Update CLI Entry Points

### 4.1 `cli/master_training_data.py`

**Line 300:** Pass league to BballModel:
```python
model = BballModel(
    classifier_features=all_features,
    league=league,  # ADD THIS
    ...
)
```

### 4.2 `cli/train.py`

Add `--league` argument and pass to model creation.

### 4.3 `cli/populate_master_training_cols.py`

Already has `--league` argument - ensure it's passed to model.

---

## Phase 5: Update Web App

### 5.1 `web/app.py`

- Update import: `from nba_app.core.models.bball_model import BballModel`
- Pass `g.league` to all `BballModel` instantiations
- Update type hints from `NBAModel` to `BballModel`

---

## Phase 6: Update Agents

### 6.1 All agent tools that use model

- `agents/tools/matchup_predict.py`
- `agents/tools/dataset_builder.py`
- `agents/tools/support_tools.py`
- `agents/tools/blend_experimenter.py`

Update imports and pass league context.

---

## Phase 7: Docstring Updates (Polish)

Update docstrings and comments to remove NBA-specific language:

- `core/features/registry.py` - "NBA Feature Definitions" → "Basketball Feature Definitions"
- `core/stats/per_calculator.py` - Update collection name references
- `core/utils/players.py` - Update references

---

## Execution Order

```
Phase 1.1  → Rename class/file (creates temp breakage)
Phase 1.2  → Fix all imports (fixes breakage)
Phase 2    → Remove hardcoded defaults (parallel tasks)
Phase 3    → Update import_collection
Phase 4    → CLI entry points
Phase 5    → Web app
Phase 6    → Agents
Phase 7    → Docstrings (low priority)
```

---

## Testing Strategy

1. After Phase 1: Run `python -c "from nba_app.core.models.bball_model import BballModel"` - should work
2. After Phase 2-4: Generate NBA training data - should work as before
3. After Phase 2-4: Generate CBB training data - should use CBB collections
4. Run existing test suite

---

## Backwards Compatibility

For transition period, add alias in old location:
```python
# core/models/nba_model.py (deprecated stub)
from nba_app.core.models.bball_model import BballModel as NBAModel
import warnings
warnings.warn("NBAModel is deprecated, use BballModel", DeprecationWarning)
```

---

## Estimated Scope

- **Files to modify:** ~30 files
- **Critical changes:** 5-6 files in core layer
- **Import updates:** 22 files (mechanical)
- **Risk:** Medium (touching core model)
