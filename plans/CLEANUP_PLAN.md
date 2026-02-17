# Legacy Code Cleanup Plan

## Executive Summary

**Goal:** Migrate all core business logic to `core/`, leaving `cli/` and `agents/tools/` as thin orchestration wrappers.

**Current State:**
| Layer | Lines | Files | Status |
|-------|-------|-------|--------|
| `core/` | ~18,500 | 23 | SSoT (complete) |
| `cli/` | ~10,200 | 19 | Mixed (logic + orchestration) |
| `agents/tools/` | ~6,400 | 13 | Mostly clean, some cli/ dependencies |

**Estimated Total Effort:** 20-30 hours of focused work

---

## Phase 1: Critical Path (Highest Impact)

These migrations unblock everything else. Complete these first.

### 1.1 Create `core/csv_utils.py` (NEW FILE)
**Effort:** LOW (1-2 hours)
**Why First:** Multiple files depend on `read_csv_safe()` from `cli/train.py`. Extract this first.

**Extract from `cli/train.py`:**
```python
# core/csv_utils.py
def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV with proper dtype handling for feature columns."""
    ...
```

**Files that will use this:**
- `agents/tools/stacking_tool.py`
- `agents/tools/experiment_runner.py`
- `agents/tools/blend_experimenter.py`
- `cli/train.py` (after refactor)

---

### 1.2 Migrate `cli/master_training_data.py` → `core/master_training_manager.py`
**Effort:** MEDIUM (2-3 hours)
**Lines:** 1,045
**Why:** Used by 6+ downstream modules. Pure business logic, no CLI code.

**Functions to migrate:**
- `get_all_possible_features()` - Feature enumeration from FeatureRegistry
- `generate_master_training_data()` - Builds master CSV from MongoDB
- `update_master_training_data_incremental()` - Incremental updates
- `extract_features_from_master()` - Feature extraction helper
- `create_or_update_master_metadata()` - MongoDB metadata management
- `register_existing_master_csv()` - Registration helper
- `MASTER_TRAINING_PATH` constant

**Post-migration:**
- Delete `cli/master_training_data.py`
- Update all imports:
  - `cli/generate_master_training.py`
  - `cli/populate_master_training_cols.py`
  - `cli/recover_master_training.py`
  - `cli/register_master_csv.py`
  - `agents/tools/dataset_builder.py`
  - `agents/tools/dataset_augmenter.py`
  - `agents/tools/support_tools.py`

---

### 1.3 Migrate `cli/train.py` training logic → `core/classifier_trainer.py` (NEW FILE)
**Effort:** HIGH (4-6 hours)
**Lines:** 2,072 (extract ~1,800)
**Why:** Core classification training logic. Used by `agents/tools/experiment_runner.py`, `stacking_tool.py`, `blend_experimenter.py`.

**Functions to migrate to `core/classifier_trainer.py`:**
```python
# Core training functions
create_model_with_c(model_type, c_value, ...)
evaluate_model_combo(X_train, y_train, X_test, y_test, ...)
evaluate_model_combo_with_calibration(...)

# Feature analysis
compute_feature_importance(model, feature_names)
compute_correlation_by_sets(df, feature_sets)

# Model caching
load_model_cache(cache_path)
save_model_cache(model, scaler, features, cache_path)

# Data utilities
get_latest_training_csv()
```

**Keep in `cli/train.py` (thin wrapper):**
```python
# CLI entry points only
def train_mode(): ...      # Parses args, calls core
def predict_mode(): ...    # Parses args, calls core
def ablation_mode(): ...   # Parses args, calls core
def layer_test_mode(): ... # Parses args, calls core

if __name__ == "__main__":
    main()
```

**Post-migration updates:**
- `agents/tools/experiment_runner.py` → import from `core/classifier_trainer`
- `agents/tools/stacking_tool.py` → import from `core/classifier_trainer`
- `agents/tools/blend_experimenter.py` → import from `core/classifier_trainer`

---

### 1.4 Migrate `cli/point_prediction_cache.py` → `core/point_prediction_cache.py`
**Effort:** LOW (1 hour)
**Lines:** 337
**Why:** Pure data management class, no CLI code. Used by agents.

**Class to migrate:**
```python
class PointPredictionCache:
    def cache_predictions(...)
    def get_predictions(...)
    def delete_predictions(...)
```

**Post-migration updates:**
- `agents/tools/experiment_runner.py`
- `agents/tools/dataset_builder.py`

---

## Phase 2: Secondary Migrations

Complete after Phase 1.

### 2.1 Migrate `cli/master_training_data_points.py` → `core/points_master_training_manager.py`
**Effort:** MEDIUM (2 hours)
**Lines:** 489
**Why:** Parallel structure to master_training_manager.py. Pure business logic.

**Functions to migrate:**
- `generate_master_training_points_data()`
- `update_master_training_points_data_incremental()`
- `extract_features_from_master_points()`
- `create_or_update_master_points_metadata()`
- `MASTER_POINTS_TRAINING_PATH` constant

---

### 2.2 Migrate `cli/points_regression_features.py` → `core/points_feature_sets.py`
**Effort:** LOW (30 min)
**Lines:** 192
**Why:** Static feature configuration. Should live with other feature sets.

**Content to migrate:**
- `ALL_POINTS_FEATURES` list
- `POINTS_FEATURE_SETS` dict
- Validation functions

---

### 2.3 Migrate `cli/update_rosters.py` → `core/roster_manager.py`
**Effort:** LOW (1 hour)
**Lines:** 311
**Why:** Pure business logic for roster building.

**Functions to migrate:**
- `update_rosters()`
- `get_position_category()`

---

### 2.4 Extract feature column calculation from `cli/populate_master_training_cols.py`
**Effort:** HIGH (3-4 hours)
**Lines:** 1,402 (extract ~800)
**Why:** Core feature calculation logic with threading. Keep CLI orchestration separate.

**Create `core/feature_calculator.py`:**
```python
def calculate_feature_columns_chunked(df, features, num_threads=4):
    """Multi-threaded feature calculation."""
    ...

def calculate_feature_column(df, feature_name, model):
    """Single feature calculation."""
    ...

def extract_metadata_from_mongodb(game_ids):
    """Fetch game metadata."""
    ...
```

**Keep in `cli/populate_master_training_cols.py`:**
- argparse entry point
- Job progress tracking (MongoDB)
- Orchestration logic

---

## Phase 3: Agent Tools Cleanup

Complete after Phase 1-2 (dependencies resolved).

### 3.1 Extract duplicated feature mapping logic → `core/feature_set_mapper.py` (NEW FILE)
**Effort:** MEDIUM (2 hours)
**Why:** Both `dataset_builder.py` and `support_tools.py` implement the same mapping.

**Create:**
```python
# core/feature_set_mapper.py
def map_feature_blocks_to_master_features(blocks: List[str], master_features: List[str]) -> List[str]:
    """Maps feature block names to actual master CSV feature names."""
    ...

def get_features_for_block(block_name: str) -> List[str]:
    """Returns all features belonging to a semantic block."""
    ...
```

**Update:**
- `agents/tools/dataset_builder.py` - remove `_map_master_features_to_blocks()`
- `agents/tools/support_tools.py` - remove duplicate mapping logic

---

### 3.2 Update agent tool imports
**Effort:** LOW (1-2 hours)
**Why:** Point all tools to `core/` instead of `cli/`.

**Files to update:**

| File | Current Import | New Import |
|------|---------------|------------|
| `experiment_runner.py` | `cli.train` | `core.classifier_trainer` |
| `experiment_runner.py` | `cli.point_prediction_cache` | `core.point_prediction_cache` |
| `stacking_tool.py` | `cli.train.read_csv_safe` | `core.csv_utils` |
| `blend_experimenter.py` | `cli.train` | `core.classifier_trainer` |
| `dataset_builder.py` | `cli.master_training_data` | `core.master_training_manager` |
| `dataset_augmenter.py` | `cli.master_training_data` | `core.master_training_manager` |
| `support_tools.py` | `cli.master_training_data` | `core.master_training_manager` |

---

### 3.3 Fix `blend_experimenter.py` internal coupling
**Effort:** LOW (1 hour)
**Why:** Uses `model.stat_handler._calculate_net_feature()` (internal API).

**Fix:** Use `core/feature_generator.py` public API instead.

---

## Phase 4: Cleanup & Deletion

### 4.1 Delete deprecated files
- `cli/pull_matchups.py` - Functionality exists in `espn_api.py`

### 4.2 Files to KEEP in `cli/` (thin wrappers / specialized tools)

| File | Reason |
|------|--------|
| `train.py` | CLI entry point (after refactor) |
| `generate_master_training.py` | CLI orchestration |
| `populate_master_training_cols.py` | CLI orchestration (after refactor) |
| `espn_api.py` | Data collection (external API) |
| `espn_nba.py` | Data collection orchestration |
| `espn_nba_players.py` | Data collection (scraping) |
| `cache_elo_ratings.py` | Caching utility |
| `feature_selection_by_importance.py` | Experimental analysis |
| `recover_master_training.py` | Maintenance utility |
| `remove_duplicate_columns.py` | Maintenance utility |
| `register_master_csv.py` | Admin utility |
| `check_zero_columns.py` | Diagnostic utility |

---

## Dependency Graph (Migration Order)

```
Phase 1 (do first, in order):
  1.1 core/csv_utils.py (NEW) ─────────────────────┐
                                                    │
  1.2 core/master_training_manager.py ◄────────────┤
       (from cli/master_training_data.py)          │
                                                    │
  1.3 core/classifier_trainer.py (NEW) ◄───────────┤
       (from cli/train.py)                         │
                                                    │
  1.4 core/point_prediction_cache.py ◄─────────────┘
       (from cli/point_prediction_cache.py)

Phase 2 (after Phase 1):
  2.1 core/points_master_training_manager.py
  2.2 core/points_feature_sets.py
  2.3 core/roster_manager.py
  2.4 core/feature_calculator.py

Phase 3 (after Phase 2):
  3.1 core/feature_set_mapper.py (NEW)
  3.2 Update all agent tool imports
  3.3 Fix blend_experimenter coupling

Phase 4 (final):
  4.1 Delete deprecated files
  4.2 Verify all tests pass
```

---

## New `core/` Files After Cleanup

```
core/
├── __init__.py (update exports)
├── csv_utils.py (NEW)
├── classifier_trainer.py (NEW)
├── master_training_manager.py (NEW - from cli/)
├── points_master_training_manager.py (NEW - from cli/)
├── point_prediction_cache.py (NEW - from cli/)
├── points_feature_sets.py (NEW - from cli/)
├── roster_manager.py (NEW - from cli/)
├── feature_calculator.py (NEW - extracted from cli/)
├── feature_set_mapper.py (NEW)
└── ... (existing files)
```

---

## Verification Checklist

After each phase, verify:

- [ ] All imports resolve (no `ModuleNotFoundError`)
- [ ] `cli/` files only import from `core/`, never vice versa
- [ ] `agents/tools/` files only import from `core/`, not `cli/`
- [ ] Run existing tests: `pytest tests/`
- [ ] Web app still works: `python web/app.py`
- [ ] Agent tools still work (manual test)

---

## Risk Mitigation

1. **Circular imports:** Always have `cli/` and `agents/` import from `core/`, never the reverse.

2. **Breaking changes:** Keep function signatures identical during migration. Only rename modules.

3. **Testing:** Run tests after each file migration, not just at the end.

4. **Rollback:** Use git branches. One branch per phase.

---

## Summary

| Phase | Files Changed | New Files | Effort |
|-------|--------------|-----------|--------|
| 1 | 8 | 2 | 8-12 hours |
| 2 | 5 | 4 | 6-8 hours |
| 3 | 6 | 1 | 4-5 hours |
| 4 | 2 | 0 | 1-2 hours |
| **Total** | **21** | **7** | **20-30 hours** |

---

## Recommended Approach

1. **Work incrementally:** Complete one migration, test, commit, then move to next.
2. **Phase 1 is critical:** Everything else depends on it.
3. **Don't optimize prematurely:** Migrate first, refactor later.
4. **Keep old files temporarily:** Use deprecation warnings before deletion.

Example deprecation pattern:
```python
# cli/master_training_data.py (temporary, after migration)
import warnings
warnings.warn(
    "cli.master_training_data is deprecated. Use core.master_training_manager instead.",
    DeprecationWarning,
    stacklevel=2
)
from core.master_training_manager import *  # Re-export for compatibility
```
