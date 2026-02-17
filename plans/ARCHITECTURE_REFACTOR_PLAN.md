# Core Layer Architecture Refactoring Plan

## Executive Summary

This plan addresses organizational inconsistencies in `core/` by:
1. **Deleting `agents/tools/`** entirely - legacy agent infrastructure that duplicates core
2. **Consolidating training infrastructure** into `core/training/`
3. **Clarifying the Model Pipeline** with distinct phases
4. **Standardizing service patterns** (GameService class)

**Key Principle**: Training data generation and model training are **two distinct processes**. Model training should NEVER entail generating features from historical data.

---

## Current State Analysis

### The `agents/tools/` Problem

The `agents/tools/` directory was created for an old agent to have its own training/prediction infrastructure. It's confusing because it duplicates `core/` functionality.

**Files in `agents/tools/` (20 files, ~9,000 lines):**

| File | Lines | Disposition |
|------|-------|-------------|
| `experiment_runner.py` | 847 | → MOVE to `core/training/` |
| `stacking_tool.py` | 1095 | → MOVE to `core/training/` |
| `run_tracker.py` | 225 | → MOVE to `core/training/` |
| `dataset_builder.py` | 729 | → MOVE to `core/training/` |
| `support_tools.py` | 1490 | → DELETE (agent-specific) |
| `news_tools.py` | 804 | → MOVE to `core/services/` |
| `model_inspector_tools.py` | 794 | → DELETE (agent-specific) |
| `team_game_window_tools.py` | 616 | → DELETE (use core/stats/) |
| `blend_experimenter.py` | 466 | → DELETE (deprecated) |
| `code_executor.py` | 363 | → DELETE (agent-specific) |
| `window_player_stats_tools.py` | 337 | → DELETE (use core/stats/) |
| `matchup_predict.py` | 283 | → MOVE to `core/services/` |
| `game_tools.py` | 250 | → DELETE (use core/services/game_service) |
| `player_stats_tools.py` | 242 | → DELETE (use core/stats/) |
| `experimenter_tools.py` | 235 | → DELETE (agent-specific) |
| `data_schema.py` | 226 | → DELETE (agent-specific) |
| `dataset_augmenter.py` | 194 | → DELETE (not used) |
| `lineup_tools.py` | 85 | → MOVE to `core/services/` |
| `market_tools.py` | 59 | → DELETE (use core/market/) |

### Data Layer Status

**Model configs DO have a data layer** in `core/data/models.py`:
- `ClassifierConfigRepository` - classifier model configs
- `PointsConfigRepository` - points regression configs
- `ExperimentRunsRepository` - experiment run tracking

This is correct architecture. The problem is `agents/tools/` bypassing it.

### Architectural Violations

1. `TrainingService` imports from `agents/tools/` (layering violation)
2. `core/services/matchup_chat/controller.py` imports from `agents/tools/`
3. `web/app.py` imports `StackingTrainer` from `agents/tools/`
4. Two "factory" files with confusing names
5. `game_service.py` uses pure functions instead of a class

---

## Target Architecture

### The Model Pipeline (5 Distinct Phases)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL PIPELINE PHASES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE A: Aggregate Metadata Collections                                     │
│  ─────────────────────────────────────────                                   │
│  Source: ESPN (configurable via league YAMLs)                                │
│  Output: players, teams collections (league-specific names from config)     │
│  Owner: core/pipeline/sync_pipeline.py                                       │
│                                                                              │
│  PHASE B: Aggregate Historic Game/Player Stats                               │
│  ─────────────────────────────────────────────                               │
│  Source: ESPN game data                                                      │
│  Output: games, player_stats collections (league-specific names from config) │
│  Owner: core/pipeline/sync_pipeline.py                                       │
│                                                                              │
│  PHASE C: Aggregate Venues, Rosters, Injuries, Derived Collections           │
│  ──────────────────────────────────────────────────────────────────          │
│  Source: Post-processing of raw data                                         │
│  Output: venues, rosters, elo_cache collections (league-specific names)     │
│          + injury fields updated in game documents                           │
│  Owner: core/pipeline/full_pipeline.py (post-processing steps)               │
│         core/pipeline/injuries_pipeline.py (injury computation)              │
│                                                                              │
│  PHASE D: Generate Master Training File                                      │
│  ──────────────────────────────────────                                      │
│  Prerequisite: Feature registry is populated                                 │
│  Input: All MongoDB collections from phases A-C                              │
│  Output: MASTER_TRAINING.csv (league-specific path)                          │
│  Owner: core/pipeline/training_pipeline.py                                   │
│         core/services/training_data.py (TrainingDataService)                 │
│  NOTE: This is a STANDALONE process, run separately from training           │
│                                                                              │
│  PHASE E: Model Training (COMPLETELY SEPARATE PROCESS)                       │
│  ──────────────────────────────────────────────────────                      │
│  Input: Carved subset of MASTER_TRAINING.csv                                 │
│  Output: Trained model artifacts (.pkl, scaler, features.json)               │
│  NEVER generates features - only reads from master CSV                       │
│  Owner: core/training/                                                       │
│                                                                              │
│  PHASE F: Prediction (Runtime)                                               │
│  ─────────────────────────────                                               │
│  Input: Model artifacts + live game context                                  │
│  Output: Win probabilities, point predictions                                │
│  Owner: core/services/prediction.py (PredictionService)                      │
│         core/models/ (model implementations)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### New Directory Structure

```
core/
├── data/                    # (UNCHANGED) Repository pattern for DB access
│   ├── models.py            # ClassifierConfigRepository, PointsConfigRepository,
│   │                        # ExperimentRunsRepository (ALREADY EXISTS)
│   └── ...
│
├── features/                # (UNCHANGED) Feature definitions, registry, parsing
├── stats/                   # (UNCHANGED) Stat computation (StatHandlerV2, PER)
│
├── pipeline/                # (UNCHANGED) Data pipeline orchestration (Phases A-D)
│   ├── sync_pipeline.py     # Phase A+B: ESPN → MongoDB
│   ├── full_pipeline.py     # Orchestrates A-D
│   ├── injuries_pipeline.py # Phase C: Injury field computation
│   └── training_pipeline.py # Phase D: Master training CSV
│
├── training/                # (CONSOLIDATED) All model training (Phase E)
│   ├── __init__.py
│   ├── constants.py
│   ├── model_factory.py     # Create sklearn instances
│   ├── model_evaluation.py  # CV evaluation
│   ├── cache_utils.py
│   │
│   │  # MOVED FROM agents/tools/
│   ├── experiment_runner.py # Run complete experiments
│   ├── stacking_trainer.py  # Ensemble training
│   ├── run_tracker.py       # Experiment tracking
│   └── dataset_builder.py   # Carve datasets from master CSV
│
├── models/                  # (REFACTORED) Model implementations
│   ├── bball_model.py       # Prediction (REMOVE create_training_data)
│   ├── points_regression.py
│   ├── ensemble.py
│   └── artifact_loader.py   # ← RENAME from factory.py
│
├── services/                # (CLEANED UP) High-level orchestration
│   ├── prediction.py        # PredictionService
│   ├── training_service.py  # ← UPDATE imports
│   ├── training_data.py     # TrainingDataService
│   ├── game_service.py      # ← CONVERT to GameService class
│   ├── matchup_predict.py   # ← MOVE from agents/tools/
│   ├── lineup_service.py    # (exists, may absorb lineup_tools)
│   └── news_service.py      # (exists, may absorb news_tools)
│
├── market/                  # (UNCHANGED)
└── utils/                   # (UNCHANGED)

agents/
├── schemas/                 # KEEP - Pydantic schemas for experiments
│   ├── experiment_config.py
│   └── dataset_spec.py
├── matchup_assistant/       # KEEP - Active agent
├── matchup_network/         # KEEP - Active agent network
├── modeler/                 # UPDATE imports, then KEEP
├── utils/                   # KEEP
└── tools/                   # DELETE ENTIRELY
```

---

## Implementation Plan

### Phase 1: Move Core Training Tools to core/training/

**Step 1.1: Copy files (don't delete yet)**

| Source | Destination |
|--------|-------------|
| `agents/tools/experiment_runner.py` | `core/training/experiment_runner.py` |
| `agents/tools/stacking_tool.py` | `core/training/stacking_trainer.py` |
| `agents/tools/run_tracker.py` | `core/training/run_tracker.py` |
| `agents/tools/dataset_builder.py` | `core/training/dataset_builder.py` |

**Step 1.2: Update internal imports in moved files**

Change:
```python
# FROM
from nba_app.agents.tools.dataset_builder import DatasetBuilder
from nba_app.agents.tools.run_tracker import RunTracker
from nba_app.agents.schemas.experiment_config import ExperimentConfig

# TO
from nba_app.core.training.dataset_builder import DatasetBuilder
from nba_app.core.training.run_tracker import RunTracker
from nba_app.core.training.schemas import ExperimentConfig  # Move schemas too
```

**Step 1.3: Move schemas to core/training/**

```
agents/schemas/experiment_config.py → core/training/schemas.py
agents/schemas/dataset_spec.py      → core/training/schemas.py (merge)
```

**Step 1.4: Update core/training/__init__.py**

```python
"""
Training infrastructure for model training operations (Phase E).

This module handles TRAINING from pre-generated master CSV.
Data generation (creating master CSV) is handled by core/pipeline/.

Components:
- ExperimentRunner: Run complete training experiments
- StackingTrainer: Train ensemble/stacking models
- RunTracker: Track experiment runs in MongoDB
- DatasetBuilder: Carve datasets from master CSV
- create_model_with_c: Create sklearn model instances
- evaluate_model_combo: Cross-validation evaluation
"""

from nba_app.core.training.model_factory import create_model_with_c
from nba_app.core.training.model_evaluation import (
    evaluate_model_combo,
    evaluate_model_combo_with_calibration
)
from nba_app.core.training.experiment_runner import ExperimentRunner
from nba_app.core.training.stacking_trainer import StackingTrainer
from nba_app.core.training.run_tracker import RunTracker
from nba_app.core.training.dataset_builder import DatasetBuilder

__all__ = [
    'create_model_with_c',
    'evaluate_model_combo',
    'evaluate_model_combo_with_calibration',
    'ExperimentRunner',
    'StackingTrainer',
    'RunTracker',
    'DatasetBuilder',
]
```

**Step 1.5: Update all import sites**

Files that need import updates:
- `core/services/training_service.py` (lines 48-50)
- `web/app.py` (line 244)
- `agents/modeler/modeler_agent.py` (lines 32-39)
- `core/services/matchup_chat/controller.py` (line 907)
- `tests/test_ensemble_meta_model.py` (line 66)
- `tests/test_chat_flow.py` (line 118)

---

### Phase 2: Move Utility Tools to core/services/

**Step 2.1: Move matchup_predict.py**

```
agents/tools/matchup_predict.py → core/services/matchup_predict.py
```

Update imports in:
- `agents/matchup_assistant/matchup_assistant_agent.py`
- `agents/tools/support_tools.py` (before deletion)
- `tests/test_prediction_workflow.py`

**Step 2.2: Absorb lineup_tools into LineupService**

Review `agents/tools/lineup_tools.py` (85 lines) and merge useful functionality into `core/services/lineup_service.py`.

**Step 2.3: Absorb news_tools into NewsService**

Review `agents/tools/news_tools.py` (804 lines) and merge into `core/services/news_service.py`.

---

### Phase 3: Update Agent Imports

**Step 3.1: Update modeler_agent.py**

```python
# FROM
from nba_app.agents.tools.data_schema import get_data_schema
from nba_app.agents.tools.dataset_builder import DatasetBuilder
from nba_app.agents.tools.experiment_runner import ExperimentRunner
from nba_app.agents.tools.support_tools import SupportTools
from nba_app.agents.tools.code_executor import CodeExecutor
from nba_app.agents.tools.run_tracker import RunTracker
from nba_app.agents.tools.dataset_augmenter import DatasetAugmenter
from nba_app.agents.tools.blend_experimenter import BlendExperimenter
from nba_app.agents.tools.stacking_tool import StackingTrainer

# TO
from nba_app.core.training import (
    DatasetBuilder, ExperimentRunner, RunTracker, StackingTrainer
)
# Remove deprecated tools or implement alternatives
```

**Step 3.2: Update matchup_assistant_agent.py**

```python
# FROM
from nba_app.agents.tools.matchup_predict import predict as predict_matchup
from nba_app.agents.tools.player_stats_tools import ...
from nba_app.agents.tools.game_tools import ...

# TO
from nba_app.core.services.matchup_predict import predict as predict_matchup
from nba_app.core.services.game_service import GameService
# Use core layer for stats
```

**Step 3.3: Update matchup_chat/controller.py**

This file has many imports from `agents/tools/`. Update to use core layer equivalents.

---

### Phase 4: Delete agents/tools/

Once all imports are updated and tests pass:

```bash
rm -rf nba_app/agents/tools/
```

Keep `agents/schemas/` if still needed, or move to `core/training/schemas.py`.

---

### Phase 5: Consolidate Factories

**Step 5.1: Rename core/models/factory.py**

```
core/models/factory.py → core/models/artifact_loader.py
```

Rename class:
```python
# FROM
class ModelFactory:
    """Centralized model creation and loading..."""

# TO
class ArtifactLoader:
    """Loads trained model artifacts from disk with caching."""
```

**Step 5.2: Update imports**

Search for `from nba_app.core.models.factory import ModelFactory` and update.

---

### Phase 6: Convert GameService to Class

**Step 6.1: Refactor game_service.py**

```python
"""Game Service - Core logic for game detail operations."""

class GameService:
    """Service for game-related operations."""

    def __init__(self, db, league: "LeagueConfig" = None):
        self.db = db
        self.league = league
        self._exclude_game_types = (
            league.exclude_game_types if league else ['preseason', 'allstar']
        )

        # Initialize repositories
        from nba_app.core.data import GamesRepository, RostersRepository
        self._games_repo = GamesRepository(db, league=league)
        self._rosters_repo = RostersRepository(db, league=league)

    def get_game_detail(self, game_id: str) -> Dict:
        """Get full game detail."""
        ...

    def calculate_player_stats(self, player_id: str, ...) -> Dict:
        """Calculate player stats for display."""
        ...

    # ... other methods


# Backwards compatibility - module-level functions
def get_game_detail(db, game_id: str, league=None) -> Dict:
    """Convenience wrapper."""
    return GameService(db, league).get_game_detail(game_id)
```

---

### Phase 7: Documentation

**Step 7.1: Update CLAUDE.md**

Add section on pipeline phases and the distinction between data generation and training.

**Step 7.2: Create core/ARCHITECTURE.md**

Document the core layer organization and dependencies.

---

## Migration Checklist

### Phase 1: Move Training Tools
- [ ] Copy `experiment_runner.py` → `core/training/`
- [ ] Copy `stacking_tool.py` → `core/training/stacking_trainer.py`
- [ ] Copy `run_tracker.py` → `core/training/`
- [ ] Copy `dataset_builder.py` → `core/training/`
- [ ] Move schemas to `core/training/schemas.py`
- [ ] Update internal imports in moved files
- [ ] Update `core/training/__init__.py`
- [ ] Update `core/services/training_service.py`
- [ ] Update `web/app.py`
- [ ] Update `agents/modeler/modeler_agent.py`
- [ ] Update `core/services/matchup_chat/controller.py`
- [ ] Update test files
- [ ] Run tests

### Phase 2: Move Utility Tools
- [ ] Move `matchup_predict.py` → `core/services/`
- [ ] Merge `lineup_tools.py` into `LineupService`
- [ ] Merge `news_tools.py` into `NewsService`
- [ ] Update all import sites

### Phase 3: Update Agent Imports
- [ ] Update `modeler_agent.py`
- [ ] Update `matchup_assistant_agent.py`
- [ ] Update `matchup_chat/controller.py`
- [ ] Run agent tests

### Phase 4: Delete agents/tools/
- [ ] Verify no remaining imports
- [ ] Delete `agents/tools/` directory
- [ ] Move/keep `agents/schemas/` if needed

### Phase 5: Consolidate Factories
- [ ] Rename `factory.py` → `artifact_loader.py`
- [ ] Rename `ModelFactory` → `ArtifactLoader`
- [ ] Update all imports

### Phase 6: GameService Class
- [ ] Convert to class
- [ ] Add backwards-compatible functions
- [ ] Update callers

### Phase 7: Documentation
- [ ] Update CLAUDE.md
- [ ] Create core/ARCHITECTURE.md

---

## Summary of Changes

| Action | Files Affected | Impact |
|--------|----------------|--------|
| Move 4 training tools to core | 4 files moved, ~15 import updates | Fixes layering violation |
| Move 3 utility tools to core | 3 files moved, ~10 import updates | Consolidates services |
| Delete agents/tools/ | 20 files deleted (~9,000 lines) | Removes confusion |
| Rename factory.py | 1 file, ~5 import updates | Clearer naming |
| Convert GameService | 1 file, ~20 caller updates | Consistent patterns |

**Net result**: Cleaner architecture, no layering violations, clear separation of pipeline phases.

---

## Part 2: Fix "nba" Naming (Generic → Basketball)

Many files incorrectly use "nba" in naming for generic basketball code. This should be fixed to support multi-league architecture.

### Function Names to Rename

| File | Current Name | New Name |
|------|--------------|----------|
| `web/app.py:323` | `get_nba_model()` | `get_bball_model()` |
| `web/app.py:7800` | `get_nba_modeler_sessions()` | `get_modeler_sessions()` |
| `cli/populate_master_training_cols.py:599` | `_infer_nba_model_flags()` | `_infer_model_flags()` |

### Collection Name: nba_modeler_sessions → modeler_sessions

This collection is league-agnostic (stores modeler session state). Rename in:
- `web/app.py` (19 occurrences)
- `agents/modeler/modeler_agent.py` (line 82)

### Hardcoded Collection Defaults to Fix

Functions with hardcoded NBA collection names as defaults should use `None` and derive from league config:

| File | Line | Current Default | Fix |
|------|------|-----------------|-----|
| `web/app.py:1094` | `config_collection='model_config_nba'` | Use league config |
| `web/app.py:1393` | `teams_collection='teams_nba'` | Use league config |
| `web/app.py:6751` | `classifier_config_collection='model_config_nba'` | Use league config |
| `core/services/game_service.py:32` | `player_stats_collection='stats_nba_players'` | Use league config |

### Direct db.collection_nba Access to Fix

Hardcoded collection accesses that should use league config:

**web/app.py:**
- `db.players_nba` (11 occurrences)
- `db.nba_rosters` (3 occurrences)

**cli/populate_master_training_cols.py:**
- `db.stats_nba`, `db.teams_nba` (5 occurrences)

**cli/scripts/populate_vegas_lines.py:**
- `db.stats_nba` (7+ occurrences)

### What NOT to Change

These are correctly named:
- Collection names from league YAML configs (e.g., `games_nba` when loaded from config)
- The `nba_app` package name itself
- References to `leagues/nba.yaml`
- `NBAModel = BballModel` backward compatibility alias
- Functions like `_nba_app_dir()` that get the app directory path
