"""
Basketball Stacking Trainer.

Thin subclass of sportscore's BaseStackingTrainer with basketball-specific:
- Repository access (ClassifierConfigRepository)
- Dataset building (DatasetBuilder)
- Legacy model loading (ExperimentRunner)
"""

from sportscore.training.base_stacking import BaseStackingTrainer


class StackingTrainer(BaseStackingTrainer):
    """Basketball stacking trainer. Provides sport-specific hooks."""

    def __init__(self, db=None, league=None):
        if db is None:
            from bball.mongo import Mongo
            mongo = Mongo()
            db = mongo.db
        super().__init__(db=db, league=league)

    def _get_model_config_repository(self):
        from bball.data import ClassifierConfigRepository
        return ClassifierConfigRepository(self.db, league=self.league)

    def _get_dataset_builder(self):
        from bball.training.dataset_builder import DatasetBuilder
        return DatasetBuilder(db=self.db, league=self.league)

    def _load_legacy_model(self, run_id: str):
        from bball.training.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(db=self.db, league=self.league)
        return runner._load_classification_model(run_id)
