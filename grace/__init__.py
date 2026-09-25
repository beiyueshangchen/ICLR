"""GRACE: graph-conditioned meta-optimization of the cognitive state.

Public API
----------
``TrainConfig``
    All hyper-parameters of a run plus dataset-root resolution.
``prepare_dataset``
    Loads one of the supported benchmarks into a :class:`DatasetBundle`.
``CognitiveModel``
    The full model.
``train_one_epoch`` / ``evaluate``
    Training and evaluation loops.
"""

from grace.config import TrainConfig, resolve_dataset_root
from grace.data import DatasetBundle, SequenceDataset, my_collate, prepare_dataset
from grace.model import CognitiveModel
from grace.train_eval import evaluate, train_one_epoch

__all__ = [
    "CognitiveModel",
    "DatasetBundle",
    "SequenceDataset",
    "TrainConfig",
    "evaluate",
    "my_collate",
    "prepare_dataset",
    "resolve_dataset_root",
    "train_one_epoch",
]

__version__ = "1.0.0"
