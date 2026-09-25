"""Configuration objects and dataset-path resolution for GRACE.

The paper uses three datasets: ASSISTments 2009, Junyi Academy and the AAAI 2023
Global Knowledge Tracing Challenge data (TAL Education / Peiyou).  Dataset
locations are never hard-coded; for a dataset ``D`` the root directory is
resolved in the following order:

1. ``--dataset-root`` on the command line,
2. the per-dataset environment variable (``GRACE_ASSIST2009_ROOT``, ...),
3. ``<--data-root | $GRACE_DATA_ROOT>/<sub-directory of D>``,
4. ``<repository>/data/<sub-directory of D>`` for the data shipped with the code.

See ``docs/datasets.md`` for the files each loader expects inside a root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: Environment variable pointing at a directory that contains all dataset roots.
DATA_ROOT_ENV = "GRACE_DATA_ROOT"

#: Directory shipped inside the repository; searched last, needs no configuration.
LOCAL_DATA_DIRNAME = "data"

#: Accepted spellings mapped onto the canonical dataset names.
DATASET_ALIASES: Dict[str, str] = {
    "assist2009": "assist2009",
    "assistments2009": "assist2009",
    "assist09": "assist2009",
    "junyi": "junyi",
    "junyiacademy": "junyi",
    "aaai2023": "aaai2023",
    "aaai2023challenge": "aaai2023",
    "peiyou": "aaai2023",
    "xes3g5m": "aaai2023",
}

#: Candidate sub-directory names, in the order they are tried.
_CANDIDATE_SUBDIRS: Dict[str, Tuple[str, ...]] = {
    "assist2009": ("assistment_2009_2010", "assist2009"),
    "junyi": ("junyi",),
    "aaai2023": ("aaai2023", "peiyou", "xes3g5m"),
}

#: Primary sub-directory used when a shared ``GRACE_DATA_ROOT`` is given.
DATASET_SUBDIRS: Dict[str, str] = {
    dataset: names[0] for dataset, names in _CANDIDATE_SUBDIRS.items()
}

#: Per-dataset environment variable overrides.
DATASET_ENV_VARS: Dict[str, str] = {
    "assist2009": "GRACE_ASSIST2009_ROOT",
    "junyi": "GRACE_JUNYI_ROOT",
    "aaai2023": "GRACE_AAAI2023_ROOT",
}

SUPPORTED_DATASETS: Tuple[str, ...] = tuple(DATASET_SUBDIRS)

#: Human-readable name of every switchable component.  The letters follow the
#: order in which the paper removes components cumulatively.
ABLATION_COMPONENTS: Dict[str, str] = {
    "a": "item_encoder",
    "b": "concept_attention",
    "c": "concept_graph_encoder",
    "d": "graph_diffused_gradient",
    "e": "preconditioner",
}

#: Cumulative ablation schedule of Table 3: each entry keeps everything to its left
#: and removes one further component.  The item encoder (``a``) is switchable too,
#: but the paper does not remove it in the cumulative study.
CUMULATIVE_ABLATION_ORDER: Tuple[str, ...] = ("e", "e,d", "e,d,c", "e,d,c,b")

OPTIMIZERS: Tuple[str, ...] = ("adam", "adamw", "sgd", "rmsprop")
MODEL_SELECTION: Tuple[str, ...] = ("val", "test")


def normalize_dataset_name(name: str) -> str:
    """Return the canonical dataset name for ``name``."""
    key = str(name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported datasets: {', '.join(SUPPORTED_DATASETS)}."
        )
    return DATASET_ALIASES[key]


def _repository_root() -> str:
    """Absolute path of the repository that contains the ``grace`` package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _first_existing(search_roots: Iterable[str], dataset: str) -> Optional[str]:
    """First ``<root>/<candidate>`` that exists on disk, or ``None``."""
    for root in search_roots:
        for candidate in _CANDIDATE_SUBDIRS[dataset]:
            path = os.path.join(root, candidate)
            if os.path.isdir(path):
                return path
    return None


def resolve_dataset_root(
    dataset: str,
    dataset_root: Optional[str] = None,
    data_root: Optional[str] = None,
) -> str:
    """Resolve the directory holding the files of ``dataset``.

    Explicit configuration wins over the conventions, and a directory that
    actually exists wins over one that does not.  Raises ``EnvironmentError``
    with setup instructions when nothing usable is found, so a missing dataset
    is reported before any file lookup happens.
    """
    dataset = normalize_dataset_name(dataset)

    if dataset_root:
        return os.path.abspath(os.path.expanduser(dataset_root))

    env_var = DATASET_ENV_VARS[dataset]
    env_value = os.environ.get(env_var)
    if env_value:
        return os.path.abspath(os.path.expanduser(env_value))

    base = data_root or os.environ.get(DATA_ROOT_ENV)
    if base:
        found = _first_existing([os.path.abspath(os.path.expanduser(base))], dataset)
        if found is not None:
            return found

    # Data shipped with the repository (`<repository>/data/...`) needs no setup.
    found = _first_existing(
        [os.path.join(_repository_root(), LOCAL_DATA_DIRNAME)], dataset
    )
    if found is not None:
        return found

    if base:
        return os.path.abspath(
            os.path.expanduser(os.path.join(base, DATASET_SUBDIRS[dataset]))
        )

    raise EnvironmentError(
        f"No root directory found for dataset {dataset!r}. Provide one of:\n"
        f"  * --dataset-root <path>\n"
        f"  * export {env_var}=<path>\n"
        f"  * export {DATA_ROOT_ENV}=<path>   (then use <path>/{DATASET_SUBDIRS[dataset]})\n"
        f"  * place the files in {os.path.join(_repository_root(), LOCAL_DATA_DIRNAME, DATASET_SUBDIRS[dataset])}"
    )


def parse_ablations(value: object) -> List[str]:
    """Normalise an ablation specification into a sorted list of letters.

    ``value`` may be a comma-separated string (``"e,d"``), an iterable of
    letters/component names (``["e", "d"]``), ``None`` or ``""`` for the full
    model.
    """
    if value is None:
        return []

    if isinstance(value, str):
        raw: Iterable[str] = value.split(",")
    else:
        raw = value

    name_to_letter = {name: letter for letter, name in ABLATION_COMPONENTS.items()}
    letters = []
    for token in raw:
        token = str(token).strip().lower()
        if not token:
            continue
        letter = name_to_letter.get(token, token)
        if letter not in ABLATION_COMPONENTS:
            raise ValueError(
                f"Unknown ablation {token!r}. Use letters from "
                f"{sorted(ABLATION_COMPONENTS)} or the component names "
                f"{sorted(name_to_letter)}."
            )
        if letter not in letters:
            letters.append(letter)
    return sorted(letters)


@dataclass
class TrainConfig:
    """Every knob of a single GRACE run.

    The defaults correspond to the configuration reported in the paper
    (concept/item width 32, rank 4, ``(lambda_1, lambda_2) = (0.6, 0.4)``, one
    learner trajectory per training unit).
    """

    # --- data -------------------------------------------------------------
    dataset: str = "assist2009"
    data_root: Optional[str] = None
    dataset_root: Optional[str] = None
    max_seq_len: int = 100
    fold: Optional[int] = None

    # --- run bookkeeping --------------------------------------------------
    output_dir: str = "runs"
    seed: int = 3407
    device: str = "auto"

    # --- optimisation -----------------------------------------------------
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 1
    optimizer: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 0.0  # 0 disables gradient clipping

    # --- model ------------------------------------------------------------
    concept_emb_dim: int = 32
    item_emb_dim: int = 32
    rank: int = 4

    # --- regularisation ---------------------------------------------------
    alpha: float = 1.0  # global scale of the two regularisers
    lambda_1: float = 0.6  # weight of the state-change penalty
    lambda_2: float = 0.4  # weight of the low-rank-factor penalty

    # --- evaluation -------------------------------------------------------
    eval_every: int = 1
    early_stop_patience: int = 10
    model_selection: str = "val"

    # --- ablation ---------------------------------------------------------
    ablations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dataset = normalize_dataset_name(self.dataset)
        self.ablations = parse_ablations(self.ablations)
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.model_selection not in MODEL_SELECTION:
            raise ValueError(
                f"model_selection must be one of {MODEL_SELECTION}, got {self.model_selection!r}"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.betas = tuple(self.betas)  # accept lists coming from YAML/JSON

    @property
    def ablation_tag(self) -> str:
        """Short identifier of the ablated configuration (``full`` if none)."""
        return "full" if not self.ablations else "_".join(self.ablations)

    def as_dict(self) -> Dict[str, object]:
        """Plain-dict view, suitable for JSON/YAML dumping."""
        return {
            "dataset": self.dataset,
            "data_root": self.data_root,
            "dataset_root": self.dataset_root,
            "max_seq_len": self.max_seq_len,
            "fold": self.fold,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "device": self.device,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "betas": list(self.betas),
            "grad_clip": self.grad_clip,
            "concept_emb_dim": self.concept_emb_dim,
            "item_emb_dim": self.item_emb_dim,
            "rank": self.rank,
            "alpha": self.alpha,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "eval_every": self.eval_every,
            "early_stop_patience": self.early_stop_patience,
            "model_selection": self.model_selection,
            "ablations": list(self.ablations),
        }
