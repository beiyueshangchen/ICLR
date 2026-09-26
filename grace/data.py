"""Dataset loading for GRACE.

Each dataset is exposed through one ``load_*_bundle`` function that returns a
:class:`DatasetBundle`.  A bundle holds learner trajectories (lists of
interactions, each a ``{"problem_idx", "label", "dt"}`` dict), the row-normalised
concept graph ``A`` and the row-normalised Q-matrix ``Q`` that the model consumes
as constant buffers.

The expected on-disk layout of every dataset is documented in
``docs/datasets.md``.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

#: Trajectories shorter than this are discarded, matching the paper's setup.
MIN_SEQ_LEN = 5

#: Time gaps are clamped to one week before the log transform.
_MAX_GAP_SECONDS = 7 * 24 * 3600

Interaction = Dict[str, float]
Sequence = List[Interaction]


@dataclass
class DatasetBundle:
    """Everything the model and the training loop need for one dataset."""

    train: List[Sequence]
    valid: List[Sequence]
    test: Optional[List[Sequence]]
    adjacency: torch.Tensor  # (num_concepts, num_concepts)
    q_matrix: torch.Tensor  # (num_items, num_concepts)
    item_id2idx: Dict
    cid2idx: Dict

    @property
    def num_concepts(self) -> int:
        return len(self.cid2idx)

    @property
    def num_items(self) -> int:
        return len(self.item_id2idx)


class SequenceDataset(Dataset):
    """Thin wrapper turning a list of trajectories into a ``Dataset``."""

    def __init__(self, sequences: List[Sequence]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self.sequences[idx]


def my_collate(batch):
    """Keep trajectories as-is; they have variable length and are dicts."""
    return batch


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _parse_int_sequence(seq_text: object) -> List[int]:
    """Parse a comma-separated string of integers into a list."""
    values = []
    for token in str(seq_text).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-8, None)


def finalize_concept_graph(adjacency: np.ndarray) -> torch.Tensor:
    """Add self-loops and row-normalise: the propagation matrix ``S = rownorm(A + I)``."""
    adjacency = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    return torch.from_numpy(_row_normalize(adjacency))


def compact_vocabulary(
    sequences_by_split: List[List[Sequence]],
    q_matrix: np.ndarray,
    adjacency: np.ndarray,
    item_id2idx: Dict,
    cid2idx: Dict,
):
    """Reduce the item/concept vocabulary to the indices the data actually uses.

    The released index spaces (124 questions for ASSISTments 2009, 835 for Junyi)
    contain entries that no surviving trajectory ever touches, because questions
    either disappear with the dropped short trajectories or fall outside
    ``max_seq_len``.  Keeping the unused slots inflates the concept count the
    model reports, so they are removed and the remaining indices are compacted
    onto ``0..K-1``; this reproduces the question/concept counts of Table 1 in
    the paper.  ``sequences_by_split`` is remapped in place.

    Returns the compacted ``(q_matrix, adjacency, item_id2idx, cid2idx)``.
    """
    used_items = sorted(
        {step["problem_idx"] for sequences in sequences_by_split for seq in sequences for step in seq}
    )
    if not used_items:
        return q_matrix, adjacency, item_id2idx, cid2idx

    used_concepts = sorted({j for i in used_items for j in np.nonzero(q_matrix[i])[0]})
    item_map = {old: new for new, old in enumerate(used_items)}
    concept_map = {old: new for new, old in enumerate(used_concepts)}

    for sequences in sequences_by_split:
        for seq in sequences:
            for step in seq:
                step["problem_idx"] = item_map[step["problem_idx"]]

    return (
        q_matrix[np.ix_(used_items, used_concepts)],
        adjacency[np.ix_(used_concepts, used_concepts)],
        {name: item_map[i] for name, i in item_id2idx.items() if i in item_map},
        {name: concept_map[i] for name, i in cid2idx.items() if i in concept_map},
    )


def _load_jsonl_sequences(path: str, max_seq_len: int) -> List[Sequence]:
    """Read one interaction pair list per line (pyKT ``[problem, label]`` format)."""
    sequences = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_seq = json.loads(line)
            seq: Sequence = []
            for pair in raw_seq:
                if not isinstance(pair, list) or len(pair) < 2:
                    continue
                # These split files store only [problem, label] pairs, so no
                # timestamp is available and the gap is 0.0 at every step; the
                # phi(dt) branch then contributes a constant on this path.
                # ``build_junyi_sequences`` reads the raw log, which does carry
                # timestamps, and computes real gaps there.
                seq.append({"problem_idx": int(pair[0]), "label": int(pair[1]), "dt": 0.0})
            if len(seq) >= MIN_SEQ_LEN:
                # Head slice: the published configuration keeps the first
                # ``max_seq_len`` interactions of a learner and drops the
                # remainder, which is what the reported numbers are computed from.
                sequences.append(seq[:max_seq_len])
    return sequences


# --------------------------------------------------------------------------- #
# Junyi Academy
# --------------------------------------------------------------------------- #
def load_junyi_graph(vertex_json: str, prereq_json: str, sim_json: str):
    """Build Junyi's raw concept adjacency from its prerequisite/similarity files.

    ``prerequisite.json`` holds curated prerequisite links and
    ``similarity.json`` data-derived similarity scores; they describe different
    relations, so both are merged.  The result is not yet normalised — see
    :func:`finalize_concept_graph`.
    """
    with open(vertex_json, "r", encoding="utf-8") as handle:
        cid2idx = json.load(handle)
    num_concepts = len(cid2idx)
    adjacency = np.zeros((num_concepts, num_concepts), dtype=np.float32)

    with open(prereq_json, "r", encoding="utf-8") as handle:
        for pre, succ in json.load(handle):
            adjacency[pre, succ] = 1.0

    with open(sim_json, "r", encoding="utf-8") as handle:
        for a, b, score in json.load(handle):
            adjacency[a, b] = adjacency[b, a] = max(adjacency[a, b], score / 9.0)

    return cid2idx, adjacency


def build_junyi_q_matrix(exercise_table: str, cid2idx: Dict):
    """Read Junyi's question-to-concept tags and build the Q-matrix."""
    frame = pd.read_csv(exercise_table)
    frame = frame[frame["name"].notna()].drop_duplicates(subset=["name"]).copy()

    item_names = frame["name"].tolist()
    item_id2idx = {name: i for i, name in enumerate(item_names)}
    q = np.zeros((len(item_names), len(cid2idx)), dtype=np.float32)

    for name in item_names:
        if name in cid2idx:
            q[item_id2idx[name], cid2idx[name]] = 1.0

    unmatched = [name for name in item_names if name not in cid2idx]
    if unmatched:
        warnings.warn(
            f"{len(unmatched)} of {len(item_names)} exercises have no concept in the graph, "
            f"e.g. {unmatched[:3]}",
            stacklevel=2,
        )

    return _row_normalize(q), item_id2idx


def build_junyi_sequences(
    log_csv: str,
    item_id2idx: Dict,
    max_rows: Optional[int] = None,
    max_seq_len: int = 100,
) -> List[Sequence]:
    """Build timestamp-ordered trajectories from Junyi's raw problem log.

    The released log stores ``time_done`` as microseconds since the epoch; parsing
    it without an explicit unit would treat the values as nanoseconds and shrink
    every time gap by a factor of 1000.

    The full log holds ~16.6M rows, so reading it costs several GB of RAM.  Pass
    ``max_rows`` to work on a head slice instead, which is a prefix of the file
    rather than a random sample.
    """
    if max_rows is not None:
        warnings.warn(f"Reading only the first {max_rows} rows of {log_csv}", stacklevel=2)
    frame = pd.read_csv(log_csv, nrows=max_rows)
    label_map = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    frame["correct"] = frame["correct"].map(label_map).fillna(0).astype(int)

    if pd.api.types.is_numeric_dtype(frame["time_done"]):
        frame["time_done"] = pd.to_datetime(frame["time_done"], unit="us", errors="coerce")
    else:
        frame["time_done"] = pd.to_datetime(frame["time_done"], errors="coerce")
    frame = frame[frame["time_done"].notna()]

    sequences: List[Sequence] = []
    for _, group in tqdm(frame.groupby("user_id"), desc="Processing students"):
        group = group.sort_values("time_done")
        seq: Sequence = []
        prev_time = None
        for _, row in group.iterrows():
            exercise = row["exercise"]
            if exercise not in item_id2idx:
                continue
            current_time = row["time_done"]
            if prev_time is None:
                gap_seconds = 0.0
            else:
                gap_seconds = max((current_time - prev_time).total_seconds(), 0.0)
            gap_seconds = min(gap_seconds, _MAX_GAP_SECONDS)
            seq.append(
                {
                    "problem_idx": item_id2idx[exercise],
                    "label": int(row["correct"]),
                    "dt": float(np.log1p(gap_seconds)),
                }
            )
            prev_time = current_time
        if len(seq) >= MIN_SEQ_LEN:
            sequences.append(seq[:max_seq_len])
    return sequences


def load_junyi_bundle(root: str, max_seq_len: int = 100) -> DatasetBundle:
    """Load Junyi.  Prefers the released ``train.json``/``test.json`` split.

    The released ``test.json`` is used as the held-out split, i.e. it fills the
    ``valid`` field and no separate test split is available.  When the released
    split is absent the trajectories are rebuilt from
    ``junyi_ProblemLog_original.csv``, which is *not* shipped with this
    repository because of its size.
    """
    cid2idx, adjacency = load_junyi_graph(
        os.path.join(root, "graph_vertex.json"),
        os.path.join(root, "prerequisite.json"),
        os.path.join(root, "similarity.json"),
    )
    q_matrix, item_id2idx = build_junyi_q_matrix(
        os.path.join(root, "junyi_Exercise_table.csv"), cid2idx
    )

    train_json = os.path.join(root, "train.json")
    test_json = os.path.join(root, "test.json")
    if os.path.exists(train_json) and os.path.exists(test_json):
        train_sequences = _load_jsonl_sequences(train_json, max_seq_len)
        valid_sequences = _load_jsonl_sequences(test_json, max_seq_len)
    else:
        raw_log = os.path.join(root, "junyi_ProblemLog_original.csv")
        if not os.path.exists(raw_log):
            raise FileNotFoundError(
                f"Junyi needs either the released split ({os.path.basename(train_json)} and "
                f"{os.path.basename(test_json)}) or the raw log {os.path.basename(raw_log)} in "
                f"'{root}'. See docs/datasets.md for the expected layout."
            )
        sequences = build_junyi_sequences(raw_log, item_id2idx, max_seq_len=max_seq_len)
        random.shuffle(sequences)
        split = int(len(sequences) * 0.8)
        train_sequences, valid_sequences = sequences[:split], sequences[split:]

    q_matrix, adjacency, item_id2idx, cid2idx = compact_vocabulary(
        [train_sequences, valid_sequences], q_matrix, adjacency, item_id2idx, cid2idx
    )

    return DatasetBundle(
        train=train_sequences,
        valid=valid_sequences,
        test=None,
        adjacency=finalize_concept_graph(adjacency),
        q_matrix=torch.from_numpy(q_matrix),
        item_id2idx=item_id2idx,
        cid2idx=cid2idx,
    )


# --------------------------------------------------------------------------- #
# ASSISTments 2009
# --------------------------------------------------------------------------- #
#: Index space of the released ASSISTments 2009 files; only 117 of the 124 slots
#: are still used once the trajectories have been preprocessed.
ASSIST2009_NUM_NODES = 124


def load_weighted_edge_adjacency(graph_json: str, num_nodes: int) -> np.ndarray:
    """Read a ``[[source, target, weight], ...]`` edge list into a raw adjacency."""
    with open(graph_json, "r", encoding="utf-8") as handle:
        edges = json.load(handle)

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        if not isinstance(edge, list) or len(edge) < 2:
            continue
        u, v = int(edge[0]), int(edge[1])
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        weight = float(edge[2]) if len(edge) >= 3 else 1.0
        adjacency[u, v] = max(adjacency[u, v], weight)

    return adjacency


def load_assist2009_bundle(root: str, max_seq_len: int = 100) -> DatasetBundle:
    """Load the skill-builder ASSISTments 2009 split (117 questions == 117 concepts)."""
    train_sequences = _load_jsonl_sequences(os.path.join(root, "train.json"), max_seq_len)
    valid_sequences = _load_jsonl_sequences(os.path.join(root, "valid.json"), max_seq_len)
    test_sequences = _load_jsonl_sequences(os.path.join(root, "test.json"), max_seq_len)

    num_nodes = ASSIST2009_NUM_NODES
    q_matrix = np.eye(num_nodes, dtype=np.float32)
    adjacency = load_weighted_edge_adjacency(
        os.path.join(root, "correct_transition_graph.json"), num_nodes=num_nodes
    )
    item_id2idx = {i: i for i in range(num_nodes)}
    cid2idx = {i: i for i in range(num_nodes)}

    # question index == concept index, so the identity Q-matrix ties the two
    # vocabularies together and both compact to the same 117 concepts.
    q_matrix, adjacency, item_id2idx, cid2idx = compact_vocabulary(
        [train_sequences, valid_sequences, test_sequences],
        q_matrix,
        adjacency,
        item_id2idx,
        cid2idx,
    )

    return DatasetBundle(
        train=train_sequences,
        valid=valid_sequences,
        test=test_sequences,
        adjacency=finalize_concept_graph(adjacency),
        q_matrix=torch.from_numpy(q_matrix),
        item_id2idx=item_id2idx,
        cid2idx=cid2idx,
    )


# --------------------------------------------------------------------------- #
# AAAI 2023 challenge data (TAL Education / Peiyou)
# --------------------------------------------------------------------------- #
def _load_peiyou_txt_sequences(txt_path: str, max_seq_len: int = 100) -> List[Sequence]:
    """Read the challenge's four-line-per-trajectory format."""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing split file: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    sequences: List[Sequence] = []
    i = 0
    while i + 3 < len(lines):
        try:
            seq_len = int(lines[i])
        except ValueError:
            i += 1
            continue

        question_ids = _parse_int_sequence(lines[i + 1])
        # Line i + 2 repeats the concepts; the Q-matrix already carries them.
        responses = _parse_int_sequence(lines[i + 3])
        i += 4

        seq: Sequence = []
        for j in range(min(seq_len, len(question_ids), len(responses))):
            question_id = int(question_ids[j])
            response = int(responses[j])
            if question_id < 0 or response not in (0, 1):
                continue
            # The challenge files carry no timestamps, so dt is 0.0 throughout.
            seq.append({"problem_idx": question_id, "label": response, "dt": 0.0})

        if len(seq) >= MIN_SEQ_LEN:
            # Head slice: keep the first ``max_seq_len`` steps, as in the other
            # loaders.
            sequences.append(seq[:max_seq_len])

    return sequences


def load_aaai2023_bundle(root: str, max_seq_len: int = 100, fold: int = 0) -> DatasetBundle:
    """Load the AAAI 2023 Global Knowledge Tracing Challenge data (question level)."""
    questions_json = os.path.join(root, "questions.json")
    keyid2idx_json = os.path.join(root, "keyid2idx.json")
    train_txt = os.path.join(root, f"train_{fold}.txt")
    valid_txt = os.path.join(root, f"valid_{fold}.txt")

    required = [questions_json, keyid2idx_json, train_txt, valid_txt]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"AAAI 2023 required files missing: {missing}")

    with open(questions_json, "r", encoding="utf-8") as handle:
        question_meta = json.load(handle)
    with open(keyid2idx_json, "r", encoding="utf-8") as handle:
        keyid2idx = json.load(handle)

    question_map = keyid2idx["questions"]
    concept_map = keyid2idx["concepts"]
    num_questions = len(question_map)
    num_concepts = len(concept_map)

    adjacency = np.zeros((num_concepts, num_concepts), dtype=np.float32)
    q = np.zeros((num_questions, num_concepts), dtype=np.float32)

    for question_id_raw, question_idx in question_map.items():
        meta = question_meta.get(str(question_id_raw), {})
        concept_indices = set()

        routes = meta.get("concept_routes", []) if isinstance(meta, dict) else []
        for route in routes:
            tokens = [t.strip() for t in str(route).split("----") if t.strip()]
            mapped = [concept_map[t] for t in tokens if t in concept_map]
            concept_indices.update(mapped)
            for u, v in zip(mapped[:-1], mapped[1:]):
                adjacency[u, v] += 1.0

        # Fallback for questions without a usable route.
        if not concept_indices and isinstance(meta, dict) and "concepts" in meta:
            for concept in meta["concepts"]:
                concept_str = str(concept)
                if concept_str in concept_map:
                    concept_indices.add(concept_map[concept_str])

        if concept_indices:
            q[question_idx, list(concept_indices)] = 1.0
        else:
            q[question_idx, 0 if num_concepts == 0 else (question_idx % num_concepts)] = 1.0

    q = _row_normalize(q)
    adjacency += np.eye(num_concepts, dtype=np.float32)
    adjacency = _row_normalize(adjacency)

    return DatasetBundle(
        train=_load_peiyou_txt_sequences(train_txt, max_seq_len=max_seq_len),
        valid=_load_peiyou_txt_sequences(valid_txt, max_seq_len=max_seq_len),
        test=None,
        adjacency=torch.from_numpy(adjacency),
        q_matrix=torch.from_numpy(q),
        item_id2idx={i: i for i in range(num_questions)},
        cid2idx={i: i for i in range(num_concepts)},
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_LOADERS = {
    "assist2009": load_assist2009_bundle,
    "junyi": load_junyi_bundle,
    "aaai2023": load_aaai2023_bundle,
}

#: Datasets with a fold split, mapped to the keyword their loader expects.
_FOLD_KEYWORDS = {"aaai2023": "fold"}


def prepare_dataset(
    dataset_name: str, root: str, max_seq_len: int = 100, fold: Optional[int] = None
) -> DatasetBundle:
    """Load the dataset with the canonical name ``dataset_name`` from ``root``.

    ``fold`` selects the cross-validation fold of the datasets that ship one
    (``aaai2023``) and is rejected for the others, which have a single fixed
    train/valid/test split.
    """
    from grace.config import normalize_dataset_name  # local import keeps this module standalone

    canonical = normalize_dataset_name(dataset_name)
    loader = _LOADERS[canonical]
    if fold is None:
        return loader(root, max_seq_len=max_seq_len)

    if canonical not in _FOLD_KEYWORDS:
        raise ValueError(f"{canonical} has a single fixed split and takes no fold argument")
    return loader(root, max_seq_len=max_seq_len, **{_FOLD_KEYWORDS[canonical]: fold})
