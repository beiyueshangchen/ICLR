"""Tests for the dataset helpers and the dataset loaders (synthetic files)."""

import json

import numpy as np
import pytest
import torch

from grace.data import (
    MIN_SEQ_LEN,
    SequenceDataset,
    _parse_int_sequence,
    _row_normalize,
    build_junyi_sequences,
    compact_vocabulary,
    my_collate,
    prepare_dataset,
)

#: Question indices used by the synthetic ASSISTments splits; deliberately
#: non-contiguous so that compaction is exercised.
ORIGINAL_INDICES = (0, 2, 5, 7, 9)

#: ... and the vocabulary the loader must produce from them.
NUM_NODES = len(ORIGINAL_INDICES)


def _write_assist2009(root, seq_len=MIN_SEQ_LEN):
    # the 4th edge touches question 99, which never occurs, so it must be dropped
    (root / "correct_transition_graph.json").write_text(
        json.dumps([[0, 2, 5.0], [2, 5, 9.0], [5, 0, 1.0], [7, 99, 2.0]]), encoding="utf-8"
    )
    for split in ("train", "valid", "test"):
        lines = []
        for start in range(3):
            row = [
                [ORIGINAL_INDICES[(start + i) % NUM_NODES], i % 2] for i in range(seq_len)
            ]
            lines.append(json.dumps(row))
        (root / f"{split}.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _write_junyi(root, seq_len=MIN_SEQ_LEN):
    """Synthetic Junyi: 4 concepts/exercises of which only 0 and 2 are ever seen."""
    (root / "graph_vertex.json").write_text(
        json.dumps({"c0": 0, "c1": 1, "c2": 2, "c3": 3}), encoding="utf-8"
    )
    # 0 -> 2 survives compaction; 0 -> 1 and 2 -> 3 leave the vocabulary
    (root / "prerequisite.json").write_text(json.dumps([[0, 1], [0, 2]]), encoding="utf-8")
    (root / "similarity.json").write_text(json.dumps([[2, 3, 9.0]]), encoding="utf-8")
    (root / "junyi_Exercise_table.csv").write_text(
        "name,pretty_name\nc0,Concept 0\nc1,Concept 1\nc2,Concept 2\nc3,Concept 3\n",
        encoding="utf-8",
    )
    for split in ("train", "test"):
        lines = [
            json.dumps([[(0 if i % 2 == 0 else 2), i % 2] for i in range(seq_len)])
            for _ in range(2)
        ]
        (root / f"{split}.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_parse_int_sequence_skips_blank_tokens():
    assert _parse_int_sequence("1, 2,,3") == [1, 2, 3]


def test_row_normalize_rows_sum_to_one():
    matrix = _row_normalize(np.eye(4, dtype=np.float32))
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_compact_vocabulary_is_a_noop_without_interactions():
    q_matrix = np.eye(3, dtype=np.float32)
    adjacency = np.ones((3, 3), dtype=np.float32)
    items, concepts = {i: i for i in range(3)}, {i: i for i in range(3)}

    q_out, adj_out, items_out, concepts_out = compact_vocabulary(
        [[], []], q_matrix, adjacency, items, concepts
    )

    assert np.array_equal(q_out, q_matrix)
    assert np.array_equal(adj_out, adjacency)
    assert items_out == items
    assert concepts_out == concepts


def test_compact_vocabulary_keeps_only_the_used_slots():
    sequences = [[{"problem_idx": 7}, {"problem_idx": 2}], [{"problem_idx": 9}]]
    q_matrix = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        q_matrix[i, i] = 1.0
    adjacency = np.ones((10, 10), dtype=np.float32)

    q_out, adj_out, items_out, concepts_out = compact_vocabulary(
        [sequences], q_matrix, adjacency, {i: i for i in range(10)}, {i: i for i in range(10)}
    )

    # 2, 7 and 9 are the only indices that occur, so they map onto 0, 1, 2
    assert [step["problem_idx"] for seq in sequences for step in seq] == [1, 0, 2]
    assert tuple(q_out.shape) == (3, 3)
    assert tuple(adj_out.shape) == (3, 3)
    assert items_out == {2: 0, 7: 1, 9: 2}
    assert concepts_out == {2: 0, 7: 1, 9: 2}


def test_build_junyi_sequences_reads_microsecond_timestamps(tmp_path):
    # the released log stores time_done as microseconds since the epoch
    rows = ["user_id,exercise,correct,time_done"]
    for user in (1, 2):
        for i in range(MIN_SEQ_LEN):
            rows.append(f"{user},c0,{int(i % 2 == 0)},{1500000000000000 + i * 10000000}")
    log = tmp_path / "junyi_ProblemLog_original.csv"
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    sequences = build_junyi_sequences(str(log), {"c0": 0}, max_seq_len=100)

    assert len(sequences) == 2
    assert [step["problem_idx"] for step in sequences[0]] == [0] * MIN_SEQ_LEN
    assert sequences[0][0]["dt"] == 0.0
    # consecutive interactions are 10 s apart; nanoseconds would make this log1p(0.01)
    assert sequences[0][1]["dt"] == pytest.approx(np.log1p(10.0))


def test_build_junyi_sequences_clamps_gaps_to_one_week(tmp_path):
    month = 30 * 24 * 3600 * 1000000  # microseconds
    rows = ["user_id,exercise,correct,time_done"]
    for i in range(MIN_SEQ_LEN):
        rows.append(f"1,c0,1,{1500000000000000 + i * month}")
    log = tmp_path / "junyi_ProblemLog_original.csv"
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    sequences = build_junyi_sequences(str(log), {"c0": 0}, max_seq_len=100)

    assert sequences[0][1]["dt"] == pytest.approx(np.log1p(7 * 24 * 3600))


def test_build_junyi_sequences_skips_unknown_exercises(tmp_path):
    rows = ["user_id,exercise,correct,time_done"]
    for i in range(MIN_SEQ_LEN + 2):
        exercise = "c1" if i == 2 else "c0"
        rows.append(f"1,{exercise},1,{1500000000000000 + i * 10000000}")
    log = tmp_path / "junyi_ProblemLog_original.csv"
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    sequences = build_junyi_sequences(str(log), {"c0": 0}, max_seq_len=100)

    assert len(sequences) == 1
    assert len(sequences[0]) == MIN_SEQ_LEN + 1


def test_sequence_dataset_and_collate_keep_trajectories_whole():
    sequences = [[{"problem_idx": 0, "label": 1, "dt": 0.0}] for _ in range(3)]
    dataset = SequenceDataset(sequences)
    assert len(dataset) == 3
    assert dataset[1] is sequences[1]
    assert my_collate(sequences) is sequences


def test_prepare_dataset_loads_synthetic_assist2009(tmp_path):
    _write_assist2009(tmp_path)
    bundle = prepare_dataset("assist2009", str(tmp_path), max_seq_len=100)

    assert len(bundle.train) == 3
    assert len(bundle.valid) == 3
    assert len(bundle.test) == 3
    assert bundle.num_concepts == NUM_NODES
    assert bundle.num_items == NUM_NODES
    assert bundle.cid2idx == {0: 0, 2: 1, 5: 2, 7: 3, 9: 4}
    assert bundle.item_id2idx == {0: 0, 2: 1, 5: 2, 7: 3, 9: 4}
    assert tuple(bundle.q_matrix.shape) == (NUM_NODES, NUM_NODES)
    assert tuple(bundle.adjacency.shape) == (NUM_NODES, NUM_NODES)
    assert torch.allclose(bundle.adjacency.sum(dim=1), torch.ones(NUM_NODES))
    # the identity Q-matrix makes every question its own concept
    assert torch.allclose(torch.diagonal(bundle.q_matrix), torch.ones(NUM_NODES))


def test_prepare_dataset_compacts_the_surviving_question_indices(tmp_path):
    _write_assist2009(tmp_path)
    bundle = prepare_dataset("assist2009", str(tmp_path), max_seq_len=100)

    observed = {
        step["problem_idx"] for seq in bundle.train + bundle.valid + bundle.test for step in seq
    }
    assert observed == set(range(NUM_NODES))
    # compaction preserves the order, so 0, 2, 5, 7, 9 becomes 0, 1, 2, 3, 4
    assert [step["problem_idx"] for step in bundle.train[0]] == [0, 1, 2, 3, 4]


def test_prepare_dataset_drops_edges_leaving_the_compacted_vocabulary(tmp_path):
    _write_assist2009(tmp_path)
    bundle = prepare_dataset("assist2009", str(tmp_path), max_seq_len=100)

    # 0 -> 1, 1 -> 2, 2 -> 0 survive (5.0 / 9.0 / 1.0); 3 -> 99 is dropped
    assert bundle.adjacency[0, 1] > 0
    assert bundle.adjacency[1, 2] > 0
    assert bundle.adjacency[2, 0] > 0
    # node 3 keeps its self-loop only, so the row is [0, 0, 0, 1, 0]
    assert bundle.adjacency[3].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0])


def test_prepare_dataset_drops_trajectories_shorter_than_min_seq_len(tmp_path):
    _write_assist2009(tmp_path, seq_len=MIN_SEQ_LEN)
    (tmp_path / "test.json").write_text(
        json.dumps([[0, 1] for _ in range(MIN_SEQ_LEN - 1)]) + "\n", encoding="utf-8"
    )
    bundle = prepare_dataset("assist2009", str(tmp_path), max_seq_len=100)
    assert bundle.test == []


def test_prepare_dataset_rejects_fold_for_single_split_datasets(tmp_path):
    _write_assist2009(tmp_path)
    with pytest.raises(ValueError):
        prepare_dataset("assist2009", str(tmp_path), fold=0)


def test_prepare_dataset_compacts_the_junyi_vocabulary(tmp_path):
    _write_junyi(tmp_path)
    bundle = prepare_dataset("junyi", str(tmp_path), max_seq_len=100)

    # both splits load, and test.json fills the held-out slot
    assert len(bundle.train) == 2
    assert len(bundle.valid) == 2
    assert bundle.test is None
    # only c0 and c2 are ever answered, so the 4-slot vocabulary shrinks to 2
    assert bundle.num_concepts == 2
    assert bundle.num_items == 2
    assert bundle.cid2idx == {"c0": 0, "c2": 1}
    assert bundle.item_id2idx == {"c0": 0, "c2": 1}
    assert tuple(bundle.q_matrix.shape) == (2, 2)
    assert tuple(bundle.adjacency.shape) == (2, 2)
    assert torch.allclose(bundle.adjacency.sum(dim=1), torch.ones(2))
    assert [step["problem_idx"] for step in bundle.train[0]] == [0, 1, 0, 1, 0]


def test_prepare_dataset_drops_junyi_edges_leaving_the_vocabulary(tmp_path):
    _write_junyi(tmp_path)
    bundle = prepare_dataset("junyi", str(tmp_path), max_seq_len=100)

    # c0 -> c2 survives; c0 -> c1 and c2 -> c3 (similarity) leave the vocabulary
    assert bundle.adjacency[0, 1] > 0
    assert bundle.adjacency[1, 0] == 0
    assert bundle.adjacency[0, 0] > 0
    assert bundle.adjacency[1, 1] > 0


def test_prepare_dataset_explains_a_missing_junyi_split(tmp_path):
    _write_junyi(tmp_path)
    (tmp_path / "train.json").unlink()
    (tmp_path / "test.json").unlink()

    with pytest.raises(FileNotFoundError, match="junyi_ProblemLog_original.csv"):
        prepare_dataset("junyi", str(tmp_path), max_seq_len=100)
