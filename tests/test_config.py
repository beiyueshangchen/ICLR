"""Tests for configuration handling and dataset-root resolution."""

import os

import pytest

from grace.config import (
    ABLATION_COMPONENTS,
    CUMULATIVE_ABLATION_ORDER,
    DATA_ROOT_ENV,
    SUPPORTED_DATASETS,
    TrainConfig,
    normalize_dataset_name,
    parse_ablations,
    resolve_dataset_root,
)


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("assist2009", "assist2009"),
        ("ASSIST09", "assist2009"),
        ("Assistments2009", "assist2009"),
        ("junyi", "junyi"),
        ("JunyiAcademy", "junyi"),
        ("peiyou", "aaai2023"),
        ("XES3G5M", "aaai2023"),
    ],
)
def test_normalize_dataset_name(spelling, expected):
    assert normalize_dataset_name(spelling) == expected


def test_normalize_dataset_name_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_dataset_name("not-a-dataset")


def test_only_the_three_datasets_of_the_paper_are_supported():
    assert sorted(SUPPORTED_DATASETS) == ["aaai2023", "assist2009", "junyi"]
    with pytest.raises(ValueError):
        normalize_dataset_name("assist2015")


def test_parse_ablations_accepts_letters_names_and_empty():
    assert parse_ablations(None) == []
    assert parse_ablations("") == []
    assert parse_ablations("e") == ["e"]
    assert parse_ablations("e,d") == ["d", "e"]
    assert parse_ablations(["e", "d"]) == ["d", "e"]
    assert parse_ablations("preconditioner") == ["e"]


def test_parse_ablations_rejects_unknown_component():
    with pytest.raises(ValueError):
        parse_ablations("z")


def test_cumulative_order_matches_table_3():
    assert CUMULATIVE_ABLATION_ORDER[0] == "e"
    assert CUMULATIVE_ABLATION_ORDER[-1] == "e,d,c,b"
    # every row removes one more component than the row above it, and the item
    # encoder is never removed in the cumulative study of the paper
    sizes = [len(entry.split(",")) for entry in CUMULATIVE_ABLATION_ORDER]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)
    for entry in CUMULATIVE_ABLATION_ORDER:
        assert "a" not in entry.split(",")


def test_resolve_dataset_root_precedence(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "shared"))
    assert resolve_dataset_root("junyi", str(explicit)) == os.path.abspath(str(explicit))


def test_resolve_dataset_root_uses_per_dataset_env(monkeypatch, tmp_path):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    monkeypatch.setenv("GRACE_JUNYI_ROOT", str(tmp_path / "junyi"))
    assert resolve_dataset_root("junyi") == os.path.abspath(str(tmp_path / "junyi"))


def test_resolve_dataset_root_uses_shared_root(monkeypatch, tmp_path):
    monkeypatch.delenv("GRACE_JUNYI_ROOT", raising=False)
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr("grace.config._repository_root", lambda: str(tmp_path / "empty"))
    assert resolve_dataset_root("junyi") == os.path.abspath(str(tmp_path / "junyi"))


def test_resolve_dataset_root_falls_back_to_the_shipped_data_dir(monkeypatch, tmp_path):
    for name in ("GRACE_JUNYI_ROOT", DATA_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("grace.config._repository_root", lambda: str(tmp_path))
    local = tmp_path / "data" / "junyi"
    local.mkdir(parents=True)
    assert resolve_dataset_root("junyi") == os.path.abspath(str(local))


def test_resolve_dataset_root_accepts_candidate_subdir_names(monkeypatch, tmp_path):
    for name in ("GRACE_ASSIST2009_ROOT", DATA_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("grace.config._repository_root", lambda: str(tmp_path))
    local = tmp_path / "data" / "assistment_2009_2010"
    local.mkdir(parents=True)
    assert resolve_dataset_root("assist2009") == os.path.abspath(str(local))


def test_resolve_dataset_root_without_configuration_raises(monkeypatch, tmp_path):
    for name in ("GRACE_JUNYI_ROOT", DATA_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("grace.config._repository_root", lambda: str(tmp_path))
    with pytest.raises(EnvironmentError):
        resolve_dataset_root("junyi")


def test_train_config_defaults_match_the_paper():
    cfg = TrainConfig()
    assert (cfg.concept_emb_dim, cfg.item_emb_dim) == (32, 32)
    assert cfg.rank == 4
    assert (cfg.lambda_1, cfg.lambda_2) == (0.6, 0.4)
    assert cfg.batch_size == 1
    assert cfg.ablations == []
    assert cfg.ablation_tag == "full"


def test_train_config_normalises_inputs():
    cfg = TrainConfig(dataset="Junyi", ablations="e,d", betas=[0.9, 0.99])
    assert cfg.dataset == "junyi"
    assert cfg.ablations == ["d", "e"]
    assert cfg.ablation_tag == "d_e"
    assert cfg.betas == (0.9, 0.99)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimizer": "cifar"},
        {"model_selection": "train"},
        {"batch_size": 0},
    ],
)
def test_train_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
