"""Command-line entry point for training and evaluating GRACE.

Examples
--------
``python -m grace --config configs/assist2009.yaml``

``python -m grace --dataset junyi --dataset-root /data/junyi --ablations e,d``
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import fields
from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from grace.config import (
    ABLATION_COMPONENTS,
    MODEL_SELECTION,
    OPTIMIZERS,
    SUPPORTED_DATASETS,
    TrainConfig,
    resolve_dataset_root,
)
from grace.data import DatasetBundle, SequenceDataset, my_collate, prepare_dataset
from grace.model import CognitiveModel
from grace.train_eval import evaluate, train_one_epoch

CSV_HEADER = (
    "epoch,train_loss,train_auc,train_acc,val_loss,val_auc,val_acc,"
    "test_loss,test_auc,test_acc,selected_auc\n"
)


def _build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="grace",
        description="Train and evaluate GRACE for knowledge tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file whose keys override the defaults below.",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, default=None, choices=SUPPORTED_DATASETS)
    data.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Directory containing one sub-directory per dataset (env: GRACE_DATA_ROOT).",
    )
    data.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Directory holding the files of the selected dataset (overrides --data-root).",
    )
    data.add_argument("--max-seq-len", type=int, default=None)
    data.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index for the datasets that ship one (aaai2023 only).",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", type=str, default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--epochs", type=int, default=None)
    optimisation.add_argument("--lr", type=float, default=None)
    optimisation.add_argument("--batch-size", type=int, default=None)
    optimisation.add_argument("--optimizer", type=str, default=None, choices=OPTIMIZERS)
    optimisation.add_argument("--weight-decay", type=float, default=None)
    optimisation.add_argument("--momentum", type=float, default=None)
    optimisation.add_argument(
        "--betas", type=float, nargs=2, default=None, metavar=("BETA1", "BETA2")
    )
    optimisation.add_argument(
        "--grad-clip", type=float, default=None, help="Max gradient norm; 0 disables clipping."
    )

    model = parser.add_argument_group("model")
    model.add_argument("--concept-emb-dim", type=int, default=None)
    model.add_argument("--item-emb-dim", type=int, default=None)
    model.add_argument("--rank", type=int, default=None)

    regularisation = parser.add_argument_group("regularisation")
    regularisation.add_argument("--alpha", type=float, default=None)
    regularisation.add_argument("--lambda-1", type=float, default=None)
    regularisation.add_argument("--lambda-2", type=float, default=None)

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument("--eval-every", type=int, default=None)
    evaluation.add_argument("--early-stop-patience", type=int, default=None)
    evaluation.add_argument(
        "--model-selection",
        type=str,
        default=None,
        choices=MODEL_SELECTION,
        help="Split used to pick the best checkpoint.",
    )

    ablation = parser.add_argument_group("ablation")
    ablation.add_argument(
        "--ablations",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated components to remove, e.g. \"e,d\". "
            + " ".join(f"{k}={v}" for k, v in sorted(ABLATION_COMPONENTS.items()))
        ),
    )
    # `argparse` cannot render dataclass defaults when every flag defaults to
    # `None`, so record them in the epilog for discoverability.
    parser.epilog = "Defaults: " + ", ".join(
        f"{field.name}={getattr(defaults, field.name)!r}" for field in fields(TrainConfig)
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    """Build a :class:`TrainConfig` from CLI flags layered on top of an optional YAML file."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    overrides: Dict[str, object] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            overrides.update(yaml.safe_load(handle) or {})

    overrides.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})

    valid_keys = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(overrides) - valid_keys)
    if unknown:
        parser.error(f"unknown option(s) {unknown}; valid keys are {sorted(valid_keys)}")

    return TrainConfig(**overrides)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is available")
    return torch.device(name)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    betas = tuple(cfg.betas)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=betas
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=betas
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.RMSprop(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def build_loaders(bundle: DatasetBundle, cfg: TrainConfig):
    """Return ``(train, valid, test)`` loaders; ``valid``/``test`` may be ``None``."""
    train_loader = DataLoader(
        SequenceDataset(bundle.train),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=my_collate,
    )
    valid_loader = (
        DataLoader(
            SequenceDataset(bundle.valid),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=my_collate,
        )
        if bundle.valid
        else None
    )
    test_loader = (
        DataLoader(
            SequenceDataset(bundle.test),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=my_collate,
        )
        if bundle.test
        else None
    )
    return train_loader, valid_loader, test_loader


def make_run_dir(cfg: TrainConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.output_dir, cfg.dataset, f"{cfg.ablation_tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    root = resolve_dataset_root(cfg.dataset, cfg.dataset_root, cfg.data_root)
    print(f"Dataset root: {root}")
    bundle = prepare_dataset(cfg.dataset, root, max_seq_len=cfg.max_seq_len, fold=cfg.fold)

    print(f"Using {cfg.dataset} split: train={len(bundle.train)} valid={len(bundle.valid)} "
          f"test={0 if bundle.test is None else len(bundle.test)}")
    print(f"num_concepts={bundle.num_concepts} num_items={bundle.num_items}")
    if cfg.ablations:
        removed = ", ".join(f"{letter}={ABLATION_COMPONENTS[letter]}" for letter in cfg.ablations)
        print(f"Ablations enabled (removed components): {removed}")

    train_loader, valid_loader, test_loader = build_loaders(bundle, cfg)
    if train_loader is None or len(bundle.train) == 0:
        raise RuntimeError("the training split is empty")

    selection_loader = None
    if cfg.model_selection == "test" and test_loader is not None:
        selection_loader = "test"
    elif valid_loader is not None:
        selection_loader = "val"
    elif test_loader is not None:
        selection_loader = "test"
    else:
        raise RuntimeError("no validation or test split available for model selection")
    print(f"Model selection split: {selection_loader}")

    run_dir = make_run_dir(cfg)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg.as_dict(), handle, indent=2, sort_keys=True)
    print(f"Run directory: {run_dir}")

    model = CognitiveModel(
        num_concepts=bundle.num_concepts,
        num_items=bundle.num_items,
        concept_adj=bundle.adjacency,
        q_matrix=bundle.q_matrix,
        concept_emb_dim=cfg.concept_emb_dim,
        item_emb_dim=cfg.item_emb_dim,
        rank=cfg.rank,
        ablations=cfg.ablations,
    ).to(device)
    optimizer = build_optimizer(model, cfg)

    best_auc = 0.0
    best_epoch = 0
    no_improve_evals = 0
    checkpoint_path = os.path.join(run_dir, "best_model.pt")
    metrics_path = os.path.join(run_dir, "metrics.csv")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write(CSV_HEADER)

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_auc, train_acc = train_one_epoch(
                model,
                train_loader,
                device,
                optimizer,
                alpha=cfg.alpha,
                lambda_1=cfg.lambda_1,
                lambda_2=cfg.lambda_2,
                grad_clip=cfg.grad_clip,
            )
            print(
                f"Epoch {epoch} | Train loss {train_loss:.4f} "
                f"AUC {train_auc:.4f} ACC {train_acc:.4f}"
            )

            if epoch % cfg.eval_every != 0 and epoch != cfg.epochs:
                continue

            val_loss = val_auc = val_acc = float("nan")
            test_loss = test_auc = test_acc = float("nan")

            if valid_loader is not None:
                val_loss, val_auc, val_acc = evaluate(model, valid_loader, device)
                print(f"Epoch {epoch} | Val   loss {val_loss:.4f} AUC {val_auc:.4f} ACC {val_acc:.4f}")
            if test_loader is not None:
                test_loss, test_auc, test_acc = evaluate(model, test_loader, device)
                print(
                    f"Epoch {epoch} | Test  loss {test_loss:.4f} AUC {test_auc:.4f} "
                    f"ACC {test_acc:.4f}"
                )

            selected_auc = test_auc if selection_loader == "test" else val_auc

            if selected_auc > best_auc:
                best_auc = selected_auc
                best_epoch = epoch
                no_improve_evals = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.as_dict(),
                        "epoch": epoch,
                        "selected_auc": selected_auc,
                        "val_auc": val_auc,
                        "test_auc": test_auc,
                    },
                    checkpoint_path,
                )
                print(f"--- Best model saved (selected {selection_loader} AUC {best_auc:.4f}) ---")
            else:
                no_improve_evals += 1

            handle.write(
                f"{epoch},{train_loss:.6f},{train_auc:.6f},{train_acc:.6f},"
                f"{val_loss:.6f},{val_auc:.6f},{val_acc:.6f},"
                f"{test_loss:.6f},{test_auc:.6f},{test_acc:.6f},{selected_auc:.6f}\n"
            )
            handle.flush()

            if no_improve_evals >= cfg.early_stop_patience:
                print(
                    f"--- Early stop: no improvement in {cfg.early_stop_patience} evaluations. "
                    f"Best {selection_loader} AUC {best_auc:.4f} at epoch {best_epoch} ---"
                )
                break

    print(f"Best {selection_loader} AUC {best_auc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics:    {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
