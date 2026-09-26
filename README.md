# GRACE: Graph-Conditioned Meta-Optimization of Cognitive State for Knowledge Tracing

Official implementation of

> **Learning How Students Update: Graph-Conditioned Meta-Optimization of Cognitive State
> for Knowledge Tracing**

GRACE keeps the usual knowledge-tracing prediction head and replaces the *hand-designed*
state transition with a **learned gradient step on the learner's own prediction error**.
At every interaction the cognitive state is updated by a graph-conditioned,
preconditioned meta-optimization step. The gradient is computed from the learner's
prediction loss, while the concept graph is used to diffuse the gradient across
related concepts. A diagonal-plus-low-rank preconditioner further controls the
update direction and scale, so the state transition itself is learned end to end
instead of being hand-designed.

## Repository layout

```
.
├── grace/                  # the package
│   ├── config.py           # TrainConfig, dataset-root resolution, ablation parsing
│   ├── data.py             # dataset loaders (Junyi + two optional benchmarks)
│   ├── model.py            # ConceptGNN, ItemEncoder, ConceptAttention, meta-optimizer
│   ├── train_eval.py       # unrolled training / evaluation loops
│   ├── runner.py           # CLI entry point
│   └── __main__.py         # `python -m grace`
├── configs/                # one YAML config per dataset (paper configuration)
├── data/junyi/             # the Junyi subset the example runs on
├── docs/datasets.md        # the on-disk layout each loader expects
├── scripts/                # reproducibility drivers (main table, ablation, sweeps)
│   └── make_submission_zip.py  # build the code-submission archive
└── tests/                  # pytest smoke tests (synthetic data, no downloads)
```

## Installation

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate grace
```

The code was developed with Python 3.9 and PyTorch 2.x on CPU/CUDA. It only uses
`torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm` and `pyyaml`.

Every command below is written relative to the repository root, so run them from
the directory that contains this README — the one holding `grace/`, `configs/` and
`data/`. Nothing outside the repository is needed for the shipped example, and no
absolute path appears anywhere in the code.

## Data

`data/junyi/` ships a 38 MB subset of **Junyi Academy** — exactly the six files
the loader reads — so the example runs with no configuration:

```bash
python -m grace --config ./configs/junyi.yaml
```

| Dataset | Shipped here | Vocabulary after preprocessing |
|---|---|---|
| Junyi Academy (`data/junyi/`, 38 MB) | 33,843 train / 8,311 held-out trajectories, 2,172,984 interactions | 715 questions / 715 concepts / 3,743 graph edges |

Those splits reproduce the Junyi row of Table 1 of the paper exactly. The
released index space is larger than the vocabulary the model uses (835 -> 715),
because questions that only ever appear in a trajectory removed by preprocessing
are dropped and the surviving indices are compacted onto `0..K-1`. See
[docs/datasets.md](docs/datasets.md) for the layout, the run example and what the
subset leaves out.

The other two benchmarks are not redistributed; their loaders live in
`grace/data.py` and read a directory given with `--dataset-root`, so the paper's
remaining tables stay reproducible from your own copy of the data (the headers of
`configs/assist2009.yaml` and `configs/aaai2023.yaml` list the accepted paths).

To keep a dataset outside the repository — the shipped one included — point GRACE
at it with the CLI or the environment:

```bash
export GRACE_DATA_ROOT=/path/to/datasets            # shared parent directory
export GRACE_JUNYI_ROOT=/path/to/datasets/junyi     # per-dataset override
python -m grace --dataset junyi --dataset-root /path/to/datasets/junyi
```

For a dataset `D` the root is resolved as `--dataset-root`, then the
`GRACE_<D>_ROOT` environment variable, then `--data-root` or `GRACE_DATA_ROOT`,
and finally `<repository>/data/`; the first existing directory wins (the middle
two steps try candidate sub-directory names such as `junyi/`).

## Quick start

```bash
# paper configuration on the data shipped with the repository
python -m grace --config ./configs/junyi.yaml

# everything can be overridden on the command line
python -m grace --config ./configs/junyi.yaml \
    --epochs 100 --lr 1e-3 --batch-size 1 --rank 4 \
    --lambda-1 0.6 --lambda-2 0.4 --seed 3407

# cumulative ablation of the paper with a single flag
python -m grace --config ./configs/junyi.yaml --ablations e,d
```

Each run writes to `<output-dir>/<dataset>/<ablation-tag>_<timestamp>/`:

| File | Content |
|---|---|
| `config.json` | the full resolved configuration of the run |
| `metrics.csv` | per-epoch train/val/test loss, AUC and ACC |
| `best_model.pt` | best checkpoint (state dict + config + selected metric) |

`python -m grace --help` lists every flag; the defaults are printed in the epilog.

## Reproducing the paper

Three datasets, with the configuration of Appendix B: concept embedding width 32,
item embedding width 32, preconditioner rank 4, mixing weight `alpha` = 1, loss
weights `lambda1` = 0.6 and `lambda2` = 0.4, one learner trajectory per training
unit, Adam with learning rate 1e-3, and the cognitive state initialised to zero.

```bash
bash ./scripts/run_main_table.sh               # Table 2  (GRACE row)
bash ./scripts/run_ablation.sh                 # Table 3  (cumulative ablation)
bash ./scripts/run_sensitivity.sh              # Table 4  (four one-factor sweeps)
```

The drivers default to all three benchmarks, so they need the two datasets that
are not redistributed here; set `DATASETS` to run a subset:

```bash
DATASETS="junyi" bash ./scripts/run_main_table.sh
```

Reported test AUC / ACC (paper, Table 2):

| Dataset | Test AUC | Test ACC |
|---|---|---|
| ASSIST2009 | 0.7835 | 0.7222 |
| Junyi | 0.8896 | 0.8147 |
| AAAI2023 | 0.8451 | 0.8397 |

Comparisons reported in the paper use the public implementations of DKT, DKT2, GKT,
SAKT, AKT, simpleKT, DTransformer, stableKT, extraKT, LefoKT, RobustKT, UKT and ACEKT
on the same splits and the same evaluation protocol; this repository contains GRACE
only.

### Packaging the submission

```bash
python ./scripts/make_submission_zip.py -o grace_submission.zip
```

The archive contains the code, the tests and the shipped Junyi subset — nothing
else is required to run the example. See [docs/datasets.md](docs/datasets.md) for
what the subset contains.

### Checkpoint selection

`--model-selection` chooses the split used to pick the best checkpoint:

* `val` (default) — validation AUC, the honest protocol for new results;
* `test` — test AUC, which is how the numbers above were selected.

`scripts/run_main_table.sh`, `scripts/run_ablation.sh` and
`scripts/run_sensitivity.sh` pass `--model-selection test` so that they reproduce the
published tables exactly. Note that the Junyi and AAAI2023 loaders are given a single
held-out split (it fills the validation slot), so for those two datasets both settings
select on the same split; for ASSIST2009, which has separate `valid.json` and
`test.json`, the two settings differ.

## Ablations

Each component of GRACE can be removed independently with `--ablations`, a
comma-separated list of letters:

| Letter | Component | Effect when removed |
|---|---|---|
| `a` | item encoder | item content vector is all zeros |
| `b` | concept attention | global context is the mean concept embedding |
| `c` | concept graph encoder | concept embeddings come from a linear map |
| `d` | graph-diffused gradient | the meta-optimizer sees the raw gradient |
| `e` | preconditioner | the update uses the un-preconditioned gradient |

The cumulative schedule of Section 5.3 / Table 3 (each row removes one more component
than the row above it) is `e` → `e,d` → `e,d,c` → `e,d,c,b`, reported test AUC:

| Variant | ASSIST2009 | Junyi | AAAI2023 |
|---|---|---|---|
| GRACE (full) | 0.7835 | 0.8896 | 0.8451 |
| w/o preconditioner | 0.7747 | 0.8873 | 0.8405 |
| w/o graph-diffused gradient | 0.7671 | 0.8842 | 0.8380 |
| w/o concept graph encoder | 0.7669 | 0.8813 | 0.8366 |
| w/o concept attention | 0.7660 | 0.8740 | 0.8344 |

Component names are accepted in place of letters, e.g.
`--ablations preconditioner,graph_diffused_gradient`. The item encoder (`a`) is
switchable as well, but the paper does not remove it in the cumulative study.

## Sensitivity

`scripts/run_sensitivity.sh` runs the four one-factor sweeps of Appendix D — concept
width `d_c`, item width `d`, preconditioner rank `r` and the regularization weights
`(lambda_1, lambda_2)`. The paper runs them in stages, each stage keeping the best
value found by the earlier ones; the grids are the ones in the script. Table 4 lists
the raw AUC/ACC behind Figure 5.

## Tests

```bash
python -m pytest ./tests -q
```

The tests are synthetic (no dataset download): they check configuration and path
resolution, the ablation parsing, the loaders against tiny generated files, the
documented bounds of `eta` (`[1e-6, 0.1]`), `d` (`[1e-6, 10]`) and `U` (`[-10, 10]`),
and that every ablation variant trains and evaluates end to end.

## Citation

```bibtex
@inproceedings{grace2026,
  title     = {Learning How Students Update: Graph-Conditioned Meta-Optimization
               of Cognitive State for Knowledge Tracing},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## License

Released under the MIT License; see [LICENSE](LICENSE).
