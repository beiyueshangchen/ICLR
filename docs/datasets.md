# Dataset preparation

The code, the tests and the runnable example in this repository use **Junyi
Academy**. Only the subset of that dataset which is needed to run the example is
distributed here; everything else is read from a directory you point the loader
at.

## What is shipped

`data/junyi/` holds exactly the six files the Junyi loader reads:

| File | Size | Role |
|---|---|---|
| `graph_vertex.json` | 28 KB | concept id → index (`{"<concept id>": <index>, ...}`) |
| `prerequisite.json` | 25 KB | curated prerequisite links, `[[predecessor, successor], ...]` |
| `similarity.json` | 75 KB | data-derived similarity, `[[a, b, score in [0, 9]], ...]` |
| `junyi_Exercise_table.csv` | 137 KB | exercise table; `name` is the exercise id |
| `train.json` | 31 MB | released training split |
| `test.json` | 6.9 MB | released held-out split |

Nothing else is required: the raw problem log
(`junyi_ProblemLog_original.csv`, ~2.5 GB) is only read by the fallback path
below, and the released splits are used instead.

## Running the example

```bash
python -m grace --config configs/junyi.yaml
```

The run resolves the dataset, builds the vocabulary, trains the full model and
writes per-epoch metrics to a CSV under `--output-dir`. The first lines and the
first training steps look like this:

```
$ python -m grace --config configs/junyi.yaml --output-dir runs/example
Dataset root: <repository>/data/junyi
Using junyi split: train=33843 valid=8311 test=0
num_concepts=715 num_items=715
Model selection split: val
Run directory: runs/example/junyi/full_20260926_020723
Training:   0%|          | 1/33843 [00:04<45:09:38,  4.80s/it]
Training:   0%|          | 2/33843 [00:06<26:06:16,  2.78s/it]
...
```

`test=0` is expected for this dataset: only one held-out split is released
(`test.json`), and GRACE uses it as the validation split, so the number reported
in the paper is produced from that split. Each trajectory is one batch
(`batch_size = 1` in the paper), so trust the steady-state iteration time rather
than the first steps, which include warm-up.

To keep the dataset somewhere else and point GRACE at it:

```bash
python -m grace --config configs/junyi.yaml --dataset-root /data/kt/junyi
export GRACE_JUNYI_ROOT=/data/kt/junyi        # environment form of the same thing
```

### Root resolution order

For a dataset `D` the root directory is resolved in this order, and the first
entry that yields an existing directory wins:

1. `--dataset-root <path>`
2. `$GRACE_<D>_ROOT` (for Junyi: `GRACE_JUNYI_ROOT`)
3. `--data-root <path>` / `$GRACE_DATA_ROOT` — the candidate sub-directory
   `junyi/` is tried under it
4. `<repository>/data/` — the copy shipped with the code

If nothing exists, the path derived from step 3 is returned, so the loader
reports the file it cannot find instead of failing inside `pandas`.

## Junyi Academy

```
junyi/
├── graph_vertex.json               # concept id -> index
├── prerequisite.json               # curated prerequisite links (weight 1.0)
├── similarity.json                 # data-derived similarity (score / 9)
├── junyi_Exercise_table.csv        # columns: name (exercise id), ...
├── train.json                      # released split
└── test.json                       # released split (held-out)
```

* Preferred path: if `train.json` and `test.json` are present they are read as
  the training and held-out splits. Both are JSON lines, one trajectory per
  line:

  ```json
  [[0, 1], [12, 0], [7, 1], [3, 0], [9, 1]]
  ```

  `test.json` fills the *validation* slot and `bundle.test` is `None`.
* Fallback path: if the released split is missing, trajectories are rebuilt from
  `junyi_ProblemLog_original.csv` (columns `user_id`, `exercise`, `correct`,
  `time_done`), ordered by timestamp and split 80/20 into train/valid.
  `time_done` is an integer count of **microseconds** since the epoch, so it is
  parsed with `unit="us"`; the whole 16.6M-row log is read (several GB of RAM).
  `build_junyi_sequences(..., max_rows=N)` reads only the first `N` rows and warns
  that the result is a head slice, not a sample. When neither the released split
  nor the raw log is present the loader raises `FileNotFoundError` naming the
  files it looked for.
* The concept graph combines prerequisites (weight `1.0`), similarities
  (`score / 9`) and self-loops, then is row-normalised. Unlike a single
  transition statistic, these two files describe *different kinds* of relation,
  so both are merged before self-loops are added.
* The Q-matrix is built from `junyi_Exercise_table.csv`: exercise `name` →
  concept id, row-normalised. Exercises that carry no concept trigger a warning.
* `graph_vertex.json` and `junyi_Exercise_table.csv` both index **835** concepts
  and exercises, but a concept only contributes to the model if it survives the
  paper's preprocessing, i.e. if some trajectory of at least `MIN_SEQ_LEN` (= 5)
  interactions answers one of its exercises within the first `max_seq_len`
  (= 100) steps. Only **715** indices survive, so the vocabulary is compacted
  from 835 onto `0..714`, which is the count reported in Table 1 of the paper
  (715 questions / 715 concepts). The same rule is applied to the concept graph,
  so the 835 x 835 raw adjacency becomes a 715 x 715 matrix with 3,743 weighted
  edges. Compaction preserves the relative order of the original indices, so the
  mapping is deterministic, and the vocabulary size follows `max_seq_len` the
  same way it does in the shared preprocessing below.

The shipped files reproduce the paper's Junyi row exactly: 33,843 training and
8,311 held-out trajectories, 2,172,984 interactions, 715 concepts / 715 items,
mean trajectory length 51.5 and median 43.

## Preprocessing shared by all loaders

* Trajectories shorter than `MIN_SEQ_LEN = 5` interactions are discarded.
* Sequences are truncated to `--max-seq-len` (default `100`) interactions.
* The time gap `dt` is the number of seconds between consecutive interactions of
  the same learner, clamped to one week and then transformed with `log1p`; the
  first interaction of a trajectory has `dt = 0`.
* The concept graph and the Q-matrix are both row-normalised
  (`row / max(row_sum, 1e-8)`); self-loops are added to the concept graph before
  normalisation, so the propagation matrix is `S = rownorm(A + I)`.
* Every loader then drops the item and concept slots that no surviving
  interaction refers to and compacts the remaining indices onto `0..K-1`, so
  `num_items` and `num_concepts` reflect the vocabulary actually used rather
  than the index space of the released files. This is what turns Junyi's 835
  indices into the 715 reported in the paper.
* Learner interactions are ordered by timestamp before being fed to the model.
