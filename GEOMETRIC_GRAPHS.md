# Straight OCTA graph reconstruction

This implements the initial [experiment plan](EXPERIMENT_PLAN.md): graphs in, graphs out; XY coordinates, binary connectivity, and shortest-path reconstruction. Splines, radii, images as input, and 3D are deferred. COLORING defaults remain available.

## Implementation

- [GeometricDataset](GRALE/data/geometry.py) reads the existing manifest and NPZ ZIP archive. It preserves source splits, node identities, coordinates and edges; it does not skeletonize masks or refit targets. Dense 64-slot batches are constructed at loading. Overflow raises an error.
- [Encoder](GRALE/model/encoder.py) consumes `nodes.features` as normalized XY. Empty inputs use a learned empty embedding rather than fully masked attention. The decoder receives only graph embeddings.
- [Target builder](GRALE/model/target_builder.py) reconstructs XY, binary adjacency/edge labels, and hop distances divided by the allocated capacity. Unreachable distances use the capacity as a finite sentinel.
- [Decoder](GRALE/model/decoder.py) has no categorical node output in geometric mode. Pair logits/features are symmetrized. Predictions use node existence and adjacency probabilities at 0.5, without a GT node count.
- [Loss](GRALE/loss/objective.py) retains the existing transport-aware BCE/L2 terms. Empty targets contribute existence/marginal supervision and zero node/pair attribute cost. Coefficients are in the YAMLs; node classification is disabled, not replaced by artificial coordinate classes.
- The [Laplacian variant](GRALE/configs/octa_straight_laplacian.yaml) computes up to 16 positive-eigenvalue modes on active nodes only, fixes signs by largest-magnitude entries, and zero-fills unavailable channels. Repeated eigenspaces can still be ambiguous. The [coordinate-only variant](GRALE/configs/octa_straight.yaml) uses none.

## Run

For the ten-patch, **at most 20 nodes per patch, 200,000 optimizer-step,
learned L1** experiment on Jean Zay,
use the dedicated module-backed SCRATCH environment and submission wrapper:

```bash
cd "$WORK/projects/GRALE"
# When switching from the A100 environment, deactivate it first.
deactivate 2>/dev/null || true
unset GRALE_VENV
bash cluster/jean_zay/setup_environment.sh h100
bash cluster/jean_zay/submit_geometric_tiny.sh
```

Setup creates/reuses `$SCRATCH/venvs/grale-h100-torch231`, inheriting
`arch/h100` and `pytorch-gpu/py3/2.3.1`. The wrapper activates this environment
for input checks, and the compute job activates it again before training.
The source dataset root is
`/lustre/fsn1/projects/rech/vnc/upz25mj/datasets` (override with
`GRALE_GEOMETRIC_DATASET_ROOT`). This directory contains source datasets; the
loader needs the prepared OCTA NPZ/ZIP manifest described below. The wrapper
checks that preparation layout under the dataset root and the sibling GNBM
checkout, then the same prepared cache under the
original `/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/` checkout
recorded by the research audit, and uses it only if readable. If relocated,
pass its actual path as the wrapper's first argument, or set
`GRALE_GEOMETRIC_MANIFEST`. The manifest's `data_root` and referenced ZIPs must
also be accessible. This launcher does not consume the 3D MRI NPZ export.

The [configuration](GRALE/configs/octa_straight_l1_200k.yaml) retains the
coordinate-only model, learned L1 Sinkhorn cost, loss weights, seed 1234,
learning rate 1e-4 and 50-step warmup. The cosine schedule spans 200,000 updates.
`selection_max_nodes: 20` / `--max-nodes 20` filters **both train and validation**
before sampling. Graphs with 20 nodes remain eligible; larger graphs are
excluded whole, never truncated. The model still allocates 64 slots, preserving
the architecture, padding and distance normalization. The frozen subset stores
the cutoff and source/eligible split counts; changing it requires a fresh output.
The frozen selection has eight nonempty/two empty training graphs and four
validation graphs; batches contain 4, 4 and 2 training graphs per epoch.
Original splits are retained and no test targets are read. The inherited
effective AdamW decay remains 0.01; this run does not change the optimizer.

Before submitting, the wrapper imports/checks the environment and reads all
14 selected targets on the login node with `--check-inputs`, without creating
a model or using a GPU. The frozen subset, effective config and input report
are saved alongside the experiment. No package installation runs in the job.
One H100 is requested, with a 19-hour training timer and 19h30 allocation.
The job reports failure if it finishes short of 200,000 updates, retaining
its checkpoint and metrics for diagnosis or continuation.

Outputs default to a new timestamped directory under
`$WORK/experiments/grale-octa-tiny-l1-max20-200k/`; override with `GRALE_TINY_OUTPUT`
before submission if needed. The wrapper prints the output path and saves
the Slurm ID in `job-id.txt`. Logs go to `$WORK/logs/grale/`. Every 10,000
updates it saves a checkpoint and three reconstruction examples; complete
train/validation evaluation and `last.ckpt` are saved at the end. Runtime
completion does not establish reconstruction quality: inspect `result.json`
and `train/metrics.json` under `tiny/octa_straight_l1_200k/`.

The earlier short runs can still use the commands below.

From the repository root, with the existing Jean Zay environment:

```bash
module load arch/h100
module load pytorch-gpu/py3/2.3.1
export IDR_DEBUG=WARN OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/grale-mpl PYTHONDONTWRITEBYTECODE=1

python scripts/run_geometric.py --mode smoke --device cpu
python scripts/run_geometric.py --mode smoke --device cpu --config octa_straight_laplacian
```

The [runner](scripts/run_geometric.py) uses prepared targets under `../GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028/octasynth/`. Override with `--manifest /path/to/manifest.json` if relocated. The input schema is the existing native 2D manifest, with `train` and `validation` membership and XY NPZ targets.

On an allocated GPU worker, the same entry point runs the bounded experiments:

```bash
python scripts/run_geometric.py --mode tiny --device cuda --output logs/geometric-fit
python scripts/run_geometric.py --mode tiny --device cuda --config octa_straight_laplacian --output logs/geometric-fit
# Only after the tiny-set fit is informative:
python scripts/run_geometric.py --mode pilot --device cuda --output logs/geometric-fit
python scripts/run_geometric.py --mode pilot --device cuda --config octa_straight_laplacian --output logs/geometric-fit
```

`smoke` runs 2 optimizer steps, `tiny` up to 1,000, and `pilot` up to 2,000, capped at 30 minutes. `--steps 20` permits a deliberately shorter runtime check, not evidence of convergence. CPU batches are limited to 2; GPU batches use 4. The pilot warmup is shortened to 50 steps, and further shortened for explicit short runs. No spatial augmentation is applied.

Subsets are frozen under `logs/geometric/subsets/`, shared by both positional variants, with source/archive hashes. Tiny uses 8 nonempty + 2 empty training graphs; pilot uses 256 training / 128 validation crops. Source membership never changes and no test targets are opened. Sampling covers source/size groups and records the empty population. Data order uses its own fixed seed, independent of model initialization.

Use `--output logs/another-run` for a new experiment. The runner refuses to overwrite a completed checkpoint or silently reuse a subset whose source/archive identity changed.

To recover evaluation/export from an already trained checkpoint without retraining, use the same mode, configuration and output root with `--evaluate-only`, for example:

```bash
python scripts/run_geometric.py --mode tiny --config octa_straight --output logs/geometric-fit-20261029 --device cuda --evaluate-only
```

This preserves the checkpoint and original training config. Recovery results mark `evaluation_only=true`; unavailable training timings/gradient counts are not reconstructed. After Lightning validation, the runner explicitly restores the requested device before checkpoint comparison and inference.

To continue a completed tiny fit for another 100,000 updates:

```bash
python scripts/run_geometric.py --mode tiny --config octa_straight --device cuda \
    --resume logs/geometric-overfit-20261029/tiny/octa_straight/last.ckpt \
    --steps 110000 --max-time 00:05:00:00 --output logs/geometric-continue-20261029 \
    --checkpoint-every 10000 --snapshot-every 10000
```

`--steps` is the **total** optimizer-step target, including the checkpoint's 10,000 steps. Resumption restores weights and Adam state, retains the checkpoint's model/loss/batch settings, and extends cosine decay to the new total. The first resumed update uses the extended schedule (about `9.8e-5` here), because the finished schedule saved a zero learning rate. This is a learning-rate reactivation, not an uninterrupted original schedule. The shuffled loader restarts from its seed; exact continuation of the old sample order is not claimed. Reuse the byte-identical frozen subset and a new output root to preserve prior evidence.

Periodic checkpoints go under `tiny/<config>/checkpoints/`. Three fixed GT/prediction pairs are saved at the starting step and every 10,000 steps under `tiny/<config>/intermediate/step-XXXXXX/`: the first nonempty training graph, densest training graph, and first nonempty validation graph. Each folder contains PNGs, predicted NPZ graphs, and per-example metrics. These diagnostic examples are not whole-dataset evaluation. The existing full train/validation export runs at completion. Plots label node/edge counts and retain out-of-frame nodes using shared axes.

The original `train.py` also accepts these configurations and a prepared or frozen geometric manifest as `--dataset_path`. For the planned bounded comparison, prefer the runner because it saves the subsets, predictions, baselines and evaluation artifacts together.

## Outputs and interpretation

Each run saves its effective config, CSV losses, checkpoint, environment/code/data identity, inference checks and timings. `train/` and `valid/` contain predicted NPZ graphs, overlays, and `metrics.json`. Yellow overlays are supplied guide paths; cyan lines are straight chords. Predictions include selected XY/edges, node/edge confidence, all-slot existence scores, crop origin, source and pixel metric.

Metrics separate empty/nonempty graphs, source IDs and size bins. Node association uses minimum total pixel distance; matches within 2 pixels count toward node/edge F1. Coordinate RMSE includes every assigned pair before the tolerance filter, with coverage reported; missing localization is `null`, never a favourable zero. Exact recovery also requires correct node count and connectivity. Learned-matcher localization is reported separately.

Scientific aggregate metrics are in the saved evaluator's `metrics.json`, calculated from raw counts/errors over the entire subset. CSV values prefixed `batch/` are step diagnostics only, not epoch F1/accuracy/RMSE. The additive `edit_graph` diagnostic and losses retain epoch logging.

The evaluator records all-empty and fixed training-prototype baselines, predicted-chord versus target-chord distances, and the target-chord versus supplied-guide approximation floor. Guides are existing graph polylines, not new raster-centreline GT. The saved `tiny_fit_gate` checks the plan's tiny-fit criteria; finite execution alone does not pass that quality gate.

Verification uses real OCTA cases, including empty/disconnected graphs, coordinate/edge round trips, one backward/update path, active-PE padding invariance, identity scoring and checkpoint reload. No general test framework, CI, broad boundary-case suite or full training is introduced.

## Verification

See the [consolidated research record](research/overview.md#validation) for CPU/GPU checks, the completed 1k/10k tiny fits and their unmet reconstruction gate, and the two-update GPU verification of optimizer resumption and snapshots. Runtime checks do not establish learned reconstruction; no held-out pilot has been performed.
