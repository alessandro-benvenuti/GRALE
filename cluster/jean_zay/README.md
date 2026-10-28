# Vascular reconstruction pilot on Jean Zay

Branch: `experiment/vascular-reconstruction-pilot`.

This first step prepares the environment and graph dataset. It does not launch
autoencoder training. The continuous-coordinate model, loss integration and
spatial reconstruction evaluator are the next implementation step. Do not send
the exported NPZ archives to the upstream categorical HDF5 `train.py` loader.

## Storage and environment

The layout follows Graph-Native-Betti-Matching's working module setup, using a
separate GRALE environment. Existing GNBM environments are never modified.

| Item | Default location |
|---|---|
| Source checkout | `$WORK/projects/GRALE` |
| A100 environment | `$SCRATCH/venvs/grale-a100-torch230` |
| H100 environment | `$SCRATCH/venvs/grale-h100-torch231` |
| Package/plot/model caches | `$SCRATCH/.cache/grale` |
| Temporary files | `$JOBSCRATCH`, or `$SCRATCH/tmp/grale-$USER` |
| Prepared graph dataset | `$SCRATCH/datasets/grale/synthetic-mri-pilot-v1` |
| Slurm logs | `$WORK/logs/grale` |
| Package/module records | `$WORK/environment-records/grale-a100` or `grale-h100` |
| Experiment outputs | `$WORK/experiments/grale-vascular-pilot` |

SCRATCH exports and environments are rebuildable. Keep the small manifests,
reports and environment records in durable project storage, and preserve any
important checkpoints outside temporary storage according to project policy.

Setup uses `arch/a100` + `pytorch-gpu/py3/2.3.0`, or `arch/h100` +
`pytorch-gpu/py3/2.3.1`. The virtual environment inherits module packages; PyTorch
is not installed into it. NumPy, SciPy and the Torch family are constrained to
the loaded module versions during dependency installation. The optional existing
`$WORK/tools/uv/uv` creates the venv; standard `python -m venv` is the fallback.
Both routes store the environment in SCRATCH.

Use a fresh login shell when switching architecture profiles. If you explicitly
set `GRALE_VENV`, it must resolve inside SCRATCH and match the selected module.

## Dataset and split recovered from the previous project

The recent GNBM runs use:

```text
/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
```

This is the corrected boundary graph dataset, not legacy `patches/` or the
uncorrected `new_patches/`. The saved patient assignments are in:

```text
/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_split.csv
```

The documented generated split has 95 training, 20 validation and 21 test
patients (70/15/15 rounded across 136 patients). Reuse those assignments.
The old `splits.csv` with 55/14/67 patients is a different benchmark.
These paths/protocol were recovered from code and documentation, not inspected
on Lustre from the local Mac. The preparation job verifies them on Jean Zay.

`configs/vascular/pilot_data.json` fixes the patch selection:

- 4,000 training, 1,000 validation and 5,000 reserved test patches;
- SHA256 seeded membership with seed **364505**, identical to GNBM's
  `select_capped_records(..., mode='seeded_random')` contract;
- selection within each existing split, independent of graph properties;
- normalized D/H/W coordinates (`voxel_index / axis_size`) in 64-cubed patches;
- no Laplacian positional features.

The patient-generation seed (42 by default) and patch-selection seed (364505)
serve different purposes; the existing patient split is not regenerated.
The script checks every patch-index patient assignment against `new_split.csv`,
the generation metadata's split fingerprint and patch size, and the dataset's
completion marker. It fails on mismatches instead of falling back to old data.

## Commands on the login node

For a first checkout on Jean Zay:

```bash
mkdir -p "$WORK/projects"
git clone --branch experiment/vascular-reconstruction-pilot \
  https://github.com/alessandro-benvenuti/GRALE.git "$WORK/projects/GRALE"
```

From an existing GRALE checkout, fetch the branch, then set up and prepare:

```bash
cd "$WORK/projects/GRALE"
git fetch origin
git switch experiment/vascular-reconstruction-pilot

# One-time installation; no GPU requested. Internet/package access is needed here.
bash cluster/jean_zay/setup_environment.sh a100
source cluster/jean_zay/env.sh a100

# CPU data smoke: 8 train + 8 validation graphs, full 5,000-entry test manifest.
bash cluster/jean_zay/submit_prepare.sh smoke
```

Monitor the returned job ID using `squeue` / `sacct`, then inspect:

```bash
cat "${GRALE_PREPARED_DATA}-smoke/report.md"
cat "${GRALE_PREPARED_DATA}-smoke/COMPLETED.json"
```

If the smoke is successful, submit the full CPU export:

```bash
bash cluster/jean_zay/submit_prepare.sh full
```

The CPU job uses account `vnc@cpu`, QoS `qos_cpu-t4`, two CPUs, 16 GB RAM and a
four-hour limit. It loads the corresponding PyTorch module without an `arch/*`
module, following the previous project's CPU analysis jobs. No GPU is reserved.

Separately, verify the GPU environment in a bounded allocation:

```bash
bash cluster/jean_zay/submit_environment_smoke.sh a100
```

This uses `vnc@a100`, partition `gpu_p5`, one A100 and the development QoS for
ten minutes. It imports GRALE/Lightning and checks a CUDA forward/backward pass.
It does not train a model. An environment record is saved under
`$GRALE_OUTPUT_DIR/environment-smoke-JOBID.json`.

For H100, use a fresh shell and replace `a100` by `h100` in setup, activation
and GPU smoke commands (`vnc@h100`, `gpu_p6`). CPU preparation can use either
profile. Do not mix environments across modules.

Useful overrides, set before activation/submission:

```bash
export GRALE_REPO_DIR="$WORK/projects/GRALE"
export GRALE_DATASET_ROOT=/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
export GRALE_PREPARED_DATA="$SCRATCH/datasets/grale/synthetic-mri-pilot-v1"
```

`GRALE_DATA_CONFIG` can point to a reviewed replacement JSON configuration.
`GRALE_PREPARE_OUTPUT` overrides one job's destination. Output directories must
be new; completed or interrupted exports are not overwritten. For an interrupted
job, preserve its partial output and select a new output directory before retrying.
Package downloads occur during login-node setup; compute jobs do not install
packages. W&B compute logging defaults to offline.

## What the preparation job extracts

It reads existing VTP graph labels, without loading MRI/segmentation arrays or
regenerating source patches. It retains disconnected components, isolated nodes,
sparse graphs and empty graphs. It rejects malformed coordinates, invalid edge
indices, duplicate undirected edges and self-loops rather than silently editing
the target. No selected graph is filtered or truncated by node count.

Outputs:

```text
resolved_config.json
manifests/{train,val,test}.csv
graphs/{train,val}.npz
graph_statistics.csv
report.json
report.md
COMPLETED.json
```

The report contains source fingerprints, exact selected patch/patient counts,
node/edge distributions, endpoints, bifurcations, isolated nodes, components and
cycles. Its minimum required node capacity comes from train/validation only.
Inspect it before fixing the decoder capacity, batch size and training budget.

Test entries are reserved in `manifests/test.csv`; their VTP contents are not
read or exported by the development launcher. Keep this set for the frozen
evaluation protocol, including the later image-to-latent experiment. The Python
CLI has an explicit `--include-test-graphs` option for that later evaluation;
it must not be added to development submissions for tuning.

The NPZ format uses no pickled/object arrays and preserves each variable-sized
graph without padding:

- `sample_ids`, `patient_ids`: strings in manifest order;
- `node_features`: concatenated `[total_nodes, 3]` float32 normalized D/H/W;
- `node_offsets`: graph boundaries in `node_features`;
- `edge_index`: concatenated `[total_edges, 2]` int64 **local graph indices**;
- `edge_offsets`: graph boundaries in `edge_index`;
- `sp_values`: concatenated flattened uint32 shortest-path matrices;
- `sp_offsets`: matrix boundaries; reshape a graph's slice to `[n, n]`;
- unreachable shortest-path value: `n` for that graph (diagonal is zero).

Node existence is implicit in these offsets. The future adapter will pad graphs,
construct masks/adjacency and place coordinates into GRALE's continuous feature
path. Archives can be opened with `numpy.load(path, allow_pickle=False)`.

## Local validation

```bash
python3 -m unittest discover -s tests -v
bash -c 'for f in cluster/jean_zay/*.sh cluster/jean_zay/*.slurm; do bash -n "$f" || exit 1; done'
```

Tests cover selection reproducibility, patient-split integrity, held-out test
isolation, graph geometry/connectivity, shortest paths, archive round trips and
SCRATCH environment defaults. The VTK reader check runs when VTK is available.
