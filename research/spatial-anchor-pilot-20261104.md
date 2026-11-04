# Spatial-initialization pilot, 2026-11-04

User request: test whether spreading node predictions over the image at the
start of training prevents the observed learned-L1 matching collapse, execute
bounded experiments on Jean Zay, and report progress.

## Frozen comparison

- Reference: `$WORK/experiments/grale-octa-tiny-l1-max20-200k/20261029-171558/tiny/octa_straight_l1_200k`.
- Reuse the exact frozen ten training/four validation OCTA graphs, at most 20
  nodes, with eight/three nonempty and two/one empty. No resampling or test data.
- Subset SHA256: `c199e3a4d18229b067a3b779c6f17df5941624a5a94c3d77a86cbedb6484ddec`.
- Preserve 64 slots, coordinate-only input, learned L1 Sinkhorn correspondence,
  existing loss coefficients, epsilon 1e-4, cost-sum normalization, 50 Sinkhorn
  iterations, batch size 4 (4/4/2), base LR 1e-4, gradient clipping and effective
  AdamW decay 0.01. No fixed correspondence or spatial matching-cost term.
- Fresh seed-1234 fits, 5,000 optimizer updates per arm, 50-step LR warmup and
  cosine decay to the common 5k horizon. These are paired short fits, not
  continuations or reproductions of the 200k learning-rate schedule.

| Arm | Coordinate initialization | Initial residual constraint |
|---|---|---|
| default | Original decoder, unchanged | None |
| center | All slots at (0.5, 0.5), zero final coordinate-head layer | None |
| spread | One randomly jittered anchor per 8x8 grid cell, seeded slot permutation, zero final coordinate-head layer | None |
| spread_warmup | Same anchors and weights as spread | Residual scale zero for the initial hold, then ramps from zero to one |

Anchored coordinates are `anchor + scale * learned_residual`. Anchors are fixed
buffers, drawn once without consuming the model's global RNG, shared across
graphs, and never use GT. The warmup scale is
`clip((update_index - 500) / 500, 0, 1)`, using the zero-based update index.
Thus the residual is fully released by update 1001. All arms subsequently have
unrestricted coordinate predictions; there is no permanent spatial-spread loss.
Holding coordinates gates residual gradients; the other heads and learned
matcher continue training. The optimizer and its decay behavior are retained.

All parameters other than the final coordinate-head layer must have identical
initial hashes. Center and the two spread arms have identical full parameter
hashes; their fixed buffers/scales differ. Training batch-order hashes must match
across all arms. The center arm controls for zero-initializing the head, so its
effect is not attributed to spatial spreading.

## Measurements and interpretation

At 0, 50, 200, 500, 1000, 2000, 3000 and 5000 updates, evaluate all 14 graphs:
node/edge F1, exact reconstruction, node counts, coordinate RMSE and coverage,
spatial spread/centroid errors, real-node matching concentration and entropy,
and residual to transport-weighted GT averages. Save graph predictions and
transport arrays; plot three fixed examples and aggregate learning curves.
Log all training loss components, LR, residual scale and gradient norm.

The original tiny-fit gate requires exact nonempty topology/localization,
training RMSE <=1 pixel, and exact empty predictions. Improvement should remain
after residual release and involve correct localization/edges, rather than
spread alone. A poor short fit does not establish impossibility at other budgets
or seeds. One seed and this tiny cohort do not establish population performance.

The matcher still uses learned embeddings rather than predicted XY. Anchor
initialization changes the coordinate loss seen during optimization, not the
initial matching cost. Sharpness and averaging measurements distinguish a
temporary geometric constraint from learned correspondence.

This implementation tests fixed spatial anchors plus unconstrained learned
offsets, with an optional temporary offset hold. The additive anchors persist
as a slot-specific parameterization even after release; a positive outcome
would support this parameterization, rather than isolate arbitrary XY-head
weight initialization alone. Report selected-node spread separately from all
64 coordinate slots: inactive slots can remain dispersed while selected nodes
still collapse. Correct counts and global slot spread are not reconstruction.

## Execution boundaries

One isolated H100 job, four CPU cores, 90-minute wall limit, four sequential
arms and at most 20,000 total optimizer updates. No automatic chained jobs.
Source goes in `$WORK/projects/GRALE-spatial-anchor-20261104`; outputs go under
`$WORK/experiments/grale-spatial-anchor-20261104`. Reuse the existing H100
SCRATCH venv and shared targets read-only. Do not modify/pull/switch the existing
remote GRALE checkout, colleague files, environments or completed runs.

The requested Eurecom nested connection was verified to `jean-zay2`, account
`upz73jr`, after the user privately reloaded its dedicated key. Live Slurm showed
the previous training/inspection jobs COMPLETED, exit0:0, and H100 t3 permits
20 hours. Official IDRIS page retrieval was attempted but timed out; use a
Slurm test-only validation before actual submission. No explicit memory request.

## Local validation

A three-update CPU fixture exercises all arms, the warmup release, finite
backward, snapshot/transport exports, shared parameter hashes and identical
batch-order hashes. Separate probes check original-decoder parameter/output
parity, one fixed anchor per grid cell, unchanged initial matching embeddings,
and checkpoint reload. Local fixture outcomes are not experimental findings on
the frozen OCTA cohort. Remote submission and results are recorded after
execution, not inferred from these tests.

## Remote submission and A100 fallback

The frozen scientific source is commit
`d66affa17bc835142b1d0f4b2c47a49dfcc4b988`, source-archive SHA256
`1a6609b0efadd8f73020320d0368bf3aa711c29fbce15435fa72ddfb0e823626`.
It was deployed into the isolated directory above without changing the existing
remote GRALE checkout. The archive identity and source commit are saved there.

H100 development-queue job **1951843** was submitted at 14:35 Paris time.
It remained pending with `QOSGrpGRES`. After the user requested trying A100 if
waiting, both development queues and regular A100 were checked using Slurm
test-only submissions. At 14:43, A100 estimated 14:49 versus H100 14:54; regular
A100 estimated the next morning. These estimates are not guaranteed starts.

The H100 job was cancelled while pending, with zero elapsed time; absence from
the queue and accounting's cancellation were checked before submitting A100.
Replacement **1952135** requests one A100, four CPU cores, 90 minutes,
`qos_gpu_a100-dev`, and the same four sequential 5k-step arms. There is one
active pilot and no automatic chaining. The fallback changes hardware and its
existing environment (PyTorch 2.3.0 versus H100 2.3.1); all four compared arms
run on the same hardware/environment. It does not change the frozen scientific
source or original cohort.

- Batch: `cluster/jean_zay/spatial_anchor_pilot_a100.slurm`, SHA256
  `ddf770275400c7c3fac5142f38c15b589b675e29e760ea21881ced449af0ecb0`.
  The identical remote wrapper is saved separately from the frozen source as
  `$WORK/experiments/grale-spatial-anchor-20261104/a100-fallback.slurm`.
- Existing venv: `$SCRATCH/venvs/grale-a100-torch230`; no installation or update.
- New output: `$WORK/experiments/grale-spatial-anchor-20261104/paired-5000-a100`.
- Logs: `$WORK/logs/grale/grale-anchor-pilot-1952135.{out,err}`.
- Remote records: `submission.json` preserves the initial H100 submission;
  `submission-a100.json` records its replacement. Both are under the new
  experiment parent directory.

At submission the A100 job was pending. Completion and experimental outcomes
must be established from its actual logs, endpoint result and accounting.

### Missing-source infrastructure failure and exact training-target recovery

Job 1952135 passed the environment/GPU check (A100 80GB, existing SCRATCH venv)
but failed before creating the training output or performing optimizer updates:
the frozen manifest's shared `data_root` no longer exists. Accounting: FAILED,
exit 1:0, elapsed 69 seconds. The original archive hash could not be verified.
The user's suggested `.../datasets/octa-synth-packed` contains source CSV graphs
and a different patch layout, but not the prepared archive at the examined paths.
No raw-data regeneration or replacement cohort was used.

The original inspection folder `inspection-20261030-104603` retains all 14
targets: original float32 training nodes, decoded simple undirected edges and
pixel metric. Its report verifies the original frozen-subset SHA256 and lists
no mismatches against saved reference predictions. An explicit optional
`--inspection-targets` loader reuses these arrays without changing the frozen
manifest. It checks all sample/split identities, node/edge counts, edge validity,
metric, finite coordinates and spatial spread against that report, and records
SHA256 for the report and every target artifact. Original images and curved
paths remain unavailable and are not inputs or scoring targets for this pilot.
The original archive hash is explicitly recorded as **unverified**, rather than
claiming archive-level identity from the recovered training arrays.

Local validation compared every model-input tensor for all 14 generated fixture
graphs (activations, coordinates, adjacency, edge labels and shortest paths):
recovery versus the original ZIP loader were exactly equal. A three-step fit of
all four arms through the recovery mode completed. The default archive-loading
path retains its hash check. The new frozen source/output will be isolated from
the failed attempt. A replacement A100 job is limited to 88 minutes, keeping
the combined maximum allocation below the original 90-minute budget including
the 69-second startup failure; there is still no automatic chaining.

Replacement job **1952925** was submitted at 14:56:59 Paris time. Scientific
source commit: `41d18afe61dd01d365b1ea8d41dac539e83f4442`; archive SHA256:
`0b9023a345bdeb18fc8ceb9e63affa253baaba5b9e1ac42ccda0e52cc38ff696`.
The new isolated source is `$WORK/projects/GRALE-spatial-anchor-recovery-20261104`;
new output is
`$WORK/experiments/grale-spatial-anchor-20261104/paired-5000-a100-recovered`.
The parent directory's `submission-recovery.json` records the source, batch,
reference, recovered target directory, resource request and preceding failure.
Logs: `$WORK/logs/grale/grale-anchor-pilot-1952925.{out,err}`.

An initial source upload exceeded the local SSH terminal's input-line limit and
did not append any data or deploy code. The connection was reestablished using
the user's Duo approval. Resending as short lines succeeded; the entire archive
checksum was verified before extraction/submission. Failed and earlier pending
attempts remain preserved. No original remote checkout, environment or shared
source data was modified.

### Independent reference replay

A separate bounded diagnostic, job **1954449**, requested one CPU core, zero
GPUs and five minutes in `qos_cpu-dev`. It leaves the running GPU pilot unchanged.
It loads the original 200k checkpoint in the original H100 PyTorch 2.3.1 venv,
on CPU, from the recovered inputs, verifies the checkpoint/subset identities,
and compares native decoded coordinates/slots/edges against every saved graph.
All **14/14 predictions reproduced** (coordinate tolerance `atol=rtol=1e-5`,
exact slots and edges). This supports recovery of the model inputs even though
the original ZIP's byte identity remains unavailable. The source script is
`scripts/verify_recovered_reference.py`, SHA256
`eec21b81f9d871dbc4abdb2d8a692a249e92d38a751da6cd1188d1611c15642b`.
The remote aggregate verification is
`$WORK/experiments/grale-spatial-anchor-20261104/reference-replay.json`; its
code, batch and submission record are preserved alongside it. No raw targets,
predictions, checkpoints or image data were transferred away from the cluster.
Accounting verified COMPLETED, exit 0:0, elapsed 45 seconds. Recomputed train
RMSE was 56.3630781 px and validation RMSE 56.8976882 px, consistent with the
original run within CPU floating-point tolerance.
