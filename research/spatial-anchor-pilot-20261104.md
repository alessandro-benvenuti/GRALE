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
