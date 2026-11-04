# Geometric correspondence pilot, 2026-11-04

The user explicitly requested Hungarian and fused Gromov–Wasserstein training
experiments in place of GRALE's learned transport. The hypothesis is that hard
one-to-one supervision avoids the transport-averaging coordinate targets observed
in the previous initialization pilot. FGW may additionally help by considering
predicted/GT adjacency, but its non-convex solve and unreliable early edge scores
can also hurt. This is a bounded comparison, not a hyperparameter search.

## Frozen comparison

Three fresh seed-1234 arms: original learned L1 Sinkhorn, geometric Hungarian,
and partial FGW with global one-to-one hardening. All use the original decoder
initialization, 64 slots, no spatial anchors, the same frozen 10-train/4-validation
cohort (8/3 nonempty), GT graphs limited to 20 nodes, original reconstruction-loss
weights, batch size 4 (4/4/2), 5,000 optimizer updates, LR 1e-4, 50-step LR warmup
and cosine decay to 5k. The effective AdamW decay remains 0.01. The original
200k reference run is used for frozen data/configuration, not weights or resume.
No test data, augmentation, GT-count decoding or fixed index correspondence.

Full initial-parameter hashes and minibatch-order hashes must agree across arms.
The original learned matcher remains registered to preserve identical parameter
initialization/checkpoint compatibility. It is bypassed and receives no matching
gradient in the hard-assignment arms. Their coordinate/edge losses still receive
gradients through native predictions. Core training/validation/forward/canonical
APIs dispatch to the configured `loss_assignment`, including after checkpoint reload.

Hungarian uses `5 * L1(predicted XY, GT XY) - 2 * sigmoid(node logit)`, with XY
normalized by patch dimensions. FGW uses the same unary cost, symmetric predicted
adjacency probabilities from GRALE's inference head, and symmetric zero-diagonal
binary GT adjacency. It uses alpha 0.4, 32 candidates containing every unary
Hungarian match plus highest-confidence unused queries, uniform GT mass and
one-target-per-query capacity, 10,000 solver iterations maximum and tolerance
1e-7. The coupling starts at a feasible unary Hungarian solution. A global
Hungarian projection of its negative coupling yields distinct queries for all
targets. Unused output slots are paired deterministically with padding columns
to supply the existing GRALE objective a full row/column-unit permutation matrix.
Padding never attracts a coordinate target. Assignments are recomputed each pass.

The unmodified reference matcher is copied from Graph-Native-Betti-Matching
commit `9cc28da9d5324a6beaa5fcef6bed5f629f69a08a`, file SHA256
`c179c3492981d0aa684a7cf1d272fbcd77979c63fa683106e89b13aad2995660`.
See [source provenance](geometric-matcher-reference-20261104.json). The historical
FGW branch pilot used alpha 0.4 with an epoch schedule; this fresh tiny-patch
comparison deliberately fixes alpha from the start rather than transplanting a
different dataset's epoch schedule. This changes cost, hardening and gradient
path together relative to learned Sinkhorn; superiority would not isolate only
the effect of hardening. Hungarian versus FGW is the graph-aware comparison.

## Data and environment

Frozen subset SHA256:
`c199e3a4d18229b067a3b779c6f17df5941624a5a94c3d77a86cbedb6484ddec`.
Reference:
`$WORK/experiments/grale-octa-tiny-l1-max20-200k/20261029-171558/tiny/octa_straight_l1_200k`.
The missing original prepared ZIP is not regenerated or claimed hash-verified.
Use its saved inspection `inspection-20261030-104603` for original GT coordinates,
simple edges and metric, with the existing strict identity/count/geometry checks.
The previous CPU replay verified 14/14 original predictions from these inputs.

Reuse the existing SCRATCH GRALE A100 venv and numerical stack. POT 0.9.7 is copied
from the user's existing GNBM A100 environment into a new isolated directory:
`$SCRATCH/venvs/grale-matcher-pilot-deps-20261104`. Only its `ot`, `pot.libs`, and
distribution metadata are copied; existing environments are unchanged. A package
provenance file hashes every copied file and is verified inside the job. POT uses
its NumPy backend; optional Torch/JAX/TensorFlow/CuPy backends are disabled only
for POT. Local validation installs POT only into a temporary directory.

## Execution and measurements

One isolated A100 job, four requested CPU cores, 90-minute wall limit, three
sequential arms and at most 15,000 optimizer updates. No automatic chaining.
Source: `$WORK/projects/GRALE-geometric-matchers-20261104`.
Output: `$WORK/experiments/grale-geometric-matchers-20261104/paired-5000`.
Existing remote checkouts, completed runs and shared datasets remain unchanged.
Check both queue estimates before submitting; use a single active pilot.

At 0/50/200/500/1000/2000/3000/5000, evaluate native thresholded predictions on all
14 graphs, independently of the training assignment: node/edge F1, coordinate
RMSE and coverage, exact reconstruction, counts, components/cycles, empty graphs,
selected-node versus all-slot spread, and target-average residual. Hard assignment
concentration equals one by construction and is not evidence of learning.
Save all hard plans and predictions, three fixed example plots, and FGW's
pre-hardening coupling, candidates and solver diagnostics. Log changed GT matches
versus unary Hungarian, solver warnings, timing, losses and gradients. Solver
warnings fail explicitly; there is no silent Hungarian fallback. Compare fixed
5k endpoints, retaining unfavorable/partial outcomes. One seed/tiny cohort does
not establish generalization or systematic FGW superiority.

## Validation before deployment

A three-update generated-fixture run of all three arms completed with finite
backward, identical initial-parameter/order hashes, and no solver warnings.
All three checkpoints reload; public model APIs and the native pilot forward
agree, and the learned control retains the original objective. Analytic checks
cover noncontiguous GT padding, matrix orientation, known coordinate gradients,
edge-gradient signs, empty graphs, alpha-zero equivalence, represented mass for
15 targets, and a graph-structure disambiguation example. Float32 analytic
gradients include GRALE's existing division by sqrt(feature dimension).

Sources checked: [SciPy assignment API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)
and [POT partial FGW API](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.partial_fused_gromov_wasserstein).
Actual submission IDs, accounting and results are recorded after execution.
