# Experiment plan: GRALE for sparse geometric graphs

Date: 2026-10-29. **Planning only: do not implement or launch training from this document without a subsequent instruction.** This consolidates the [design discussion](research/grilling-design.md); settings and acceptance thresholds labeled proposed are not previously agreed facts.

## Scientific motivation

Determine whether GRALE can reconstruct sparse vessel graphs while preserving node positions and connectivity through a fixed-size graph embedding. This is a graph-only autoencoding task, without an image encoder or a shortcut carrying input nodes/edges to the decoder.

Dense centrelines preserve vessel shape but require many predicted nodes. Junction/termination graphs reduce that burden, while straight chords lose tortuous branch geometry. A later continuous-edge representation could retain shape with fewer graph nodes. **First establish straight-graph reconstruction; only then investigate splines.** Successful reconstruction is a prerequisite for later image-conditioned graph prediction, not evidence that image prediction will work.

## Established decisions

| Decision | Agreed scope |
| --- | --- |
| Starting implementation | Current GRALE repository, with minimum necessary geometric-data changes; do not first reproduce the paper's different featurizer. |
| Input and target | Compatible deterministic graph representations from the same supplied annotations; initially sparse straight graphs. |
| First dataset | Existing OCTA-Synth patches and supplied connectivity; no new mask skeletonization. |
| Coordinates | XY only; ignore source Z. Preserve spatial placement/orientation for eventual image alignment. |
| Connectivity | Binary edge/no-edge, undirected; retain current dense GRALE computations. |
| Initial variants | Straight/coordinates and straight/coordinates-plus-Laplacian, as an immediate comparison. Verify coordinates first. |
| Reconstruction pattern | Follow current implementation, including shortest-path reconstruction. Avoid an unnecessary objective redesign. |
| Empty graphs | Include them with correct handling; report empty and nonempty performance separately. |
| Deferred work | Full images/volumes, 3D, degree-two node-budget comparisons, radii, splines and image-conditioned prediction. |

For later splines, the agreed candidate is fixed cubic K8, with endpoints shared with graph nodes. Control-coordinate supervision is preferred initially; centreline distance is primarily evaluation. Controls would be reconstructed edge attributes within the same jointly trained autoencoder. The exact spline implementation remains deferred.

## Evidence and data contract

The [upstream COLORING CPU smoke check](research/upstream-coloring-smoke.md) passed: syntax/imports, data loading, three optimizer steps with finite gradients, validation, encode/decode, hard matching and checkpoint reload. Core model/loss source remained unchanged. This establishes runtime feasibility, **not convergence or GPU correctness**. Environment: Python 3.11.9, PyTorch 2.3.1, Lightning 2.4.0; older than README versions. No further full COLORING training is needed now.

Reuse [prepared OCTA targets](../GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028/octasynth/): `manifest.json`, `targets.zip`, `audit.json`. Raw source: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/octa-synth-packed`. [Verified lineage and schema](research/grilling-geometry-evidence.md).

- Prepared cohort: 10,648 training / 4,840 validation crops, 128×128 pixels, from 88 / 40 source groups. The 372 reserved test sources remain unopened and outside pilot selection.
- Training maximum: 52 nodes. Empty crops: 1,503 training / 747 validation. Preserve official source membership; never randomly split neighboring crops across train and validation.
- Each target provides normalized `nodes[N,2]`, `edges[E,2]`, scale metadata, guide paths and existing K8 controls. Initial experiments ignore controls and render edges as endpoint chords. Do not change graph anchors between positional-feature variants.
- Normalization is XY divided by `[W,H]`, not `[W-1,H-1]`; `metric=128 I` converts displacement to pixels. Preserve crop origin and source identity for reversing normalization and plotting.
- Keep existing crop-cut nodes and topology-preserving anchors for loops/parallel paths. Do not merge nodes merely because their projected positions coincide. Exact artificial-boundary flags are absent from these NPZs; do not invent them.
- Guides trace supplied annotation polylines, not independently skeletonized raster centrelines. Graph reconstruction and geometric agreement with those guides are different measurements.

## Minimal implementation approach

Use configuration-selected geometric input/target handling inside existing modules. Keep upstream COLORING behavior available. No new framework, generic adapter hierarchy, CI or extensive test suite.

| Existing location | Necessary work |
| --- | --- |
| [Data loading](GRALE/data/dataset.py), [dense containers](GRALE/data/dense_data.py) | Read manifest/NPZ targets, carry float XY and metadata, construct dense binary relations and finite hop distances at batching. Reuse sparse files and existing tensor containers. |
| [Input builder](GRALE/model/input_builder.py), [encoder](GRALE/model/encoder.py) | Consume XY explicitly instead of atom/color labels. Concatenate optional active-graph Laplacian features; retain endpoint features and sinusoidal hop-distance pair inputs. |
| [Target builder](GRALE/model/target_builder.py) | Preserve XY as continuous node targets; retain binary connectivity and normalized shortest-path targets. Replace molecular label-derived targets only where their meaning is inapplicable. |
| [Decoder](GRALE/model/decoder.py) | Predict node existence, continuous XY and the current binary relation/shortest-path outputs. Straight segments need no separate geometry prediction. |
| [Objective/metrics](GRALE/loss/objective.py) | Retain BCE/L2 primitives and soft transport reconstruction. Skip inapplicable atom-label losses rather than treating XY as classes. Make empty-graph denominators and valid-attribute reductions finite. |
| [Attention](GRALE/attention/utils.py), [encoder](GRALE/model/encoder.py) | Provide a finite empty-graph path, such as an explicit empty embedding. Fully masked softmax is currently unsafe. Do not add a fictitious vessel node. |
| [Positional helper](GRALE/model/utils.py) | Compute PE on active nodes before padding; current padded eigendecomposition mixes dummy zero modes into features. Record mode/sign conventions and disconnected-component behavior. |
| [Orchestration](GRALE/main.py), [matcher](GRALE/model/matcher.py), [training](train.py) | Retain learned soft Sinkhorn training and hard matching at evaluation. Add geometric metrics and a bounded smoke/pilot configuration; no replacement matching method initially. |

The existing binary adjacency and edge-label outputs can remain to minimize changes. Retain shortest paths as input and reconstruction targets; sinusoidal/Laplacian encodings remain input descriptors under the current repository pattern. Log each retained loss separately. Do not silently remove auxiliary terms or alter their weighting between variants.

For empty graphs, supervise absence across output slots; nonexistent node/pair attributes contribute zero. For decoding, use predicted existence rather than GT node count. Declare a symmetric edge-probability rule and thresholds, and remove edges incident to rejected slots. Match GT only for scoring, never to supply decoder output structure.

## Proposed first configuration

These are starting defaults to make the experiment concrete, not claims of optimality:

- `N=64`, selected from training maximum 52; fail visibly on overflow, never truncate. Audit the chosen population before training.
- Begin with the five-layer `demo_checkpoint` model scale, latent `K=4`, `D=128`, unchanged hidden widths and base learning rate `1e-4`. A fixed-size latent is not automatically compression relative to every small graph.
- Continuous node width 2; binary relation classes; retain normalized hop-distance target and marginal regularization. Reuse existing corresponding loss coefficients initially; document inactive categorical node terms and all effective coefficients.
- Compare PE off versus 16 PE channels; compute active-graph modes and zero-fill missing channels. Proposed convention: exclude zero eigenmodes, fix each mode's sign by its largest-magnitude entry, and record repeated-eigenvalue ambiguity. This is a declared PE variant, not an exact reproduction of the old helper.
- Reversible local coordinate normalization; report localization distances in pixels. Start without spatial augmentation to isolate reconstruction. This does not imply that augmentation would require rotation-invariant outputs.
- Same splits, sample IDs, latent size, seed, optimizer and step budget for both variants. Report their parameter counts; the PE input projection can change parameter count slightly.

## Lightweight checks and initial experiments

| Stage | Bounded check | Evidence required |
| --- | --- | --- |
| 0 — existing baseline | Completed COLORING smoke; no full training | Link existing result and environment; do not interpret three steps as learned accuracy. |
| 1 — geometric sanity | A few real OCTA graphs spanning small/large, disconnected and empty cases; one forward/loss/backward/update; checkpoint round trip | Correct shapes, unchanged topology/identity, reversible XY conversion, finite outputs/loss/gradients, parameters update. Verify PE ignores added dummy padding and empty graphs remain finite. |
| 2 — tiny-set fit | Fixed 8 nonempty training graphs plus 2 empty graphs; bounded coordinate-only overfit check, then PE variant | Demonstrate that the actual learned decoder and matcher can recover a known small set. No GT count/topology shortcuts. |
| 3 — held-out pilot | Proposed 256 training / 128 validation crops, one recorded seed, both straight variants | Source-separated reconstruction results, failures and resource measurements. Select subsets across source IDs and graph-size bins, retain a documented empty fraction; never tune on test data. |

Proposed budget: tiny fit up to 1,000 optimizer steps; pilot up to 2,000 steps or 30 minutes of GPU time per variant, whichever comes first. Start with batch size 4 and measure memory/step time before committing that budget. Use a documented short warmup (for example 50 steps), since the upstream 2,000-step warmup would consume this entire pilot. These are explicit smoke/pilot overrides, not full training or paper settings. No sweeps or multiple seeds until one run is informative.

Stop at the first failed sanity gate, make the smallest relevant correction, and repeat that check. If tiny-set recovery fails, inspect loss, matching and optimization before interpreting held-out failure as an architectural limitation.

## Measurements and decision criteria

Use fixed decoding thresholds, initially 0.5 for node existence and binary edge probability. For scoring, assign predicted active nodes to target nodes by minimum total Euclidean distance. Count assigned pairs within 2 pixels as node matches; distant/unassigned nodes and their incident edges contribute errors. Compute localization RMSE over **all assigned pairs before that tolerance filter**, and report assignment coverage, so the RMSE criterion is not automatically satisfied by rejecting distant predictions. With no assigned nodes, report localization as unavailable, never zero. Report learned-matcher diagnostics separately: an external scoring correspondence must not conceal a training-matcher problem. Flag co-located-node/crossing ambiguities.

Primary metrics on **nonempty** graphs: node-count accuracy; node precision/recall/F1; assigned-node localization error in pixels; edge precision/recall/F1; exact graph recovery; component-count and cycle-count agreement. Exact recovery requires correct node count, every node associated within tolerance, and identical connectivity after association. Do not use high nonedge/padding accuracy as evidence of successful vessel reconstruction.

Report empty-graph accuracy and false-positive nodes/edges separately, plus all loss components and auxiliary shortest-path error. Also report peak GPU memory and step time. Plot input/predicted chords over the same patch and retained guide paths.

**Provisional decision gates, not observed results or previously agreed benchmarks:**

| Gate | Criterion | Interpretation |
| --- | --- | --- |
| Runtime | All checked outputs, reductions and backward passes finite, including empty inputs | Runnable geometric adaptation. |
| Tiny-set fit | Correct node counts and exact topology on all 8 nonempty graphs; assigned-node RMSE ≤1 pixel; both empty graphs decode empty | Basic representation/matching/optimization feasibility. |
| Promising pilot | Nonempty node/edge F1 ≥0.90, node-count accuracy ≥90%, assigned-node RMSE ≤2 pixels, and empty accuracy ≥95% | Worth a larger controlled experiment; report exact graph recovery too, with a provisional target ≥50%. |
| PE benefit | At least +2 percentage points in nonempty edge F1 at the same step budget, with node RMSE worsening by no more than 0.25 pixels and no reduction in empty accuracy | Suggestive only with one seed; confirm with further seeds before claiming an improvement. |

Always include performance by graph-size bin and source, and compare with an all-empty predictor and a fixed training-graph prototype baseline selected using training data only. These baselines prevent empty prevalence or a low-diversity subset from masquerading as reconstruction. Thresholds may be revised after the pilot, but retain the original targets and reason for revision in the report.

Centreline/guide agreement is secondary at this stage. Score predicted chords against target chords to measure reconstruction, and target chords against retained guides to expose the straight representation's approximation floor. Poor chord-to-guide agreement alone is not an autoencoder failure. No new annotations or favourable GT substitutions.

## Expected outputs

- A runnable geometry configuration pair and one short verification entry point; upstream COLORING still runs.
- A frozen subset manifest recording source IDs, split membership, empty/size distribution, annotation provenance, coordinate conventions and overflow checks.
- Per-run config, seed, code/data identity, environment, checkpoint, scalar losses and resource measurements.
- Saved predicted graphs containing existence/confidence, XY, connectivity and source/crop metadata; no decoder access to original nodes or edges.
- A compact report under `research/` with metric tables, reconstruction overlays, representative failures and a clear pass/fail/inconclusive assessment against the gates above.

## Unresolved hypotheses and later work

- Whether the current matcher/latent capacity learns useful geometric reconstruction at all; small-set success does not establish source-level generalization.
- Whether correctly handled Laplacian features improve on coordinates, especially for disconnected graphs and repeated eigenvalues.
- Whether auxiliary shortest-path supervision helps topology or competes with localization; retain it initially, then ablate if necessary.
- Whether loss coefficients, latent size or training budget need adjustment. Failure under one small configuration is not proof that the model family is unsuitable.
- Whether OCTA's supplied graph guides offer enough curvature to assess a later spline benefit; a bounded sample was predominantly straight. Exact spline layout, edge-only control losses, reversal conventions and joint gradients require a later plan.
- Whether findings transfer to 3D, real data, full graphs or image conditioning. None is established by this pilot.

After this plan is reviewed, implementation should address only the straight-graph experiment. No production infrastructure, broad refactoring, extensive test suite or full-scale training is part of this scope.
