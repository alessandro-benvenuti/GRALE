# GRALE research record

Consolidated 2026-10-29 from the paper review, representation discussion, dataset audits, grilling decisions and runtime reports. This is the single research reference; superseded proposals and repeated explanations have been removed. Historical upstream limitations are distinguished from fixes in implementation commit `1739776`.

Navigation: [code map](../project-graph.md) · [experiment specification](../EXPERIMENT_PLAN.md) · [implementation and run guide](../GEOMETRIC_GRAPHS.md).

## Method and representation

Reference: Paul Krzakala, Gabriel Melo, Charlotte Laclau, Florence d’Alché-Buc and Rémi Flamary, *The quest for the GRAph Level autoEncoder (GRALE)*, NeurIPS 2025; arXiv:2505.22109v3, 20 October 2025. The full paper, including appendices A–G, was reviewed. [Local PDF](<../Krzakala et al. - 2025 - The quest for the GRAph Level autoEncoder (GRALE).pdf>) · [versioned HTML](https://arxiv.org/html/2505.22109v3).

GRALE encodes a complete graph into fixed-size tokens and reconstructs node existence, attributes and connectivity. Evoformer layers update node and pair states; the default encoder pools symmetrized upper-triangle pair states with learned queries. The decoder reconstructs node slots and pair predictions. A learned matcher uses soft Sinkhorn assignments during training and hard Hungarian assignments for evaluation. The model has no learned latent prior or variational sampling objective: reconstructing encoded graphs does not establish generation from arbitrary vectors. [Model](../GRALE/main.py) · [encoder](../GRALE/model/encoder.py) · [decoder](../GRALE/model/decoder.py) · [matcher](../GRALE/model/matcher.py).

### Tensor meanings

| Object | Meaning |
| --- | --- |
| `N` / `n` | Allocated node capacity / actual number of nodes. Padding appends dummy slots; these are the same mechanism, not two node types. |
| `h[N]` | Target existence: 1 for real nodes, 0 for padding. An isolated real node still has 1. It is not a probability distribution over nodes. |
| `F[N,n_f]` | Attributes per node; feature width is independent of node capacity. |
| `C[N,N,n_c]` | Attributes for every ordered node pair, including nonedges. Storing a pair does not create a connection. |
| `Z[K,D]` | Graph latent tokens. For example, upstream `32.yaml` uses 32 slots and four 128-value tokens; 128 is not its node count. |

The code stores named node/pair fields rather than literal concatenated F/C tensors; batching adds a leading batch dimension. `DenseData.size` is allocated width, not active node count. Stored HDF5 `node_mask` has the opposite convention to `h`. Decoder existence and adjacency start as logits; default decoding converts them to probabilities but retains all slots. Thresholding is a separate step. [Containers](../GRALE/data/dense_data.py) · [original loader](../GRALE/data/dataset.py).

Node ordering is arbitrary: relabel existence and node rows together, and both pair axes. Soft tensor mixing is not necessarily a discrete permutation. The matching matrix has row/column sums one, including dummy slots; `h` is not its transport marginal.

### Paper versus upstream implementation

| Item | Paper Appendix D | Original checkout, before geometric adaptation |
| --- | --- | --- |
| Node input | One-hot labels plus first- and second-round label sums; input noise | One-hot labels plus Laplacian eigenvector features |
| Pair input | Both endpoint features, binary adjacency one-hot, sinusoidal shortest paths; optional edge labels | Both endpoint features, edge-label one-hot, sinusoidal shortest paths |
| Continuous node target | Clean diffusion-enriched features | Neighbor-averaged label proportions |
| Continuous pair target | Clean constructed pair features | Shortest-path distance divided by allocated width |
| Matcher / decoder details | Matcher MLPs; transformer updates latent before decoding | Linear matcher projections with normalized L1 cost and learned output positions; default decoder directly consumes latent |

These are verified differences, not evidence about why the code changed. The paper does not describe Laplacian node encodings; its sinusoidal encoding concerns shortest-path distances. Laplacian values describe structural position, not physical XY/XYZ, and are not guaranteed unique identifiers. [Input construction](../GRALE/model/encoder.py) · [target construction](../GRALE/model/target_builder.py).

Diffusion sums are not the same as neighbor proportions: two carbon neighbors and one oxygen contribute counts `[2,1]`, or normalized proportions `[0.667,0.333]`. A second diffusion round includes returning paths, not just distinct nodes exactly two steps away. For a path A–B–C, pair A–C has no direct edge but hop distance 2. Hop distance counts connections, not physical length. Sinusoidal encoding is a fixed sine/cosine transformation at several frequencies, not a new distance measurement. [SineEmbedding](../GRALE/model/utils.py).

Original molecular edge classes are NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC and unknown; COLORING uses NOEDGE/EDGE. Category count is independent of graph size. Continuous coordinates or controls need not become artificial classes.

### Loss and computational limits

Existence is supervised across all N slots. Node attribute costs are weighted by target existence; pair costs by existence of both target endpoints, including nonedges. Normalization uses N, active n, and n squared respectively. Predicted absence does not gate away attribute errors. One assignment factor weights node correspondence; two weight pair correspondence. This transport-weighted loss differs from first averaging a graph under a soft assignment and comparing afterward. [Objective](../GRALE/loss/objective.py) · [linear losses](../GRALE/loss/linear_losses.py) · [pair losses](../GRALE/loss/quadratic_losses.py).

Sparse edge lists suffice for storage; unlisted pairs can mean no edge. Original HDF5 stores dense edge labels and shortest paths, deriving adjacency when loading. Learned pair channels are not duplicate adjacency matrices: Evoformer updates all pairs, including nonedges, and the decoder must discover connectivity. Sparse files with batch-time densification preserve GRALE; fully sparse computation or negative sampling changes the method.

The paper's main datasets have at most 20 COLORING nodes, 16 or 32 molecular atoms. Reported reconstruction results are:

| Dataset | Edit distance | Graph-isomorphism accuracy |
| --- | ---: | ---: |
| COLORING | 0.02 | 99.20% |
| PUBCHEM 16 | 0.11 | 93.0% |
| PUBCHEM 32 | 0.78 | 66.80% |

These are published results, not reproduced locally. Pretraining populations are 300k / 14M / 84M respectively. Molecular downstream evaluation uses graphs with at most 32 heavy atoms, random 90/10 splits and standardized regression targets; it is not directly comparable to full/scaffold benchmarks. Evaluated tasks include prediction, interpolation, editing, denoising and matching. Appendix C reports batch-one memory 0.8 / 3.0 / 10.7 GB at 64 / 128 / 256 nodes for its PUBCHEM32 configuration on an RTX 3500 Ada. These scaling measurements are not our memory estimates. Cubic cost and symmetric-graph matching failures remain relevant limitations. [Paper §§5–6 and appendices A–C](https://arxiv.org/html/2505.22109v3#S5).

## Datasets and targets

Root: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets`. Findings below distinguish inspected files, manifest-wide counts and bounded samples; they are not new extraction results.

| Source | Verified availability and qualifications |
| --- | --- |
| REAVER | 36 native 2D TIF cases. Manual RGB: red segmentation, green centreline, blue raw intensity; calibration in `Manual/Results.mat`. Sample diaphragm60x1: 1024², 106,141 segmentation pixels, 4,633 centreline pixels, 0.207194 microns/pixel. Raster counts are not sparse-node counts. |
| AV Paper Online Files | DRIVE/INSPIRE/WIDE MAT graphs: XY `G.V`, connectivity `G.A`, edge weights `G.E`, artery/vein labels. WIDE includes images; DRIVE/INSPIRE here are graph-only. Sample supplied graphs have 1,028–5,233 nodes before contraction. |
| HALVS | 144/6 annotated train/validation image-mask pairs; README reports 5,057 mostly unlabeled originals. Selected planar K8 cache has 25,000/480 targets. Multicolor masks need explicit interpretation. |
| Plants | `plants_3d2cut` contains 1,511 native 2D image/mask/graph-JSON triplets despite the name. Original patch counts: 25,923/6,421/6,737 train/validation/test. Five sampled full annotations contain 104–159 nodes before native contraction. |
| OCTA | 500 2D raster/CSV source pairs; sampled image 1400², graph 1,277 nodes/1,261 edges. CSV contains varying Z and edge radius, but our agreed task uses XY only. Original uploaded patch counts 480/220/2,000 differ from the newer prepared grid below. |
| 20cities_2d | All train/validation/test raw/seg/VTP directories empty locally. |
| RETA, folder `16960855` | README and unopened RARs: 81 fundus images, 54/27 train/test; test vessel GT withheld. No inspected graph/K8 targets. |
| Biological mixture | Derived HALVS/plant caches, not independent data: 12,500 training and 240 validation examples per source. Preserve source lineage to avoid leakage. |

Sources: [REAVER description](</lustre/fsn1/projects/rech/vnc/upz25mj/datasets/REAVER_Vascular_Networks_Image_Dataset/REAVER Dataset Summary.docx>), [AV README](</lustre/fsn1/projects/rech/vnc/upz25mj/datasets/AV Paper Online Files/README.txt>), [HALVS README](/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/README.md), [HALVS cache manifest](/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015/leaves-k8/manifest.json), [plant splits](/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/plants_3d2cut/splits.csv), [OCTA splits](/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/octa-synth-packed/splits.csv), [RETA README](/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/16960855/Readme.md).

### Prepared 128×128 cohorts

These refer to the native 2D preparations, not raw patch-directory counts.

| Cohort | Train / validation | Training nodes median / p95 / max | Additional scope |
| --- | ---: | ---: | --- |
| AV-WIDE | 1,385 / 506 | 13 / 41 / 76 | 17/6 source images |
| REAVER diaphragm60x | 256 / 128 | 6 / 11.25 / 16 | 4/2 source images |
| OCTA | 10,648 / 4,840 | 18 / 32 / 52 | 88/40 source groups; validation maximum 48 |
| Plants | 25,867 / 6,398 | 6 / 14 / 28 | 1,003/251 parents; validation maximum 34 |
| HALVS legacy selected cache | 25,000 / 480 | 12 / 27 / 49 | Different extraction; validation maximum 36 |

All 23 audited AV full graphs have 359–933 contracted nodes, median 637. All 128 OCTA train/validation originals have 682–1,494 contracted nodes, median 1,099; before contraction, median 1,456. These populations support starting with patches. Patient/specimen independence beyond recorded source IDs is unproven. Preserve source splits and components, never truncate overflow or treat overlapping crops as independent subjects. [Cohort manifests](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028/) · [OCTA topology audit](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028/source-topology-audit.json) · [AV audit](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_20261028/av-wide-audit.json).

HALVS legacy arrays have three coordinates with zero Z and `source_png_normalized_yxz` ordering; a sampled target has 12 nodes, 8 edges, controls `[8,8,3]`, guides `[8,64,3]` and metric `diag(128,128,1)`. They are planar geometry, previously used as slabs. Its selection records 391 exclusions, including parallel branches, and previously pruned curvature; it is not interchangeable with topology-preserving native targets. At 32 slots, 405 selected train and 13 validation graphs overflow. [Cache documentation](/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015/README.md).

### Native target contract and OCTA lineage

| Field | Meaning |
| --- | --- |
| `nodes[n,2]`, `edges[e,2]` | Normalized XY and integer endpoint IDs |
| `controls[e,8,2]` | K8 controls, with endpoints equal to indexed nodes |
| `full_paths[p,2]`, `path_offsets[e+1]` | Original annotation polylines, separate from spline approximation |
| `metric[2,2]`, `shape_hw`, `units` | Convert normalized displacements by `delta @ metric.T`; explicit spatial units |

Sampled AV/REAVER/OCTA/plant targets share this schema; endpoint disagreement was at most 1.2e-16. No radius or per-edge knot arrays were present. Normalization divides by image size: the last 128-pixel centre is 127/128. Pixel targets use `128 I`; calibrated targets require their recorded metric.

Reuse [OCTA manifest/ZIP/audit](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028/octasynth/). The later [mixed campaign plan](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_mixed_20261028/plan.json) changes training protocol but points to the same target population. [Identity record](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_mixed_20261028/data-identities.json).

- Existing preparation maps source CSV XY as `(y,x) × (W,H)`; prepared NPZs already have canonical normalized XY. Do not swap axes again.
- Supplied connectivity is authoritative at projected crossings. Z/radius are intentionally omitted; ignoring Z is the chosen 2D task, not a claim that the source column is numerically constant.
- There are 1,503 empty train and 747 empty validation crops. Every edge-empty OCTA crop is node-empty. Test has 372 reserved source images and no prepared targets; test arrays were not opened.
- Manifest rows retain source/subject and crop origin. Native NPZs lack exact artificial-boundary flags/source-node IDs. Being geometrically on the border does not establish whether a node was clipped.
- Cropping clips paths, including reentries and outside/outside intersections, and preserves crossing identities. Closed/parallel paths gain degree-two anchors rather than being deleted. Mask extraction, used by other sources, flags ambiguous crossings/junction clusters without implicit spur removal or reconnection.
- A bounded sample of 32 nonempty training patches covered 501 edges: 423 guides had two points; 78 had more, maximum six. Curvature exists, but this is not a cohort-wide curvature estimate or dense raster tracing.

Source: [OCTA preparation](../../GNBM/gnbm-native-2d-20261027/scripts/prepare_native2d_long.py) · [extraction/cropping](../../GNBM/gnbm-native-2d-20261027/native2d/targets.py).

### Radius limitations

A centreline alone does not determine thickness. REAVER masks permit calibrated in-plane distance-transform estimates, not independently measured node radii. HALVS/plant masks need provenance checks: rendered polyline width is not anatomical radius. Sampled AV `thick` fields were zero in DRIVE/INSPIRE or 0–1 in WIDE, so not validated calibrated targets. OCTA has source edge radii around 0.00107–0.00691 in five sampled graphs, with unresolved unit semantics; these are absent from native XY targets.

## Spline and 3D evidence

Fixed open-uniform cubic K8 uses knots `[0,0,0,0,.2,.4,.6,.8,1,1,1,1]`. Share first/last controls with graph nodes; six independent interiors require 12 scalars per 2D branch or 18 per 3D branch. Existing PyTorch heads predict chord residuals, then multiply controls by a fixed spline basis. Zero residual gives a chord. Controls are not curve samples and can lie outside the crop; do not clamp them.

Endpoint reversal reverses the controls. Undirected topology does not make ordered curve channels numerically symmetric; averaging raw forward/reversed controls can collapse geometry. Dense diagonal entries can represent self-loops; excluding them is a preprocessing convention. Distinct parallel curves need additional encoding or the existing extra-anchor construction.

The corrected 2D fitter samples each unchanged guide at 64 uniform physical arc-length locations before endpoint-constrained least squares. On 1,891 crops/25,263 branches, maximum absolute control coordinate fell from 2,007,485.85 to 131.245 pixels; nodes, edges, guides, metrics and splits stayed identical. Do not copy the older fitter from the Representation checkout. [Corrected fitter](../../GNBM/gnbm-native-2d-20261027/native2d/targets.py) · [fit audit](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_20261028_targetfix/target-fit-audit.json).

A **GT-only** audit of 256 AV-WIDE development branches, seed 364505, 1-pixel tolerance and 0.25-pixel arc sampling reported Curve F1 0.99471 for K8 versus 0.53083 for chords. K8 symmetric maximum-distance p50/p90/p99/max: 0.169/0.688/1.455/2.659 pixels. This establishes approximation fidelity only, not learned reconstruction. Existing evaluators compare curves and chords on the same predicted nodes/edges; they do not compare independently trained autoencoders. [Geometry audit](../../GNBM/gnbm-native-2d-20261027/experiments/native2d_20261028_targetfix/gt-geometry-capacity.json) · [same-graph evaluation](../../GNBM/gnbm-native-2d-20261027/scripts/train_native2d_study.py).

Future learning constraints:

- Current native 2D spline code detaches endpoints and pair tokens, so its loss trains the curve head alone through that route. Do not inherit these detaches for our intended joint training. Its SmoothL1 residual loss subtracts interpolated endpoint displacement; it is not absolute curve error. [2D head](../../GNBM/gnbm-native-2d-20261027/native2d/geometry.py) · [caller](../../GNBM/gnbm-native-2d-20261027/native2d/model.py) · [loss](../../GNBM/gnbm-native-2d-20261027/native2d/losses.py).
- Baseline 3D code does not explicitly detach there and supports control/curve SmoothL1 and residual targets. Its normalized DHW is not automatically world XYZ. A sampled IXI target has 4 nodes/2 edges, controls `[2,8,3]`, guides `[2,64,3]`, no radius, exact shared endpoints, and a non-diagonal affine-derived metric carrying orientation/anisotropy. Unlike native 2D, its contractor rejects patches with closed/parallel branches or overflow: different target populations. [3D head](../../Graph-Native-Betti-Matching-Representation/models/spline.py) · [criterion](../../Graph-Native-Betti-Matching-Representation/training/losses/criterion.py) · [preparation](../../Graph-Native-Betti-Matching-Representation/scripts/prepare_transfer_splines.py) · [contractor](../../Graph-Native-Betti-Matching-Representation/data/spline_targets.py).
- Control-coordinate L2 and sampled-curve L2 are different metrics. Fixed-basis sampled L2, or equivalent control Gram-matrix error, retains the norm-plus-bilinear form needed for efficient transport loss. Chamfer, reversal minimization and predicted arc-length resampling are not automatically the same decomposition.
- Future curve loss must apply only to real target edges, not arbitrary zero curves for nonedges. GRALE's current pair loss accepts endpoint weights rather than arbitrary target adjacency weights; extending that interface remains necessary for curves. Such weighting can retain cubic matrix-product structure. None of this spline adaptation is implemented yet.

## Agreed experiment

The final design supersedes earlier REAVER-first or four-model recommendations:

1. Graph-only autoencoding with the current repository and minimal changes; no image encoder or input-graph shortcut to the decoder.
2. Existing OCTA 128×128 XY patches, supplied topology and official splits. Preserve positions/orientation; no rotation-invariance objective. Include empty graphs.
3. Straight branches first. Compare coordinate-only against coordinate-plus-Laplacian immediately; no additional curvature-driven degree-two nodes initially.
4. Keep dense GRALE, soft learned matching, binary relation supervision and shortest-path reconstruction. Coordinates are continuous targets; positional encodings remain input descriptors.
5. In implementation `1739776`, use 64 slots, latent 4×128, preserve upstream COLORING, and keep sparse files while densifying batches. Geometric decoding symmetrizes pair outputs and uses predicted existence.
6. The geometric path fixes empty attention/loss handling and computes PE on active nodes, excluding zero modes, fixing signs and zero-filling unavailable channels. Repeated eigenspaces remain ambiguous; the legacy COLORING featurizer is preserved.
7. Splines follow only if straight reconstruction is promising: K8 shared endpoints, preferred control-coordinate supervision, original-guide distance as evaluation, joint gradients through the autoencoder. Radii, 3D, full images and image conditioning remain deferred.

The [experiment plan](../EXPERIMENT_PLAN.md) owns budgets, provisional criteria and implementation locations. The [run guide](../GEOMETRIC_GRAPHS.md) owns commands and artifact layout. Avoid treating proposed settings as empirical findings.

## Validation

### Upstream COLORING

Original sampler and `demo_checkpoint.yaml`: 16 slots, five layers, 3,312,693 parameters. Tiny dataset: 8 train / 4 validation / 2 generated but unevaluated test graphs, medium sampling parameters, seed 1234; observed 6–15 real nodes. Three CPU optimizer/backward steps, finite metrics/gradients, inference, hard matching and checkpoint round trip passed. 661 parameter tensors changed; final loss 2.7918055. The original run took 18.08 seconds inside the script. The same loss was observed after geometric changes. These are runtime results, not convergence or paper reproduction.

Environment: Python 3.11.9, PyTorch 2.3.1, Lightning 2.4.0, older than README versions. Use the Jean Zay module to supply MPI/CUDA dependencies; direct invocation of its Python without module loading failed on `libmpi.so.40`. No dependencies/shared environment were changed. [Smoke script](../scripts/smoke_coloring.py); artifacts `logs/coloring-smoke/{result.json,last.ckpt}`, data `data/h5/COLORING_smoke.h5`.

An upstream lazy-loader boundary case remains: using an epoch chunk equal to the full tiny training dataset produced an empty slice on the next wrap. The successful check uses chunks of four from eight graphs. The legacy loader was not modified. No `torch.compile`, GPU/distributed or full COLORING training was performed.

### Geometric implementation

| Check | Recorded outcome |
| --- | --- |
| Coordinate-only OCTA smoke | Two finite optimizer/backward steps; inference and checkpoint reload passed |
| Coordinate + Laplacian smoke | Same checks passed on the same frozen sample manifest |
| Empty/data/PE checks | All-empty path finite, absent attribute costs zero, XY/edge identity preserved, PE unaffected by added dummy padding |
| Brief coordinate-only tiny run | 20 finite steps on 8 nonempty + 2 empty training graphs; checkpoint reload passed |
| Reconstruction quality | Tiny quality gate **not met**: predictions remained empty at threshold 0.5; nonempty F1/count accuracy zero, empty accuracy one, localization unavailable |
| Review | Standards and spec review found the same CSV aggregation issue; fixed and confirmed by both reviewers |

A concurrent PE process was externally killed with exit 137; sequential batch-two verification passed without an architecture change. CPU smoke batches changed during verification, so timings are not a controlled PE efficiency comparison. Those runtime checks did not include a full tiny fit, held-out pilot or GPU execution; the subsequent training submission is recorded below.

Scientific metrics in `metrics.json` aggregate raw counts/errors over the complete subset and separate empty/nonempty populations, sources and graph-size bins. CSV `batch/` values are step-only diagnostics; only additive error/loss terms retain epoch aggregation. Coordinate RMSE includes assigned pairs **before** the 2-pixel match filter, with coverage reported; unavailable localization is null rather than zero. The decoder never receives GT node counts/topology for scoring.

Artifacts: `logs/geometric/{smoke,tiny}/`, frozen manifests `logs/geometric/subsets/`; Git-ignored. Both smoke variants used subset SHA256 `e8e483e3c5f71f710a96c68becfe38e42b81f80b461d06dbae4c4813fa1e5727`. Saved metrics include all-empty/training-prototype baselines and chord/guide comparisons. Twenty steps establish execution, not useful reconstruction or architectural failure. [Runner](../scripts/run_geometric.py) · [metrics](../GRALE/loss/geometry.py).

### Submitted tiny fits, 2026-10-29

| Variant | Training job | Outcome |
| --- | --- | --- |
| Coordinates only | `1690370` | 1,000 steps and checkpoint saved; postprocessing failed |
| Coordinates + Laplacian | `1690371` | 1,000 steps and checkpoint saved; postprocessing failed |

Both request one H100, 1,000 optimizer steps, batch size 4, 50 warmup steps, and the same frozen 8-nonempty/2-empty training graphs with 4 validation graphs. Training is capped at 30 minutes per variant, within a 40-minute allocation. The user clarified that these two jobs may overlap the two current GPU runs; the inherited two-GPU scheduling cap does not apply to this overlap. These are diagnostic fits, far below the paper's 300k/700k/1.5M steps, not paper-budget reproductions. No held-out pilot is automatically launched.

Launcher: [jeanzay_geometric.slurm](../scripts/jeanzay_geometric.slurm). Submission record, scheduler logs, checkpoints and metrics: `logs/geometric-fit-20261029/`. Both initially had dependency holds, subsequently removed at user request.

### Recovered 1,000-step results, 2026-10-29

Both training jobs failed **after training** in the checkpoint round-trip check: Lightning validation teardown moved the model to CPU, while the runner supplied a CUDA batch. The runner now restores the requested device before comparison/export. Its new `--evaluate-only` mode preserves saved checkpoints and performs no optimizer steps.

Evaluation-only GPU jobs `1694052` and `1694053` both completed with exit 0 in 32 seconds. GPU validation, checkpoint comparison, predictions and metric export passed; checkpoint hashes match the separately recovered CPU copies. No retraining or held-out pilot was performed.

| Variant | Nonempty split | Correct node count | Node F1 (2 px) | Edge F1 | Assigned-node RMSE (px) | Exact recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Coordinates | Train, 8 graphs | 87.5% | 0 | 0 | 54.26 | 0 |
| Coordinates | Validation, 3 graphs | 33.3% | 0.0095 | 0 | 56.16 | 0 |
| Coordinates + Laplacian | Train, 8 graphs | 100% | 0 | 0 | 54.17 | 0 |
| Coordinates + Laplacian | Validation, 3 graphs | 66.7% | 0 | 0 | 56.25 | 0 |

Both correctly decoded the 2 training and 1 validation empty graphs. **Neither passed the tiny-fit quality gate.** Node-count learning has not translated into accurate placement or connectivity; these results do not establish generalization or a spline-ready representation. No automatic larger training was started.

Use GPU reports in `logs/geometric-fit-20261029/tiny/{octa_straight,octa_straight_laplacian}/{train,valid}/metrics.json` and adjacent overlays. Preliminary CPU recovery is kept separately under `cpu-recovery/`. PE metrics differ between CPU and GPU because eigenspace representations need not agree across solvers; the table uses evaluation on the training device. No cross-device prediction equivalence is claimed.

## Open hypotheses

- Whether useful geometric reconstruction can be learned after resolving ambiguous correspondence; the completed 10k fits failed, and the tracked examples still collapse at 30k. Longer training alone has not resolved those examples.
- Whether active-graph Laplacian features help coordinates, and whether repeated-eigenspace ambiguity matters.
- Whether shortest-path supervision helps connectivity or competes with localization; retain it initially before ablation.
- Whether latent size, loss weights, matching or optimization budget limit learning. A single small configuration cannot settle model-family suitability.
- Whether OCTA's supplied polylines have enough curvature to test a spline benefit; shape fidelity and learned reconstruction must remain separate.
- Whether results transfer to real data, 3D, full graphs or image conditioning. No such transfer is established.

Append future findings here with code/config/data identity and evidence paths; keep large per-run outputs in their run directories. Do not recreate a separate note for each discussion round.


### Longer tiny-set overfit, 2026-10-29

Submitted jobs `1694875` (coordinates) and `1694876` (coordinates + Laplacian), each targeting **10,000 steps** on one H100, with no dependency on existing runs. These restart from the same seed rather than continuing the 1,000-step checkpoints: cosine decay now spans 10,000 steps. All other model/loss settings, batch size 4 and warmup 50 stay unchanged. The copied frozen subset is byte-identical: 8 nonempty + 2 empty training graphs, 3 nonempty + 1 empty validation graphs; SHA256 `834a3d36a66c529522121d9612c57dcf57f4da4c5cd1e4a201010775cedbb5cb`.

Aim: determine whether longer optimization can memorize the ten graphs. Apply the existing tiny-fit gate (exact training topology/counts, assigned-node RMSE <=1 px, correct empty outputs); validation is descriptive, not a generalization claim. Each run retains the 30-minute training cap and 40-minute allocation; report the actual completed step count if capped. Previous results are preserved. Records, losses, checkpoints, predictions and overlays: `logs/geometric-overfit-20261029/`, with launch commands in `submission.json`. No new method changes or larger-dataset training are included.


### Completed 10,000-step overfits and learning diagnostics

Both variants completed all 10,000 steps and exported predictions/checkpoints (training + export: 1,377 s coordinates, 1,348 s Laplacian). Neither met the tiny-fit gate. These are fresh fits with a 10,000-step schedule, not intermediate checkpoints of the earlier 1,000-step schedule.

| Variant | Nonempty training count accuracy | Training RMSE, 1k → 10k (px) | Predicted training edges / 89 GT | Training node/edge F1 |
| --- | ---: | ---: | ---: | ---: |
| Coordinates | 7/8 | 54.26 → 52.87 | 2 / 89 | 0 / 0 |
| Coordinates + Laplacian | 7/8 | 54.17 → 55.20 | 0 / 89 | 0 / 0 |

Both still decode the two empty training graphs correctly. Validation node/edge F1 are zero; RMSE is 57.34 / 57.31 px. Node-count/existence and graph-centroid learning have not produced useful reconstruction. Median training centroid errors are 0.027 / 0.012 px; median predicted spatial spreads are only 1.81% / 0.40% of GT, where spread means RMS distance from each graph's own centroid. The coordinate-only eight-node example splits into two clusters and is an exception to single-centroid collapse, but still has no correctly localized nodes or edges under the 2-pixel matching rule.

Batch-size-weighted means over the first/last 300 training updates show total loss 0.473 → 0.154 / 0.463 → 0.161, existence loss 0.300 → 0.0115 / 0.298 → 0.0112. Most changes occur early, with long plateaus; the coordinate-only fit makes a late partial change around 8k updates. A falling total loss is therefore insufficient evidence of node/edge reconstruction. No root cause has been established and no model/loss settings were changed.

Artifacts: `logs/geometric-overfit-20261029/comparison/{learning_curves.png,gt_predictions.png,learning_diagnostics.json}`. `inspect_learning.py` reproduces the collapse on the saved 40-node training graph with an explicit spread assertion; it intentionally exits with failure on these outputs. This is an artifact-level diagnostic, not a new test suite. Next useful diagnosis is to inspect matching and per-node supervision before increasing the budget again.


### Submitted 100,000-update continuations, 2026-10-29

Jobs `1696063` (coordinates) and `1696064` (coordinates + Laplacian) restore their respective 10,000-step checkpoints and Adam states, targeting **110,000 total steps**: 100,000 additional updates each. The same byte-identical ten-graph training/four-graph validation subset and model/loss/batch settings are retained. Since the old cosine schedule ended at zero, its horizon is extended to 110k, reactivating LR at `9.79929e-5` on the first resumed update. This is a schedule extension, not a fresh model or an uninterrupted original schedule. Loader shuffling restarts from its seed; exact old sample order is not claimed.

Each job uses one H100, a 5-hour training timer and 5.5-hour allocation, with no job dependencies. Estimated new training time is about four hours per variant, running in parallel when allocated. No reconstruction improvement is claimed before results. Commands, source checkpoint hashes and subset identity: `logs/geometric-continue-20261029/submission.json`.

Every 10k steps, save a full checkpoint and three fixed GT/prediction examples (first nonempty training graph, densest training graph, first nonempty validation graph). PNGs, predicted NPZ graphs and per-example metrics go under `tiny/<config>/intermediate/step-XXXXXX/`; a baseline is saved at resumed step 010000. Full train/validation evaluation remains at completion. Plots label node/edge counts, identify GT guide overlays and use common axes that include all predicted nodes.

Targeted verification: GPU smoke job `1696031` completed in 36 seconds with exit 0; restored global/Adam counters advanced from 10000 to 10002, gradients/weights changed, LR was nonzero, three PNGs per snapshot were exported, and checkpoint reload matched. An earlier assertion incorrectly required a conditionally used parameter's Adam counter to advance on both updates; the corrected check uses the maximum counter. A login-node CPU attempt was externally killed. No model/loss changes or broad test suite were introduced. Verification artifacts: `logs/geometric-continue-20261029/resume-smoke-gpu/`.

### 30k collapse and matching diagnosis, 2026-10-29

Current user scope is **graph input -> reconstruction of that same graph, without
images**. Straight edges are the intended target; curvature and later image
conditioning are outside this diagnosis.

Both continuation runs reached saved 30k checkpoints. On the tracked 40-node,
31-edge training graph `octa_20251028_242246_x1024_y1152`, both decode **40 nodes
and zero edges**. Predicted spatial spreads are 0.381 px (coordinates) and
0.424 px (Laplacian), versus 60.756 px for the target. Centroid errors are
0.751 / 0.835 px. Spatial spread means RMS distance from each graph's own
centroid. Thus the count can be right while the graph is completely wrong.
The yellow paths on prediction overlays are GT guides, not reconstructed edges.
These are selected-example diagnostics, not a new full-subset 30k evaluation.

CPU inference checks on saved 10k and 30k checkpoints identify ambiguous learned
correspondence as a concrete failure mechanism:

- On the inspected two-node training graph, both variants assign each target
  approximately 50% to each of two output slots. Encoder node embeddings are
  distinct, so the input nodes are not completely indistinguishable internally.
- On the 40-node coordinate-only example at 30k, the mean largest assignment
  weight per target is 0.02489, close to the uniform reference 1/40. Its predicted
  coordinates lie about 0.84 px from their transport-weighted target averages.
- In an isolated coordinate-loss gradient check on that 40-node example,
  substituting a geometric one-to-one correspondence produces a gradient norm
  about 72 times larger. This holds the predictions fixed and does not retrain
  anything; it demonstrates weak coordinate supervision under the current plan,
  not that changing matching alone has already fixed learning.

Plain-language mechanism: when each predicted node is told to fit nearly every
target node, coordinate L2 encourages their average position, the centroid.
Diffuse pair supervision can similarly favor average edge probabilities below
the decoding threshold. The marginal penalty only checks row/column sums; a
uniform assignment can satisfy those sums without establishing correspondence.

The inherited matcher uses L1 distance between **learned projected embeddings**,
not between graph XY coordinates. On the two-node example, all 32 projected
dimensions have the same sign of output-minus-input across the active 2x2
block. L1 then separates into a row term plus a column term: both one-to-one
assignments have the same cost. At 30k, double-centered active cost residuals
are at most 2.4e-7 for both variants. Sinkhorn therefore has essentially no
preference between those correspondences; lowering temperature alone cannot
break an exact tie. This failure can occur with categorical or continuous input
attributes. Replacing categorical labels with coordinates may affect learning,
but continuous features have **not** been established as the main cause.
Collapsed outputs and ambiguous matching can reinforce each other.

Other hypotheses remain open. Cost-sum normalization includes all 64 slots, so
reusing upstream epsilon does not ensure comparable sharpness after increasing
capacity. Decoder/optimization effects and relative loss strengths have not
been excluded by a controlled training intervention. These findings do not
establish that GRALE or transport matching cannot reconstruct geometric graphs.

Evidence: [10k matcher audit](review-20261029/matcher_audit.json),
[audit script](review-20261029/matcher_audit.py),
[30k inference/gradient check](review-20261029/collapse_30k.json), and
[30k check script](review-20261029/collapse_30k.py). For the latter, use the same
Jean Zay module environment and single-thread settings as the 10k audit.
Checkpoint and snapshot paths are under
`logs/geometric-continue-20261029/tiny/<config>/{checkpoints,intermediate}/`.

### Continuations stopped; synthetic matching comparison submitted

At the user's request, jobs `1696063` and `1696064` were cancelled after scheduler
elapsed times 57m19s and 53m15s respectively. Existing checkpoints and predictions
are retained. They did not complete the planned 110k total updates.

The next experiment isolates matching on **one synthetic bent path**, with XY
nodes `(0.18,0.22)`, `(0.48,0.76)`, `(0.84,0.32)` and edges `0-1`, `1-2`.
No images are used. Node order remains fixed. Three arms share identical initial
parameter values, the existing coordinate-only architecture including 64 slots,
all reconstruction coefficients, seed 1234, batch size one, learning rate 1e-4,
50 warmup steps, and a 2,000-update cosine schedule:

| Arm | Training correspondence |
| --- | --- |
| `l1` | Existing learned Sinkhorn matcher and L1 embedding cost |
| `squared_l2` | Same learned matcher with squared-L2 cost; cost-sum normalization, epsilon 1e-4 and 50 Sinkhorn iterations unchanged |
| `fixed` | Identity assignment supervises the first three slots and all padding; matcher bypassed in the loss |

Fixed correspondence supplies supervision only: the decoder still receives only
the latent, and inference uses predicted existence/adjacency at 0.5. Success
requires exact recovery under the geometric scorer and assigned-node RMSE <=1 px
at scale 128. Save predictions, loss curves, spatial spread and transport
concentration. This measures single-graph memorization, not generalization,
permutation robustness, or whether the latent depends on the input.

An inherited optimizer discrepancy was verified: the YAML declares
`weight_decay: 0`, but `configure_optimizers` omits that argument and AdamW uses
0.01. The diagnostic preserves existing behavior across all arms and records
the effective value; no optimizer correction is mixed into the comparison.
The new `matcher_cost` option defaults to `l1`, preserving existing configurations.

The two-update CPU smoke completed all arms with finite gradients and identical
initial parameter hashes. Artifact: `/tmp/grale-synthetic-smoke-20261029/result.json`.
GPU job **1698475** runs the three arms sequentially on one H100, within a
20-minute allocation. **Completed successfully**, scheduler exit code `0:0`,
elapsed time 6m32s. All three arms completed 2,000 updates and passed the
single-graph reconstruction gate; measured results follow below.

Entry point: [diagnose_synthetic_matching.py](../scripts/diagnose_synthetic_matching.py).
Submission: `sbatch --parsable scripts/jeanzay_synthetic_matching.slurm`.
Expected outputs: `logs/synthetic-matching-20261029/`; scheduler logs:
`logs/synthetic-matching-1698475.{out,err}`.

Interpretation: fixed-only success implicates learned correspondence under these
settings. Squared-L2 success alongside L1 failure supports changing the cost.
Both learned arms failing does not establish that transport cannot work, since
temperature and padding remain unchanged. All arms failing would also require
examining decoder/optimization; all succeeding would require a more representative
reproducer of the vessel failure. These were the prespecified interpretations;
the observed result is that all three arms succeeded.

### Completed single-graph comparison

| Arm | Correct nodes / edges | Assigned-node RMSE at 2k (px) | First saved snapshot passing the gate |
| --- | --- | ---: | ---: |
| Original L1 | 3 / 2, exact recovery | 0.0421 | 500 updates |
| Squared-L2 | 3 / 2, exact recovery | 0.0399 | 500 updates |
| Fixed correspondence | 3 / 2, exact recovery | 0.0292 | 200 updates |

All initial parameter hashes match. Final predicted spatial spreads are
45.728 / 45.731 / 45.736 px versus target 45.763 px. The learned matchers end
with mean maximum target-assignment weights 0.999989 (L1) and 0.999999
(squared-L2), essentially one-to-one. At the 200-update snapshot, L1 still has
1.14 px spatial spread and 45.16 px localization RMSE, but escapes that collapse
by the 500-update snapshot. Squared-L2 also passes by 500 updates. Snapshot
spacing does not establish the exact update when either method first succeeds.

This revises the diagnosis: continuous XY, 64 padded slots, and the inherited
L1 matcher do **not inherently prevent** learning this graph. Fixed assignment
learns earlier in this run, while squared-L2 and L1 achieve similarly accurate
final reconstructions. One graph/seed does not establish a general advantage
for squared-L2. The previously measured L1 ties and diffuse plans on vessel
checkpoints remain real observations, but are not by themselves proof that L1
is the sole root cause of those failures. A more representative reproducer is
needed; fitting one previously failing small vessel patch alone would separate
single-example difficulty from training on multiple graphs. No further runs
were launched as part of checking these results.

Results: `logs/synthetic-matching-20261029/result.json`; per-arm
`snapshots.json`, `losses.json`, `prediction-002000.{png,npz}`, transport arrays
for learned arms, and `final.ckpt`. The L1 final prediction was visually checked
against the target and agrees with the geometric scorer. Comparison figure:
`logs/synthetic-matching-20261029/learning-curves.png`.

### Single existing OCTA patch: matching comparison submitted

The user requested the same comparison on a small dataset graph. Selected
training sample **`octa_20251028_260381_x640_y384`**, with **two nodes and one
edge**, from the frozen manifest
`logs/geometric-continue-20261029/subsets/tiny-1234.json`. This is an existing
OCTA-Synth graph, also inspected in the 10k/30k collapse diagnosis. Its original
normalized XY, connectivity and node order are used without augmentation or
refitting. No image enters the model. Saved `target.npz` is checked against the
original archived NPZ, preserving both nodes and the edge.

Repeat `l1`, `squared_l2`, and `fixed` on this one graph, with the same 64-slot
coordinate-only model, seed 1234, identical initial weights, batch size one,
2,000 updates, 50-step warmup, base LR 1e-4, and effective AdamW weight decay
0.01 as the completed synthetic comparison. Losses, transport normalization,
epsilon and decoding thresholds remain unchanged. This isolates fitting the
previously failing example alone from fitting it alongside other graphs; it
does not establish performance across the dataset.

The shared diagnostic runner now accepts `--manifest` and `--sample-id` and
uses the actual active-node count for transport diagnostics. Default synthetic
behavior is retained. The prepared-data path verifies the expected 128x128 pixel
metric and records the selected manifest/sample identity in the output.
Two-update CPU checks passed for all arms with finite gradients, equal initial
parameter hashes, and an unchanged target round trip. Smoke artifacts:
`/tmp/grale-real-single-smoke-20261029/`.

GPU job **1699402** completed all three arms successfully, exit `0:0`, elapsed
6m49s. All three passed the two-node reconstruction gate; results below. Artifacts:
`logs/real-single-matching-20261029/`; scheduler logs:
`logs/real-single-matching-1699402.{out,err}`.

Reproduction command from the repository root:

```bash
sbatch --parsable --job-name=grale-real-single \
  --output=logs/real-single-matching-%j.out \
  --error=logs/real-single-matching-%j.err \
  scripts/jeanzay_synthetic_matching.slurm \
  --manifest logs/geometric-continue-20261029/subsets/tiny-1234.json \
  --sample-id octa_20251028_260381_x640_y384 \
  --output logs/real-single-matching-20261029
```

### Eight-node existing OCTA patch: additional comparison submitted

At the user's request, submit the same three-arm single-graph diagnostic on
training sample **`octa_20251028_297896_x896_y768`**: **8 nodes, 5 edges,
3 connected components**, using the same frozen tiny manifest. This patch also
failed in the original ten-graph fit. Keep its original coordinates, node order,
disconnected topology and 128-pixel metric. No images enter training.

All settings match the two-node comparison: coordinate-only 64-slot model,
L1 / squared-L2 / fixed correspondence, identical initial weights, seed 1234,
batch size one, 2,000 updates, 50 warmup steps, unchanged loss coefficients,
temperature and optimizer behavior. This is one additional dataset patch,
not a sweep or a new data split. A two-update CPU smoke completed all three
arms with finite gradients and equal initialization hashes; artifacts:
`/tmp/grale-real-eight-smoke-20261029/`.

GPU job **1699478** completed all three arms, exit `0:0`, elapsed 6m11s.
Only fixed correspondence passed the eight-node reconstruction gate; results
below. Outputs: `logs/real-eight-matching-20261029/`; scheduler logs:
`logs/real-eight-matching-1699478.{out,err}`.

```bash
sbatch --parsable --job-name=grale-real-eight \
  --output=logs/real-eight-matching-%j.out \
  --error=logs/real-eight-matching-%j.err \
  scripts/jeanzay_synthetic_matching.slurm \
  --manifest logs/geometric-continue-20261029/subsets/tiny-1234.json \
  --sample-id octa_20251028_297896_x896_y768 \
  --output logs/real-eight-matching-20261029
```

### Completed two-node and eight-node dataset-patch comparisons

Both jobs completed all 2,000 updates per arm without training errors. All six
initial parameter hashes match the earlier synthetic run. Model, optimizer,
loss weights and decoding settings are shared; each arm fits one unchanged
dataset graph. This is one seed per graph and method, not a population estimate.

| Patch | Matching | Localized nodes (2 px) | Predicted / GT edges | Correct edges | Assigned-node RMSE (px) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2 nodes, 1 edge | L1 | 2/2 | 1/1 | 1 | 0.0090 | Pass |
| 2 nodes, 1 edge | Squared-L2 | 2/2 | 1/1 | 1 | 0.0290 | Pass |
| 2 nodes, 1 edge | Fixed | 2/2 | 1/1 | 1 | 0.0109 | Pass |
| 8 nodes, 5 edges | L1 | 2/8 | 3/5 | 1 | 16.7469 | Fail |
| 8 nodes, 5 edges | Squared-L2 | 0/8 | 0/5 | 0 | 27.5368 | Fail |
| 8 nodes, 5 edges | Fixed | 8/8 | 5/5 | 5 | 0.0364 | Pass |

All methods predict the correct node count on both patches at 2k. Edge
correctness includes the node-localization association; a predicted edge between
mislocalized nodes does not count as recovered. RMSE includes all assigned
nodes, including those outside the 2-pixel matching tolerance.

On the two-node patch, fixed first passes at the saved 500-update snapshot;
both learned matchers first pass at the saved 1,000-update snapshot. Both
learned methods initially cluster the nodes and later escape, ending with
essentially one-to-one assignments. This previously failing dataset example is
therefore learnable alone using the original matcher.

On the eight-node patch, fixed first passes at the saved 500-update snapshot
and retains accurate positions/topology through 2k. L1 partially separates
nodes but leaves four outputs sharing almost uniform 0.25 assignments to the
four upper target nodes; another two outputs mix two lower leaves at roughly
0.5 each. Squared-L2 divides the graph into two groups of four nodes, with
roughly 0.25 assignment weights within each group. Neither learned arm passes
at any saved snapshot. Mean maximum target assignment weights at 2k are
0.498 (L1) and 0.255 (squared-L2), compared with values essentially 1 on the
successfully reconstructed two-node patch.

The saved transport arrays explain the positions directly: every active L1
prediction is within 0.12 px of the target average induced by its transport
row; every squared-L2 prediction is within 0.21 px. These are **group-average
collapses**, not necessarily a single global-centroid collapse. Their overall
spatial spreads can be fairly large (60.51 / 56.50 px versus target 62.98 px)
while individual nodes and connectivity remain wrong. Spatial spread alone is
therefore insufficient to certify reconstruction.

Controlled conclusion: this model and coordinate supervision can memorize the
eight-node graph when correspondence is supplied. Learning correspondence is
a demonstrated obstacle on this example under the tested budget/settings.
Squared-L2 alone does not resolve it and performs worse here; the problem is
not specific to L1's additive-cost degeneracy. These results do not establish
that learned matching can never succeed with other settings, more steps, or
other seeds. They also do not yet isolate which part of learned matching
(feature quality, sharpness, optimization, etc.) prevents one-to-one assignments.
No further training was launched while inspecting these results.

Evidence: `logs/{real-single,real-eight}-matching-20261029/result.json`,
per-arm `snapshots.json`, final `prediction-002000.{png,npz}`, and learned-arm
`transport-002000.npy`. All three final eight-node prediction plots were
visually checked against the target; they agree with the scorer and transport
analysis. Learning curves are saved in each run's `learning-curves.png`.

### Ten-patch learned-L1 run prepared for Jean Zay

The requested next run uses **200,000 optimizer steps**, not epochs, and batch
size 4. It retains learned L1 Sinkhorn correspondence, the coordinate-only
64-slot architecture, seed 1234, all existing loss weights and the inherited
optimizer behavior. It starts fresh with 50 warmup steps and cosine decay over
200k updates. No fixed-correspondence supervision is used.

`octa_straight_l1_200k.yaml` and
`cluster/jean_zay/{submit_geometric_tiny.sh,train_geometric_tiny.slurm}` configure
the existing tiny selection: eight nonempty/two empty OCTA training patches,
four validation patches, and no test targets. Original source splits and target
geometry remain unchanged. Batches are 4, 4, 2 per epoch.

The existing setup script's H100 profile creates a module-backed virtual
environment under `$SCRATCH/venvs/grale-h100-torch231`; the submission and job
both activate it. Login preflight freezes the subset and validates every
selected target before GPU submission. Checkpoints and three fixed
reconstruction snapshots are saved every 10k steps. The training timer is 19h
inside a 19h30 one-H100 allocation; stopping early is reported explicitly.
Commands and output locations are in [the run guide](../GEOMETRIC_GRAPHS.md#run).
This records prepared configuration, not submission or reconstruction results.
Local verification used a synthetic 14-target fixture: preflight selected 10/4
graphs with the intended batch size and L1 cost, a four-graph backward pass had
finite gradients, and a two-update CPU run saved snapshots/checkpoints and
passed checkpoint reload. The loader produces batches 4/4/2; the configured
schedule reaches LR 1e-4 at step 50 and zero at step 200,000. Shell syntax and
diff whitespace checks passed. Actual Jean Zay environment/data checks run in
the submission wrapper before allocation.

### Patch-size restriction and recovered scripts

The user supplied `scripts.zip` and confirmed the source dataset root as
`/lustre/fsn1/projects/rech/vnc/upz25mj/datasets`. The recovered runner matches
the earlier implementation before the new input-check option. Its additional
`jeanzay_geometric_200k.slurm` resumes the existing 30k checkpoints and runs both
coordinate-only and Laplacian variants as an array; it has no node-count filter
and does not activate the SCRATCH venv. It was inspected as source material,
not executed or copied over the current launcher.

The requested fresh learned-L1 run now samples only patches with **at most 20
nodes**, independently within the original train/validation splits. Filtering
precedes the seeded 10/4 selection; complete graphs are retained and no node
truncation occurs. Empty counts remain two training and one validation graph.
The 64-slot architecture and all other training settings stay the same.
The frozen subset records the cutoff and source/eligible counts; a different
cutoff cannot silently reuse it. Outputs now use
`$WORK/experiments/grale-octa-tiny-l1-max20-200k/`.

The supplied dataset root contains source data and does not replace the
prepared OCTA manifest. The launcher checks the recorded preparation paths,
including the original GNBM checkout, and verifies the selected NPZ targets
before allocation. A relocated manifest can still be passed explicitly.
Local checks used a fixture containing 20-, 21- and 40-node graphs in both
splits: exactly 20 remained eligible, larger graphs were excluded, selected
target rows were unchanged, subset reuse was deterministic, and a changed
cutoff was rejected. Insufficient eligible populations fail before training;
a reserved test entry pointing to a nonexistent archive was not opened. The
two-update CPU run also passed finite backward, snapshots and checkpoint reload
with the filtered cohort. These checks do not audit remote dataset contents.

### Completed 200k L1 fit and inspection preparation, 2026-10-30

User-supplied logs confirm job **1701891** reached 200,000 finite backward
updates on an H100, saved snapshots through step 200k, and passed checkpoint
reload. The run is
`/lustre/fswork/projects/rech/vnc/upz73jr/experiments/grale-octa-tiny-l1-max20-200k/20261029-171558/`.
The supplied environment record confirms PyTorch 2.3.1 from its module and the
SCRATCH venv `grale-h100-torch231`. Frozen subset SHA256:
`c199e3a4d18229b067a3b779c6f17df5941624a5a94c3d77a86cbedb6484ddec`.

The reported tiny-fit gate is false. Nonempty training node/edge F1 are
0.017857 / 0.024390, coordinate RMSE 56.3631 px, node-count accuracy 1.0 and
exact reconstruction 0. Validation F1 are both zero, RMSE 56.8977 px,
node-count accuracy 1/3 and exact reconstruction 0. Component-count accuracy
is zero in both splits; cycle-count accuracy and empty exactness are one.
These aggregates establish failed training-set reconstruction, but do not
alone establish centroid or group-average collapse on this checkpoint.

`scripts/inspect_geometric_run.py` and the CPU submission wrapper implement
the requested four checks over all 14 frozen graphs, loss CSVs and the saved
geometry snapshots. The diagnostic verifies recorded hashes, reproduces saved
predictions, measures real-node transport concentration and target-average
residuals, checks the exact squared-loss decomposition, and packages plots,
arrays and a compact summary. It does not train or modify original artifacts.
The remote checkpoint is not mounted locally; actual diagnostic conclusions
require running it on Jean Zay and retrieving its output.
Local validation reproduced every saved prediction and aggregate reconstruction
metric from the two-update fixture. Constructed sharp, uniform and group-mixed
plans verified the assignment diagnostics and exact variance decomposition.
A synthetic CSV verified first/last windows, late-plateau measurements and loss
coefficients. Generated graph/loss plots were visually checked. These are tests
of the inspector, not new findings about the 200k checkpoint.
