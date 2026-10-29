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

- Whether useful geometric reconstruction emerges beyond these first 1,000-step fits; node counts improved, but placement/connectivity remain poor and the tiny-fit gate failed.
- Whether active-graph Laplacian features help coordinates, and whether repeated-eigenspace ambiguity matters.
- Whether shortest-path supervision helps connectivity or competes with localization; retain it initially before ablation.
- Whether latent size, loss weights, matching or optimization budget limit learning. A single small configuration cannot settle model-family suitability.
- Whether OCTA's supplied polylines have enough curvature to test a spline benefit; shape fidelity and learned reconstruction must remain separate.
- Whether results transfer to real data, 3D, full graphs or image conditioning. No such transfer is established.

Append future findings here with code/config/data identity and evidence paths; keep large per-run outputs in their run directories. Do not recreate a separate note for each discussion round.
