# Geometric GRALE implementation check

2026-10-29. Implemented the straight-graph stage of [EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md); [run instructions](../GEOMETRIC_GRAPHS.md). No spline implementation, GPU job or full-scale training was launched.

## Execution evidence

| Check | Result |
| --- | --- |
| Python syntax and imports | Passed |
| Original COLORING smoke | Passed, unchanged final loss 2.7918055 and checkpoint round trip |
| OCTA coordinates-only smoke | Passed, two optimizer steps, finite gradients and checkpoint reload |
| OCTA coordinates + Laplacian smoke | Passed, two optimizer steps, finite gradients and checkpoint reload |
| Empty graphs | All-empty forward/backward finite; absent node/pair attribute losses zero |
| Data and PE | XY/edge identity preserved; PE unchanged by additional dummy padding |
| Brief coordinate-only tiny run | Completed 20 finite backward/optimizer steps on 8 nonempty + 2 empty training graphs |
| Scientific metric logging | Reviewed and corrected; final validation check confirms epoch logging contains additive error/loss terms, not averages of batch F1/RMSE |

Environment: CPU, Python 3.11.9, PyTorch 2.3.1 and Lightning 2.4.0. Prepared OCTA source and validation memberships were preserved; no test targets were opened. Both smoke variants used the same frozen manifest hash `e8e483e3c5f71f710a96c68becfe38e42b81f80b461d06dbae4c4813fa1e5727`. CPU smoke batch sizes were adjusted during runtime verification; these timings are not a controlled PE speed comparison.

The first concurrent PE check was externally killed (exit 137); a sequential batch-two check passed. No architecture change was made to address that resource interruption.

## Scientific interpretation

The implementation is runnable, but **useful reconstruction has not been established**. The 20-step coordinate run still decoded empty graphs at the fixed 0.5 existence threshold: nonempty node/edge F1 and count accuracy were zero; empty accuracy was one. Localization was unavailable because no nodes were selected. The tiny-fit quality gate was false. This is a deliberately short runtime check, not the planned 1,000-step fit or a verdict on the model family. No held-out pilot was started.

Generated configs, checkpoints, raw counts, exact dataset-level metrics, predictions and overlays are under `logs/geometric/{smoke,tiny}/`; frozen manifests are in `logs/geometric/subsets/`. These artifacts are Git-ignored. Use a fresh output directory for subsequent bounded fits; do not overwrite this evidence or infer success from the fraction of empty graphs.

## Review

Standards and spec reviews independently found the same batch-ratio aggregation issue. It was fixed: CSV `batch/` values are step-only diagnostics, while the evaluator aggregates raw counts/errors over each complete subset. Both reviewers confirmed resolution and reported no remaining findings. Verification stayed limited to the intended paths; no test framework, CI or exhaustive suite was added.
