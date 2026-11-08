# Controlled data experiments

The user specification is the design. Implement sequentially without GPU execution,
cloud writes, downloads, git staging or commits. Preserve existing baseline identities.

1. Add focused tests for photometric geometry/replay, resolution round trips,
   RetinaNet positive preservation/negative classification masks, and fixed-budget RFS.
2. Add four focused modules under experiments/. Keep scientific settings in four
   config/data_*.toml recipes. Disabled features must resolve to the baseline recipe.
3. Add optional training dataset/order hooks to Faster R-CNN and model/dataset hooks
   to RetinaNet. Existing callers retain exact defaults and checkpoint recipes.
   New ablations retain best checkpoints separately without changing baseline selection.
4. Reuse existing AMP, optimizers, evaluators, atomic snapshots and Drive API callbacks.
   Add a short explicit probe and GPU memory records; never change batch size on OOM.
5. Add four numbered research notebooks with training/probes disabled by default,
   real diagnostics, result readers, metric/class/loss plots and explicit missing states.
6. Provide a single execution-only Kaggle entry using the established code bootstrap,
   competition staging, credential probe and Drive persistence; build/verify its code ZIP.
7. Verify focused CPU behavior, resume, actual setup cells with simulated Drive,
   all notebook schema/syntax, missing-result execution, format/lint/types and full tests.

Review focus: ignore matcher uses -1 background / -2 excluded; no-positive batches;
RFS threshold .01 can be a no-op; selected order/cursor deterministic on restart;
baseline five-epoch metrics are distinct from post-hoc epoch-one candidates;
no augmentation in validation; no new factor enters a disabled recipe;
GPU probe does not overwrite training checkpoints or silently reduce batch size.

## Verified completion

Implemented all four modules, TOML recipes and research notebooks 05 through 08.
The execution entry is experiments/03_data_experiments_kaggle.ipynb, using
controlled_data_project.zip and the original Drive API pipeline. Training and
CUDA probing are disabled by default. The bundle is about 9.9 MB with original
metadata bytes, all source modules and configs, verified against the working tree.

Full suite: 74 passed. All new/changed-file pre-commit hooks, Black, isort, Flake8,
and ty passed. CPU Run All succeeded in all four research notebooks; outputs remain
below 1 MB. Hosted bootstrap, data staging/real image pairs, and Drive save/restore
were tested with simulated HTTP and real local atomic checkpoints. Real cloud
execution and CUDA memory/throughput remain untested. No GPU training, external
writes, downloads, staging, commits or pushes were performed for this task.

Exact original Faster R-CNN and RetinaNet dataset/selection hashes both matched.
RFS bus image presence is 1333/4868, so threshold .01 is a no-op. The explicitly
configured .5 threshold uses fixed-length weighted sampling, capped at four, with
4868 image draws per epoch and 12170 nominal batch attempts over five epochs.
Actual successful AMP updates remain separately recorded, matching the original
budget convention. Disabled features resolve to and dispatch through baseline paths.

One CPU geometry-only RetinaNet diagnostic retained 219 positive anchors and
suppressed 3539/109825 negative anchors (3.2224%) on one training frame. This is not
a training metric or a dataset-wide estimate. Records are under the new gitignored
ignore_aware_retinanet output directory and shown by its research notebook.

Repository-wide pre-commit fails only on pre-existing report.md mdformat formatting;
report.md has no working-tree diff and was not changed. Existing unrelated user
changes and prior-task modifications were preserved. No delegated work was retained;
the single initial delegate was interrupted at the user's request before writing files.
