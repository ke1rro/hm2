# First checkpoint implementation

Spec: `codex_hw2_first_checkpoint_prompt.md` and `docs/assignment_context.md`.

Use the existing checkout. Do not stage, commit, branch, train, or download large data
during development. Keep tests focused on coordinate and annotation contracts.

- [x] Add minimal parsing, archive discovery, inventory, and coordinate helpers in
  `src/hm2/data.py`. Test normalized/pixel conversion, crowd separation, matching frame
  identities, optional ignore files, and preserving existing archive destinations.
- [x] Add Pillow frame overlays and crop views in `src/hm2/visualization.py`.
- [x] Create `notebooks/00_data.ipynb` with visible Kaggle download logic, safe local
  extraction, actual inventory, integrity checks, and compact Parquet tables.
- [x] Create `notebooks/01_eda.ipynb` with explicit Polars analyses, Plotly figures,
  deterministic galleries, coordinate comparison, and sequence-level validation.
- [x] Validate notebook schema and syntax, test critical helpers, run project quality
  checks, and inspect example overlays. Execute on real data only if available.

Initial inspection found no dataset or previous-homework notebooks. The user supplied
KaggleHub download instructions and a token. Download and extraction subsequently
completed through KaggleHub. The competition page was inaccessible through browsing.

## Verified results

- Executed both notebooks in fresh CPU kernels from `notebooks/`. All 15 setup and
  38 EDA code cells have saved successful outputs. Schema and syntax validation passed.
- Source files match the published 30,858 frames and 111 clips across three splits.
  Train and validation share no clip IDs.
- Counted 1,219,648 target annotations and 67,838 unique ignored regions. Each ignored
  region has eight COCO category replicas. All original annotation/category IDs remain
  available in list columns. Raw source files were not modified.
- All 48 sampled YOLO/COCO frame comparisons match. Maximum coordinate difference is
  0.00256 px, below the 0.01 px rounding tolerance. Original-dimension round trips pass.
- Sampled JPEG headers, including resolution extremes, agree with metadata. Actual
  images reach 3840 × 2160 px, exceeding the supplied context's stated maximum.
- Measured small-object fraction: 24.4%. Median targets/frame: 42. Maximum: 180.
  Cars account for 44.0% of target annotations, counting repeated appearances.
- Visually reviewed sparse, crowded, tiny-object, ignored-region, occlusion-code, and
  rare-class frames, magnified crops, a test frame, four consecutive frames, and a YOLO
  overlay. Boxes align with visible objects. Ignored regions have a distinct magenta style.
- Tables and figures use direct Jupyter expressions. Explicit display remains for
  conditional or loop output and compressed JPEG galleries. EDA notebook is 951 KiB.

## Additional measured findings

- The full extracted dataset contains 10.975 GB of file contents (10.222 GiB).
  Train contributes 8.417 GB, validation 1.667 GB, and test 0.890 GB. KaggleHub removed
  the original archive after extraction. Prepared metadata occupies about 0.012 GB.
- Target GT boxes can overlap ignored regions in the supplied annotations. In frame
  `train/uav0000288_00001_v/0000918`, there are 61 targets and three unique ignored
  regions. The large right-side region at pixel xywh `(1809, 0, 93, 289)` intersects
  12 target boxes and fully contains 10. Its eight source COCO annotation IDs are
  1036480 through 1036487. This is an observed source-data property, not a prediction
  or a rendering artifact. It has not been measured across the entire dataset.
- The precise training and scoring treatment of GT/ignore overlaps remains unresolved.
  Do not infer that every overlapping GT must be dropped, or assume an overlap threshold,
  without checking the chosen model and the competition evaluator's actual conventions.

## Checks and remaining limitations

- Passed `uv run isort --check-only .`, `uv run black --check .`,
  `uv run flake8 src/ tests/`, `uv run ty check`, `uv run pytest` (5 tests),
  `uv lock --check`, `uv pip check`, and `git diff --check`.
- All hooks pass for the changed deliverable paths. Full repository pre-commit
  still fails only on the pre-existing `report.md` Markdown formatting. It is unchanged.
- `sample_submission.csv` is missing from the downloaded files and unavailable through
  the individual-file endpoint. The notebooks report it, skip unavailable sample-row
  validation, and do not generate a replacement or submission predictions.
- The local environment repeatedly marks its editable `.pth` file hidden, which Python
  skips. Notebooks explicitly resolve this checkout's `src/` before importing helpers,
  and pytest uses `pythonpath = ["src"]`. Both entry points pass repeatedly.
- Read-only independent review found one explicit-ZIP precedence issue, now fixed and
  re-reviewed. Repeat runs reuse the selected extraction instead of the prior dataset.
- No model, training, detector evaluation, submission generation, or report writing.
  No staging, commits, pushes, branches, or website-file changes.

## Updated storage policy

Per the user's correction, `.artifacts/` holds AI notes and agent working scripts only.
Notebook tables and manifests are in `data/metadata/`, and visual QA previews are in
`data/previews/`. Project code and notebooks do not reference `.artifacts/`. These data
directories are ignored by Git. Source downloads remain in KaggleHub's managed cache.
