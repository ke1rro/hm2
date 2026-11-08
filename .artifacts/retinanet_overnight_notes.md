# Overnight RetinaNet orchestration, 2026-11-04

## Scope and implementation

Batch prepared: `runs/retinanet_overnight/20261104_001829/`.
Four independent COCO initializations in order R0-S, R1-S, R0-F, R1-F. Five epochs,
batch 2, seed 2026, 640/1024 resize, shared SGD/focal/GIoU recipe. No DIoU in batch.
Subset contains 4868 frames across all 56 clips, full train 24201, full val 2846.
Planned per-run exposures: subset 24340, full 121005. Expected update counts before
AMP skips: subset 12170, full 60505. Record actual updates and exposures separately.
Full validation after every epoch determines best checkpoint by canonical mAP.

`config/retinanet_overnight.toml` stores the recipe. The epoch runner reuses existing
training, model and COCO utilities. Checkpoints include model/optimizer/scheduler/scaler,
epoch, batch cursor, history, seed/configuration and RNG states. Deterministic epoch
permutations and per-image flips permit cursor restoration, without a bit-for-bit CUDA
claim. Saves occur approximately every 600 seconds, each epoch and completion.
Interrupted pending validation repeats evaluation, not completed training.

The controller uses only official Colab CLI commands. Current persistence strategy
is periodic local downloads, superseding the initial Drive plan. No MLflow or UI.
Remote immutable snapshots and SHA256 manifests are published atomically, retained
at least 30 minutes and three generations. Local complete verified generations are
committed with an atomic CURRENT pointer before updating convenience mirrors.
Failed transfers preserve the last verified local generation. Local downloads use
`weights_only=True` checkpoint loading as an additional check. Resume uploads verified
state to a new runtime when needed; completed experiments are skipped.

Large upload support uses 32 MiB parts, streaming remote assembly, whole-file SHA256
verification and atomic replacement. Dataset staging reuses existing source JPEGs,
checks remote file sizes and verifies archive SHA256/CRC. Required image transfer is
9.063 GiB. Free Mac space initially approximately 191 GiB; remote free space 112 GiB.

Controller supports detach under caffeinate, PID locking, stop requests, two bounded
missing-runtime recoveries, live download-only polling and safe runtime shutdown after
persistence. Authentication or execution failures while the session remains present
stop orchestration. Keep the Mac online, awake and lid open. Work after the last verified
local snapshot can be lost, especially during connectivity loss or runtime expiration.

## Executed verification

- CLI session created on Tesla T4. Python 3.13.16, torch 2.11.0+cu130,
  torchvision 0.26.0+cu130, compatible hosted-runtime exception to local Python 3.14.
- Basic remote script printed HM2_REMOTE_EXECUTION_OK, created a text artifact and
  exited 0. Official `colab download` exited 0 and exact contents matched.
- A 35-second synthetic active kernel job and a separate CLI downloader ran
  concurrently. Checkpoint download exited 0 in 2.043 seconds while the active CLI
  process was still running. Active job later exited 0 with both start/end markers.
- The downloaded synthetic checkpoint was uploaded and restored remotely. Model
  weights, SGD momentum, LR schedule, epoch 3 and global step 1 were checked, with
  HM2_RESTORE_OK saved and downloaded. No GPU optimization used in this synthetic test.
- Actual RetinaNet preflight: three optimizer updates, real training pairs, real
  validation inference, 640/1024 batch 2. Measured 5.974 training images/s and 59.03
  validation ms/frame over a tiny diagnostic sample. Saved complete checkpoint was
  downloaded (277.99 MiB) and loaded locally with optimizer/scheduler/scaler verified.
- Planning estimate from that tiny probe: 13.52 h training plus 0.93 h validation,
  excluding staging and checkpoint overhead. Not a reliable sustained benchmark or
  overnight completion promise. Scientific budgets were not reduced.
- A detached caffeinate synthetic process wrote six heartbeat entries after its
  launching tool call exited. Production controller also checks its PID and log.
- Local pytest: 17 passed. Covers epoch-three-to-five continuation, AMP skipped
  batches, checkpoint compatibility, immutable snapshots, local failed-transfer
  preservation and analysis with nested real configuration fields.
- isort, Black, Flake8, ty and changed-file pre-commit passed. Full repository hooks
  still fail only existing report.md Markdown formatting. That file was not modified.
- Both notebooks pass schema and cell syntax validation. The lightweight results
  notebook was executed with a fresh CPU kernel and all runs unmeasured.

## Transfer verification gate

A direct 277.99 MiB `colab upload` failed with HTTP 400 Bad Request. Its diagnostic URL
contained a proxy query token, so it is deliberately not copied here. CLI logging now
strips URL query strings before persistence. The smaller 36 MiB code/metadata bundle
and tiny checkpoint uploaded successfully. Bounded multipart upload was implemented
to avoid oversized requests. Full checkpoint multipart round-trip verification is
required before launching the long batch. At this note's initial writing it was pending.

Final verification outcome: the multipart helper's initial small remote execution did
not return for more than 300 seconds. The local CLI process was canceled with SIGTERM
(exit -15). Its captured traceback reported `RuntimeError: Connection was lost.` in
the CLI's preliminary `/content` setup execution, before any multipart upload started.
No host source deadlock was found by independent review. Connection troubleshooting
stopped, and long training was not launched. The unused GPU session was stopped.
The full-checkpoint multipart restore path remains unverified. The manual launcher now
requires this actual save/download/upload/restore gate before starting its worker.

No graded experiment started or completed. No overnight controller is running.
Machine-readable evidence is `runs/retinanet_overnight/20261104_001829/persistence_verification.json`.

## Native Colab entry point, superseding CLI execution

Created `notebooks/03_retinanet_experiments_colab.ipynb` for manual native Colab use.
Six code cells cover dependencies/GPU, interactive Drive mount, existing data staging,
input/configuration/Drive checkpoint verification, one explicit training cell, and status.
The existing CLI controller, model, evaluator and TOML recipe remain available. The only
epoch-runner extension is an optional synchronous on_checkpoint callback, default None.
No CLI, UI automation, remote notebook execution or full training was performed for
this native path.

Prepared upload files from the existing cache, excluded from Git:

- `data/exports/retinanet_colab_project.zip`, 9.36 MiB: current src/hm2, shared TOML,
  prepared parquet metadata and validation annotation hash. No secrets or AI artifacts.
- `data/exports/visdrone_train_val.zip`, 9.09 GiB: exactly 27047 official train/val
  frames plus original validation COCO JSON. Every member passed full ZIP CRC testing.

Upload both to My Drive/hm2_retinanet/inputs. The notebook uses native interactive
google.colab.drive.mount(), copies/extracts images locally, and keeps a stable batch ID.
Colab's paired torch/torchvision stack is retained, with auxiliary packages bootstrapped
through uv only when absent. Python 3.12+ is required; Python 3.13 syntax was validated.

New colab_drive helpers copy complete immutable local checkpoint snapshots to mounted
Drive, verify destination hashes before publishing CURRENT.json, and retain the previous
committed generation. Failed handled copies remove their incomplete generation. Initial
real experiment state is synchronized before any training update. A tiny actual CPU
checkpoint write/copy/read/restore is required during setup. All original checkpoint
state is reused, including optimizer, scheduler, AMP, RNG, epoch/cursor and history.

Resume validates requested configuration, dataset hashes and actual anchors before
skipping completed runs. Newer local progress is preserved, including completed
validation at the same optimizer step. Summary JSON/CSV, curves and previews are saved
after each completed experiment. Work since the last successful Drive sync can repeat.
Drive authorization, quota, runtime limits and durable server-side behavior remain for
manual verification in the actual Colab runtime. No bit-for-bit GPU claim is made.

Verification: 20 tests passed; isort, Black, Flake8, ty, changed-file hooks passed.
Full repository hooks still fail only pre-existing report.md formatting. Notebook
schema, all cells and every packaged module parse with Python 3.13 syntax. No saved
training outputs were generated. The actual input/configuration/setup cell passed
locally against the packaged code/metadata and all cached image paths, decoding real
train/val pairs. A local filesystem stood in for Drive for that check, so this is not
an actual native Drive authorization or GPU execution claim. Independent review verified
the persistence, validation-state ordering and compatibility fixes.

## Observed native Drive progress, 2026-11-04

Read the shared experiment folder through the connected Google Drive plugin, without
Colab CLI or UI automation. Latest saved R0-S status: epochs_completed=2 of 5,
batch_cursor=0, global_step=4866, image_exposures=9736, pending_evaluation=false,
completed=false, rollback=null. Both latest.pt (292047607 bytes) and best.pt
(292021311 bytes) exist, with two retained Drive generations. No saved R1-S/R0-F/R1-F
folders were present in the inspected batch. This is a partial experiment, not a
completed four-run comparison.

Measured full official validation at epoch 2: mAP=0.1390799164, AP50=0.2679039556,
AP75=0.1303850965, AP_small=0.0391997976, AP_medium=0.1294689651,
AP_large=0.2792875469, 2846 images, canonical maxDets [1,10,100], original crowds.
Inference=8.54848 frames/s (116.9799 ms/frame), batch2 CUDA AMP end-to-end protocol.
Training batch-latency sum=1402.1572 seconds for two epochs; full epoch2 evaluation
451.8225 seconds. Epoch1 mAP=0.1360292981 and AP_small=0.0427393137. These epoch
differences are descriptive and do not answer anchor/subsampling questions yet.

A newer execution.log entry at 13:03:10 UTC reports epoch2, cursor0, updates4866,
consistent with successful state restoration. Drive metadata and JSON were inspected;
checkpoint binary payloads were not downloaded or independently deserialized. The
connector did not return provider checksum fields requested for the binary files, so
no independent binary-hash validation is claimed. Saved CURRENT manifest contains
checkpoint checksums for the notebook's existing restore validation.

## Checkpoint I/O optimization, 2026-11-04

User's Colab resource screenshot shows 4.9/15 GiB framebuffer memory, not GPU compute
utilization. NVIDIA documents GPU utilization and memory occupancy as separate metrics.
No live utilization or GPU profiler measurements were obtained, and no Colab CLI/UI
access or new training was attempted. Actual saved R0-S records show two epochs with
1402.157 s training batch latency and 952.591 s full validation, about23.4 and15.9 min.
Validation remains every epoch as specified; batch2, resize, optimizer, schedule, data,
losses, epoch counts, checkpoint interval and evaluation protocol are unchanged.

Logs show identical checkpoint state saved at periodic stop, chunk finalization and
next chunk start. At the 07:23 boundary, two duplicate Drive copies alone took24s+22s.
Added a private progress signature cache to avoid serializing/publishing unchanged
state. Compatible latest resume reuses its existing immutable snapshot; missing/stale
ready snapshots are republished without rewriting model bytes. A persistence callback
is still retried before any training, including after prior failed initial Drive copy.
Fallback best checkpoint repairs still force a fresh save. Evaluation completion, metrics,
AMP-skipped batch exposure/cursor changes and optimizer steps remain checkpointed.

An added assertion first reproduced completed resume unnecessarily changing latest.pt's
mtime, then passed after the fix. Parameterized failure regressions cover existing,
missing and placeholder ready manifests: failed persistence retries beforetraining,
and model file is not rewritten. Full suite23passed; isort/Black/flake8/ty and changed
filehooks passed. Fullrepositoryhooksretainonly preexistingreport.md formattingfailure.
Independent review verified the durable-copy retrygate correction. Real GPU speedup is
not measured and must not be claimed.

Updated only the local data/exports/retinanet_colab_project.zip with current source;
Python3.13parse, modulebyteequality andZIPCRC verified. Drive input archive and live
training were not modified. User can replace that9MiBproject archive at their next
runtime restart/resume; the9GiB dataset archive does not need replacement.

## Drive completion audit and Kaggle migration, 2026-11-04

Direct connector reads of CURRENT manifests, status/config/metrics and logs confirmed:
R0-S and R1-S completed5epochs, pending_evaluation=false, with latest.pt and best.pt
present. Each used4868trainingframes,24340exposures and full2846-image validation at
all5epochs. R0-S step12165, training batch latency3576.3372s, final mAP0.1527862620,
AP500.2911784484, AP_small0.0531348737, inference8.03993fps. R1-S step12166,
training3602.5833s, final mAP0.1487815128, AP500.3162684977, AP_small0.0524044254,
inference8.16685fps. These are descriptive single-seed measurements, not significance.
R0-F latest inspected generation1793813838101393993 completed1epoch, step12097,
24201exposures, pending_evaluation=false, mAP0.1575041397. R1-F was not started.
The summary was older than R0-F progress, so per-run status was authoritative.
No checkpoint binaries were downloaded/deserialized during this connector audit.

User exhausted Colab quota and requested native Kaggle continuation, retaining Google
Drive as the single durable store, explicitly rejecting private Kaggle checkpoint datasets.
Added04_retinanet_experiments_kaggle.ipynb, Drive API transport and local userOAuth
authorization command. Credentials stay outside the repository and in a Kaggle Secret;
no credentials have yet been supplied or used. Dataset/project are fetched directly from
the existing Drive inputs. Same batch ID, four runs, source metadata, exact saved recipe,
anchor sizes, epoch budgets and evaluator are retained. Completed experiments skip;
full-state resume uses the existing epoch runner and fixed frame order/cursor.

Drive API publishing uses resumable completed-file uploads, provider checksum verification,
server-side copies of unchanged objects, manifest-last and CURRENT-last publication.
The previous generation stays until replacement verification. Restore recovers retained
predecessors after corruption/missing folders, and persists newer local completed work
before skipping. Folder/copy POSTs avoid blind retries that could duplicate names.
Single-writer execution remains required. Notebook performs an actual tiny save/API
upload/download/restore probe plus real restored-model GPU inference before training,
when the user executes it. A six-hour invocation limit permits planned pause without
reducing total experiment budgets. Cross-runtime exact reproducibility is not claimed.

Local verification:28tests passed, isort/Black/flake8/ty passed, all notebook code and
packaged modules parse as Python3.11. Executed the actual notebook recipe block on CPU
using byte-identical packaged metadata and original full validation annotations; all
four resulting configurations exactly equal the live Drive batch_config.json, including
dataset hashes. Independent review approved after catching/fixing pretrained backbone
optimizer-group compatibility and recovery edge cases. Full pre-commit still fails only
pre-existing report.md formatting. No Kaggle session, OAuth refresh or real GPU test ran.

Updated the existing Drive project archive in place via connector (file ID preserved),
9,817,046bytes, and uploaded the35,148-byte Kaggle notebook into the existing inputs
folder. Verified both via folder readback. Project ZIP CRC and packaged source bytes
were verified locally. Dataset ZIP and all real experiment checkpoint generations were
left unchanged. Authenticating the user's Desktop OAuth JSON and running the notebook
are the remaining runtime-dependent steps.

Follow-up: user completed local OAuth authorization. On 2026-11-04, the private token
file was present outside the repository. An actual credentials.refresh(Request())
succeeded, project-folder write permission was confirmed, and verify_drive_api_checkpoint
uploaded a tiny CPU model/optimizer/scheduler/scaler checkpoint to the real batch's
_probe_api folder, downloaded it, verified integrity and restored state successfully
(step1, verified=true, command exit0). Real experiment checkpoints were unchanged.
No secrets were printed or copied into notes. Kaggle secret setup and actual Kaggle/GPU
execution remain manual and unverified.

## Local stride-5 model submission preparation

User requested a casual submission from an already trained stride-5 model, then chose
local Mac inference rather than another Kaggle workflow. Selected R0-S by measured
validation mAP, without test/leaderboard tuning. Downloaded its existing best.pt and
configuration from the committed Drive generation, verifying manifest SHA256. All3811
test JPEGs are cached locally, about890MB. The official sample_submission.csv is absent
from the cache; a single-file competition request returns404. A bounded metadata listing
did not locate it. No official sample has been fabricated or claimed verified.

Added focused original-pixel submission conversion/validation and a local CLI. It loads
saved model weights, uses the same640/1024 detector settings, maps foreground1–8 to0–7,
exports pixelxywh and retains all test IDs and empty rows. Complete predictions are cached
for later official-sample validation without re-inference. Missing sample produces only
submission_unverified.csv, clearly flagged in logs/metadata; submission.csv requires the
actual sample. No submission was sent and no full3811-frame inference was launched yet.

Actual Apple MPS forward on2realtestimages succeeded in2.954s including cold start.
Existing predict_loader on8realtestimages withbatch2 andonewarmup succeeded at
10.10436frames/s, AMPfalse, sourcepixel outputs. Formatter validation against those8
test-metadata IDs passed with791detections; this was not official-sample validation.
These very small benchmarks do not establish full-run throughput or Kaggle performance.
Review caught a single-use progress generator dropping the warmed-up first batch; removed
it and pass the reiterable DataLoader directly, matching the actual8-image smoke path.
Two focused formatting/identity/count tests passed; suite30tests passed before this
script-only correction and was rerun afterward. Python formatting/lint/types passed.
Full repository pre-commit retains only the pre-existing report.md formatting failure.
Existing Kaggle training, remote checkpoints, project ZIP and dataset ZIP were untouched.

## Shared experiment setup refactor, 2026-11-05

User requested reusable plain Python setup/checkpoint modules, operational guidance in
AGENTS.md, student notebook voice using We, and removal of unused implementation.
Moved generic full-state save/load, atomic JSON writes and immutable snapshot publication
to experiment_checkpoint.py. experiment_setup.py now stages/reuses datasets, verifies
real pairs/splits/source hashes, computes the unchanged selection identity and protects
batch configuration. Updated all callers, including supported legacyCLI generated imports.
The legacyCLI remains referenced/tested and was not treated as unused. Removed duplicate
helper definitions and unused imports. Kept the public retinanet.load_checkpoint export
because the user's already-open older Kaggle notebook imports it on reconnect.

Condensed Colab/Kaggle notebook setup through the shared modules; retained visible
experiment matrices, recipe construction, execution and result tables. Reworked Markdown
in the execution/results notebooks and selected02cells into concise We narration.
Preserved all saved outputs; no GPU notebook/training was rerun. No .artifacts references
remain in notebook sources, metadata or outputs. AGENTS.md records native platform setup,
OAuth/Secrets, input paths, persistence/resume and notebook/math conventions.

Verification:33tests passed, isort/Black/flake8/ty passed. Complete pre-commit still fails
only pre-existing report.md formatting. CPU execution of both actual refactored preflight
prefixes checked all original image paths, decoded real pairs and matched the saved batch's
four configurations and hashes exactly. Fresh-process02import from notebooks/ passed.
Notebook schemas and declared Python syntax passed. Independent review approved after
catching/fixing02bootstrap import ordering. Added one GPU-count boundary regression:
simulated Kaggle2GPU→Colab1GPU resume restores existing active-device RNG without indexing
an absent device; removing the fix reproduces the failure. No bit-for-bit GPU claim.

Updated the local and existing Drive input projectZIP with unchanged metadata bytes and
all current modules; updated the existing Kaggle notebook on Drive and added Colab's
notebook to the same inputs folder. Verified localZIPCRC/source byte equality and actual
provider checksum readback for all three uploads. Dataset and experiment generations
were not changed, and live runtime source was not overwritten or restarted.

## Completed results import and submissions, 2026-11-05

Imported the current committed Drive generations of all four finished experiments.
All downloaded small records, raw histories, exact frame selections and 12 source
previews matched the CURRENT.json SHA256 manifest. The 20 validation records all use
2,846 images, category IDs1–8, maxDets[1,10,100], original COCO crowds and original pixels.
The shared stride selection exactly equals select_frames(local_metadata, train,5);
full and validation selections also match. Training history update/image/time sums
match final status, with AMP skips5/4/21/20 in experiment order.

Epoch5 mAP / AP_small:
- standard_anchors_stride5:0.15278626203892398 /0.05313487369669437
- smaller_anchors_stride5:0.14878151275078325 /0.052404425445697095
- standard_anchors_full:0.1505428541730124 /0.04999856610665351
- smaller_anchors_full:0.15631351990770453 /0.05475879429342399
Best epochs:5,5,1,4; full best mAP0.1575041396626884 and0.1588312332257352.

Versionable compact records are results/retinanet/retinanet_stride5_full_5epochs_seed2026/:
47files/~3.16MB including12unchanged preview JPEGs, exact selections, original config,
status, metrics/environment/logs and100-attempt compact curve records. Largest file is
train_full.json847,037bytes. Provenance includes original source IDs and hashes,
including weights/raw histories/predictions left on Drive. Original training commit
was not recorded; import does not claim bit-exact training reproduction.

Results notebook notebooks/03_retinanet_results.ipynb executed fully on CPU from
versioned inputs, using a separate runs/retinanet_analysis destination; no weights,
images, credentials or GPU are needed to regenerate the presentation. Native Plotly
MIME only;12compact JPEG outputs. Full labels retained via Polars fmt_str_lengths90.
Size899,804bytes. All35tests and relevant Black/isort/flake8/ty plus selected-file
precommit checks passed. Whole-tree precommit still only flags unchanged report.md.
Local hosted project ZIP refreshed with current source and byte-preserved metadata.
No git stage/commit/push was performed; original user wording about pushing was tentative.

User explicitly requested all four test CSVs in submissions/. Best checkpoint binaries
for all four are local and SHA256 verified against committed Drive manifests. Existing
standard_anchors_stride5 MPS prediction cache matched the exact best checkpoint and was
reused. Other three models execute local MPS inference sequentially via existing script,
without training or test tuning. AI-only driver is export_completed_submissions.py in
this notes directory; it is not a runtime project dependency. Detached PID is stored in
runs/submissions/export.pid (initial18825); log runs/submissions/export.log and terminal
marker runs/submissions/EXPORT_COMPLETED. Process parent1 verified; caffeinate prevents
idle sleep. Never restart the active inference merely for cleanup.

At20:54UTC, two CSVs are ready: standard_anchors_stride5(27,555,538bytes522,251detections),
smaller_anchors_stride5(56,027,807bytes1,061,204detections). Both3811test rows and structural
validation passed. Standard full checkpoint epoch1 is active; smaller full epoch4 follows.
Official sample request again returned HTTP404. User supplied the competition format,
which matches our exporter. Validate all IDs against test_images.csv; retain explicit
metadata official_sample_verified=false. CSVs are gitignored; small metadata/manifest
and README will be versioned. User asked four, not just best-of-each-dataset.

Created same-Drive submissions batch folder1NBGbuvGnWHgeikhZFh5KPa1X1gRfuL-F under
submissions folder1SJOmP8kNexuPmQjPT4OcvlkIO_aAGTqi. First two CSVs uploaded and actual
provider checksums verified via existing Drive API (connector metadata omits checksums).
IDs/verified metadata saved in runs/submissions/drive_exports.json. Finish uploading
remaining two, generate submissions/manifest.json with all verified local/Drive hashes,
check all four CSVs, finish notebook/README links, and hand off concrete file paths.
Do not actually submit to Kaggle or stage/commit/push without explicit instruction.

Final state update:
All four CSV exports completed at logged21:01:50. Counts: standard stride522251,
smaller stride1061204, standard full520292, smaller full876729 detections, each3811rows
and max-per-image265/300/261/300. Actual MPS throughput11.06(cached)/10.76/10.98/10.87fps.
Four CSVs and all metadata/manifest uploaded to the Drive submissions batch and their
actual provider hashes verified. User then explicitly rejected committing generated
JSON/log/preview data and asked to delete those local files rather than create a cache.
Do not restore a persistent result cache or suggest committing the records again.

Compact complete result archive fileID1e5AybR4U8T-SMt2nJ8XcipM_2ye5COUn is in the existing
Drive training batch. CRC/source bytes/provider checksum verified before local deletion.
Metadata and CSV manifest are in Drive submissions folder1NBGbuvGnWHgeikhZFh5KPa1X1gRfuL-F.
The complete local results/ tree, five local submission JSONs, newly downloaded raw
histories, generated chart files and new inference model/prediction-cache directories
were removed. About1.11GiB freed. Pre-existing runs/submissions/R0-S cache was preserved.
New four CSVs remain in submissions/ along with README. No new permanent cache exists.
Notebook saved outputs are complete; rerunning requires manually downloading/extracting
the small Drive record archive as documented in its final section. Source image links
now point directly to original Drive files. No data is included in Git changes.

User reported overlapping plot axes. Fixed validation curves to a2x3grid, short metric
facet labels, one short bottom Epoch label and shared horizontal legend with short
standard/smaller+stride/full labels. Applied the compact legend to all nine plots and
used stable colors. Regenerated CPU notebook897499bytes, explicit domain/legend checks
passed, focused layout regression passed. Complete35tests passed. Formatting/lint/types
passed; known whole-tree report.md formatting remains unrelated. No stage/commit/push.


## Frame-sampling decision for subsequent experiments, 2026-11-07

The user decided to use every fifth frame within each training clip for subsequent detector experiments, retaining all 56 training clips (4,868 train frames). Keep the complete official validation split (2,846 frames). This is a practical compute/quality choice, not a claim that full-data training never helps.

Checked the original local metrics.json records restored from the completed RetinaNet Drive batch retinanet_stride5_full_5epochs_seed2026. Epoch-5 results under canonical COCO maxDets=100:

| Configuration | mAP | AP small |
| --- | ---: | ---: |
| standard_anchors_stride5 | 0.152786262 | 0.053134874 |
| smaller_anchors_stride5 | 0.148781513 | 0.052404425 |
| standard_anchors_full | 0.150542854 | 0.049998566 |
| smaller_anchors_full | 0.156313520 | 0.054758794 |

Full minus stride-5 differences in percentage points:
- standard_anchors: mAP -0.2243; AP small -0.3136.
- smaller_anchors: mAP +0.7532; AP small +0.2354.

Full-data training did not improve the standard-anchor final model; smaller anchors gained a modest 0.7532 mAP percentage points with full data. Five full-data epochs have about five times the exposures and updates, so these are different training recipes, not a controlled estimate of frame diversity alone. These are single-seed descriptive differences, without statistical significance claims. Do not rewrite this decision as universal evidence that more data cannot improve detection.

Faster R-CNN baseline completed five epochs on the same stride-5 selection. Retrieved original Drive generation 1794008813488173932 and summary generation 1794008838839340466 from baseline_stride5 (folder 10pqkOV1t-XlHyv4S2KR7IJuqUg9eoiAy). Final reported checkpoint is latest.pt, not selected by validation. Download and submission validation details are recorded in the gitignored run directory.

All 19 files in the committed Faster R-CNN checkpoint generation matched their SHA256 manifest, including the 347,313,572-byte latest.pt. The Drive plugin cannot download files larger than 256 MiB; used the existing local authenticated DriveApi transport for this checkpoint. Loaded the checkpoint on CPU with weights_only=True and verified its configuration, epoch 5, update 12166 and no pending validation. No training/inference was run.

Restored the already exported CSV instead of recomputing test predictions. Local deliverable: submissions/faster_rcnn_baseline_stride5.csv (15,195,847 bytes), SHA256 8bd7be9307fef48be31c7f2d36ca42b2f4dbd8959d7d13cc0fa7d62fc2f5caa0. Verified 3811 unique rows in exact local test_images.csv order, 287745 detections, no empty rows, maximum 142 detections/image, class IDs0–7, finite scores/pixel top-left boxes and original image bounds. Official sample_submission.csv remains unavailable; official_sample_verified is false. No Kaggle submission performed.

Final Faster R-CNN canonical maxDets100: mAP=0.1747032853, AP50=0.3535210966, AP_small=0.0806749811; these are COCO-style metrics, not competition scoring. Training batch time4537.9055seconds, image exposures24340. Do not attribute family comparison solely to architecture: ignored-region handling also differs from RetinaNet. Renamed the results notebook from05 to04 to close the gap left by the removed smoke notebook; updated tracked references and added a general numbering instruction to AGENTS.md.


## Faster R-CNN epoch-1 recovery audit, 2026-11-07

User authorized test inference from epoch 1 if its checkpoint survived. No inference/training was run because weights were not found. Checked local checkpoint paths; Drive current and predecessor generations; current latest.pt revision history (one revision only); original epoch-1 generation IDs from execution log; available named checkpoint copies, including trash, via authenticated Drive API. Both retained Faster R-CNN generations (1794008497293749034 and1794008813488173932) hold epoch5/update12166 with identical latest.pt SHA256 2e068f71d4f3c53e5830ad4c7a41f73525e11f14546f88b99169c9e75ed52203. No epoch-1 generation was returned by the API. The available larger Faster R-CNN-sized checkpoints are just these two final copies. Metadata audit: runs/faster_rcnn/epoch1_recovery_audit.json.

The existing runner saves only latest.pt (configured final-epoch reporting), and Drive transport deletes generations other than current and predecessor. Thus epoch_1_metrics.json exists but does not contain model weights; cannot reconstruct epoch1 from metrics or rewind epoch5. Preserve this distinction in future reasoning. Investigating a new schedule or early stopping would require an explicitly configured new training run, and a separate selection checkpoint retained outside rolling resume cleanup. Do not silently change the completed final-epoch protocol or fabricate an epoch1 submission.
