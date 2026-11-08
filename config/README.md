# RetinaNet experiment configuration

`retinanet_overnight.toml` stores one shared training recipe at the root. Required
keys are `seed`, `epochs`, `batch_size`, `num_workers`, `min_size`, `max_size`,
`learning_rate`, `flip_probability`, `subset_stride`, and `checkpoint_seconds`.
These explicit values define the comparison; there are no additional TOML tables
or configuration-loader defaults.

The four experiments run in order: `standard_anchors_stride5`, `smaller_anchors_stride5`,
`standard_anchors_full`, and `smaller_anchors_full`. The notebook adds the
experiment ID, standard or smaller anchor variant, training stride, actual anchor sizes, and
dataset-selection hash. Warmup occupies the same epoch fraction as 100 subset
updates. Learning rate decays at two thirds and 90% of the training budget.

Notebook tables and plots use descriptive labels. `hm2.experiment_names` maps these
names to historical storage IDs so existing directories and checkpoints resume
without changing their saved configuration.

We use the native [Colab notebook](../experiments/00_retinanet_experiments_colab.ipynb)
or [Kaggle notebook](../experiments/01_retinanet_experiments_kaggle.ipynb), with the
same durable Google Drive batch. Setup, credentials, storage paths, automatic
checkpointing and resume are specified in
[AGENTS.md](../AGENTS.md#hosted-experiment-execution). Shared setup and checkpoint
functions are in `hm2.experiment_setup` and `hm2.experiment_checkpoint`.

The original `scripts/run_retinanet_overnight.py` CLI controller remains a
separately invoked legacy interface. Remote CLI execution proved unreliable in
this environment, so it is not the default launch path or evidence of reliable
unattended execution.

We evaluate the entire official validation split after each epoch and select
`best.pt` by canonical COCO mAP. Five full-data epochs use approximately five
times the subset's image exposures and optimizer-update budget. This comparison
combines data coverage and training budget; exact competition-specific ignore
semantics remain unverified.

## Controlled data ablations

`data_photometric_augmentation.toml`, `data_input_resolution.toml` and
`data_repeat_factor_sampling.toml` retain the tables from
`faster_rcnn_baseline.toml`. The extra `[photometric]` table contains brightness,
contrast and saturation intervals plus probability. `[sampling]` contains the
image-presence threshold and explicit repeat cap. Higher resolution changes only
`[model]` min_size/max_size to 800/1280. Test export is not part of these ablations.

`yolo_baseline.toml` replaces the unrun ignore-aware RetinaNet proposal with an
anchor-free YOLO11s baseline. It uses the same dataset/training/evaluation tables,
`backend = "ultralytics"`, `model_id = "yolo11s.pt"`, eight classes and explicit
CIoU/BCE/DFL gains in `[loss]`. The shared loop owns SGD, AMP and full resumable checkpoints, so native
Ultralytics augmentation/optimizer defaults do not apply. Package version 8.3.235
is pinned and recorded. This is a model-family baseline, not a data ablation.

`experiments.protocol.load_settings` resolves each recipe. Disabled features return
the original baseline settings, and experimental settings are included in checkpoint
identity. Worker counts, storage and runtime controls keep their existing behavior.
Research notebooks 06 through 09 and the
[shared Kaggle entry](../experiments/04_data_experiments_kaggle.ipynb) expose execution
switches explicitly. Full protocol and run instructions are in
[experiments/README.md](../experiments/README.md).

## Native YOLO11s pipeline

`yolo11s_native_1280.toml` configures the final pipeline in
[notebook 10](../notebooks/10_yolo11s_native_pipeline.ipynb). It has four required
tables, read by `hm2.yolo_pipeline.load_pipeline_settings`:

- `[dataset]`: training `frame_stride`, the long side `image_size` of the prepared
  JPEGs, `jpeg_quality`, `should_mask_ignored`, `ignore_fill` and `validation_clips`.
  Validation clips are held out whole from either official split; every other labeled
  clip trains. The table also records the expected frame, clip and test counts checked
  before training.
- `[model]`: `backend = "ultralytics"`, `model`, the initial checkpoint `model_id` and
  the pinned `ultralytics_version`. `scripts/build_yolo_bundle.py` creates
  `yolo11s_visdrone_det_8cls.pt` from the pinned VisDrone-DET checkpoint and ships it in
  the code bundle.
- `[training]`: native Ultralytics arguments passed explicitly (optimizer, schedule,
  augmentation, `batch`, `device`, `seed`), the throughput-based epoch plan
  (`budget_hours`, `min_epochs`, `max_epochs`, `validation_time_margin`) and the
  `smoke_fraction` of the throughput check.
- `[inference]`: candidate `image_sizes` compared on validation, `confidence`, NMS
  `iou`, `max_detections`, `batch` and `half`.

There are no hidden defaults. Changing `device`, `batch` or `image_size` changes the
experiment. The design and run instructions are in
[docs/yolo_native_pipeline.md](../docs/yolo_native_pipeline.md).

`yolo11s_native_experiments.toml` defines the tuning runs of
[notebook 11](../notebooks/11_yolo11s_tuning_experiments.ipynb). `base_config` names the
recipe above and `baseline_checkpoint` the submitted checkpoint shipped in the bundle.
Each `[experiments.<name>]` and `[inference_experiments.<name>]` table has a
`description` and overrides for the `dataset`, `training` or `inference` tables.
`hm2.yolo_pipeline.apply_experiment` rejects unknown keys. The only keys allowed beyond
the base recipe are the Ultralytics arguments `freeze`, `box`, `cls` and `dfl` in
training and `augment` in inference.
