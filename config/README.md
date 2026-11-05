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
