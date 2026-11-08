# AGENTS.md

## Project Context

This repository is for the UCU AI 2026 object detection homework using VisDrone2019-VID
video frames. Predict bounding boxes and one of eight classes: person, bicycle, car, van,
truck, tricycle, bus, and motor. The main deliverable is a reproducible, notebook-first
comparison of detector families followed by a justified final prediction pipeline.

Reserve `notebooks/` for research notebooks and results intended for the report. Store
execution-only Colab and Kaggle notebooks in `experiments/`. Keep reusable
code in `src/` when it improves clarity, reduces duplication, or makes testing easier.
Do not introduce unnecessary infrastructure or hide all experiments behind library calls.

## Development Workflow

1. Inspect relevant code and configuration before making changes. Follow existing project conventions.
2. Make the smallest coherent change that solves the task. Avoid unrelated refactoring and unnecessary abstractions.
3. Use `uv` exclusively for Python environments and dependency management. Run project commands through `uv run`.
4. Update `pyproject.toml` and `uv.lock` together when changing dependencies.
5. Run relevant tests and quality checks after modifications. Never claim a check passed unless it was executed.
6. Report changes, verification results, and remaining issues concisely.
7. Never stage, commit, push, or create branches unless explicitly requested.
8. Keep Python, PyTorch, TorchVision, CUDA, and detector-library versions compatible with the actual runtime. Do not carry over unrelated project dependency pins. Document any necessary Python-version or GPU-specific exceptions.

## Python Style

01. Target Python 3.14 when supported by the required CV and GPU stack. Use a documented compatible Python version when a required dependency or hosted GPU runtime does not support 3.14. Write clean, idiomatic, readable Python.
02. Add parameter and return type annotations to every function and method. Prefer built-in types such as `list`, `dict`, `tuple`, and `str`, and modern unions such as `str | None`.
03. Do not use `from __future__ import annotations`. Avoid importing `typing` when built-in syntax is sufficient.
04. Prefer comprehensions and standard Python constructs when they improve readability.
05. Avoid code comments in reusable modules. Prefer expressive naming and straightforward implementations. In notebooks, use Markdown cells to explain decisions and brief code comments only where needed to clarify non-obvious steps.
06. Avoid leading underscores for public variables and functions. Use one leading underscore for private names, Python protocols, or external APIs where appropriate.
07. Use `pathlib` for filesystem operations.
08. Keep functions focused and avoid unnecessary classes, wrappers, and helper functions.
09. For imports do not use `import *`. You should import `polars` as `pl`, `polars.selectors` as `cs`, `plotly.express` as `px`, `plotly.graph_objects` as `go`, `plotly.subplots` as `sp`, `numpy` as `np`, `streamlit` as `st`. For all other libraries, use the default import name or use `from <module> import <class/function>` for specific classes or functions. Avoid aliasing unless necessary to avoid name conflicts or improve clarity.
10. Limit the length of lines to 100 characters, as required by Flake8 configuration. For long strings, use implicit concatenation with parentheses instead of backslashes. Avoid using `# noqa` to bypass style checks.
11. Read environment variables and `.env` configuration using `decouple` via `from decouple import config`. Do not use `os.environ` or `os.getenv` to read settings or secrets.
12. For all Python scripts and modules, use `logging` to log messages. Use `info` level for normal operation messages, where processing takes a long time. Avoid using `print()` for logging or debugging. For each standalone script, configure logging to output to stdout with a timestamp, log level, and logger name as follows:

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
```

## Naming Conventions

1. Use snake_case for variables, functions, and methods. Use PascalCase for classes. Use UPPER_SNAKE_CASE for constants. Prefix internal/private attributes and methods with a single underscore (e.g., \_internal_state).
2. For boolean variables, use names that clearly indicate their meaning. Always use prefixes such as `is_`, `has_`, `can_`, or `should_`. For example, use `is_valid`, `has_permission`, `can_execute`, or `should_retry`.
3. Use strong verbs that describe the exact action being performed. Avoid ambiguous names like `process_data()`, `handle_event()`, or `do_work()`. Instead, use `parse_market_data()`, `route_incoming_event()`, or `calculate_volatility()`.
4. Specific Class Nouns: Use descriptive nouns that define the entity's exact role. Avoid vague suffix crutches like `Manager`, `Handler`, or `Processor`. Instead of `DataProcessor`, use `DataNormalizer` or `MetricsAggregator`.
5. Avoid arbitrary abbreviations, but accept domain-standard acronyms (e.g., `http`, `uuid`). While descriptive loop variables (e.g., for record in records) are strongly preferred, standard mathematical or index variables (`i`, `j`, `x`, `y`) are acceptable for simple iterations or coordinate math. Always use _ for throwaway variables in unpacks or loops. Do not use `df` acronyms for DataFrames. Use descriptive names like `user_data`, `transaction_records`, or `sensor_readings` instead.

## Configuration Conventions

1. Store human-edited project runtime configurations as TOML in `config/`. Keep framework-required formats, such as Ultralytics `data.yaml` or native detector configs, when applicable. Use JSON for metadata and generated result records. Keep tooling files in their required formats, such as `pyproject.toml`.
2. Use `snake_case` keys. A configuration for one component stores its options at the root. A configuration composing multiple components groups their options in named tables, such as `[dataset]`, `[training]`, and `[evaluation]` for detection experiments.
3. Use `backend` to select an implementation when configurable, `model` for the detection architecture, and `model_id` for a pretrained checkpoint identifier or path. Keep backend options in the same table, using the names accepted by the implementation. Avoid wrapper tables or unused fields added only to enforce a uniform shape.
4. Omit optional settings to use documented defaults. TOML has no `null` value. Use native TOML values for booleans, numbers, strings, and arrays.
5. Keep secrets in environment variables or `.env`, read through `decouple`. Configuration may name a secret variable, such as `api_key_env = "WANDB_API_KEY"`, but must not contain its value.
6. Document required tables, supported backends, and defaults in the component's README. Keep configuration filenames, loader behavior, CLI defaults, and examples consistent.
7. Keep reusable dataset, annotation, training, evaluation, and inference logic in focused `src/` modules when needed. Keep model selection, experiment design, comparisons, and conclusions visible in notebooks. Do not add a framework abstraction unless it eliminates real duplication.

## Docstrings

1. Add docstrings to all public functions, methods, and classes.
2. Do not add docstrings to private functions, private methods, private classes, special methods, any code in Jupyter notebooks, or in the header of a file.
3. Use imperative mood for function and method docstrings. Use descriptive mood for class docstrings.
4. For parameters, attributes, and return values, provide 1-sentence description, unless the type is nested or complex, in which case provide a brief description of the structure and contents (e.g. key and value descriptions for a dictionary).
5. For large functions that have more than 60 lines of code, include a brief description of the algorithm between the summary and the `Args` section. Use blank lines to separate the summary, algorithm description, and `Args` section.
6. Use modified Google-style docstrings with a 1-to-3 sentence summary. Do not include types in the docstring, as they are already specified in the function signature.
7. Do not use `Raises` section in docstrings. For important exceptions, include them in the summary or algorithm description instead. But do not include every possible exception, only the ones that are relevant, very important and require the caller to handle them explicitly.

Example of docstrings:

```python
def function(param1: int, param2: str) -> bool:
    """{1-3 sentence summary of the function}

    {brief description of the algorithm if the function is large}

    Args:
        param1: {description of `param1`}.
        param2: {description of `param2`}.

    Returns:
        {description of the return value}.
    """

def generator(param1: int) -> Generator[int, None, None]:
    """{1-3 sentence summary of the generator function}

    {brief description of the algorithm if the function is large}

    Args:
        param1: {description of `param1`}.

    Yields:
        {description of the yielded values}.
    """

class Class:
    """{1-3 sentence summary of the class}

    Attributes:
        attr1: {description of `attr1`}.
        attr2: {description of `attr2`}.

    Args:
        param1: {description of `param1`}.
        param2: {description of `param2`}.
        param3: {description of `param3`}.
    """
    attr1: str
    attr2: int

    def __init__(self, param1: str, param2: int = 0, param3: list[str] | None = None) -> None:
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = [] if param3 is None else param3

    def method(self, param1: int) -> bool:
        """{1-3 sentence summary of the method}

        {brief description of the algorithm if the method is large}

        Args:
            param1: {description of `param1`}.

        Returns:
            {description of the return value}.
        """

    @property
    def readwrite_property(self) -> list[str]:
        """{description of `readwrite_property`}"""

    @readwrite_property.setter
    def readwrite_property(self, value: list[str]) -> None:
        ...
```

## Simplicity and Error Handling

1. Reason through requirements, assumptions, and likely failure modes before writing code. Prefer simple code over defensive programming. Add guards only for concrete requirements or observed failures, not hypothetical cases.
2. Assume internal inputs satisfy documented contracts. Do not repeatedly validate values already guaranteed by the calling code.
3. Avoid speculative `if` checks, redundant `raise` statements, and excessive `try/except` blocks.
4. Validate inputs at meaningful system boundaries, especially when reading external data, configuration, or API responses.
5. Catch exceptions only when recovery or useful additional context is possible. Never silently swallow errors.
6. Do not use catch-all and reraise patterns for cleaning. Instead, use context manager where possible and if not, then `try/finally` pattern.
7. Do not use `assert` for production input validation. Use normal pytest assertions in tests.
8. Do not introduce fallback behavior unless it is explicitly required.
9. Never throw errors for operations where the semantic intent can be gracefully satisfied, for example:
   - Enforce mathematical identities where they exist: `sum([])` should return `0`. Do not invent values for undefined statistics such as `mean([])`.
   - Make deletions idempotent: `delete(missing_record)` or `rm(missing_file)` must return success. The intent ("ensure it's gone") is already met.
   - Forgive boundaries: array slicing outside existing bounds should yield an empty array, not an `IndexOutOfBounds` exception.

## Tooling and Verification

| Purpose                      | Tool                                                                 |
| ---------------------------- | -------------------------------------------------------------------- |
| Dependencies and environment | uv                                                                   |
| Configuration and secrets    | decouple                                                             |
| Formatting                   | Black                                                                |
| Import sorting               | isort                                                                |
| Linting                      | flake8                                                               |
| Static type checking         | ty                                                                   |
| Tests                        | pytest                                                               |
| Git hooks                    | pre-commit                                                           |
| Interactive notebooks        | JupyterLab                                                           |
| Tabular data                 | polars                                                               |
| Plotting                     | plotly, plotly-express                                               |
| Deep learning                | torch, torchvision                                                   |
| Object detection             | torchvision; other frameworks (e.g., ultralytics) as needed          |
| Image loading and overlays   | Pillow, OpenCV                                                       |
| COCO evaluation              | pycocotools                                                          |
| Bounding-box augmentation    | albumentations, if used                                              |
| Experiment tracking          | Notebook tables and saved artifacts; external trackers if configured |

Use the existing configuration instead of introducing alternative tooling.

After relevant Python changes, run:

```bash
uv run isort --check-only .
uv run black --check .
uv run flake8 src/ scripts/ tests/
uv run ty check
uv run pytest
```

Use `uv run pre-commit run --all-files` for complete repository validation.
Run directory-specific checks only for paths that exist; follow the repository configuration.

Report failures rather than weakening checks to make them pass.

## Data and Experiments

01. Prefer Polars for tabular data processing. Use native expressions, `with_columns`. Avoid Python loops for operations that can be expressed directly in Polars.
02. Use lazy evaluation only for multi-step pipelines that benefit from query optimization or streaming processing. Avoid lazy evaluation for single-step operations.
03. Preserve existing experiment seeds and splits unless explicitly changing the experiment. Use seed 2026 for new experiments. Keep experiments reproducible; document unavoidable GPU nondeterminism.
04. Keep training, validation, and testing boundaries clear. Prevent data leakage.
05. Record configurations, seeds, metrics, and relevant outputs in the experiment notebook and appropriate project output directories. Store dataset metadata in `data/metadata/`. Keep a concise comparison table containing the model, changed parameter, validation protocol, and result.
06. Compare experiments against a consistent baseline. Do not report unmeasured improvements or invent results.
07. Split by video sequence (`clip`), never by randomly assigned frames. Keep the provided validation set separate, and avoid using test frames or leaderboard feedback to choose preprocessing or hyperparameters.
08. Track object counts per class, box sizes, scene density, frame resolutions, and distribution changes across clips during EDA.
09. Handle ignored regions as `don't care` according to each model's and evaluator's supported conventions. Do not silently treat them as negative background. Verify any ignored-region filtering or masking visually.
10. Keep box conversions explicit: YOLO uses normalized `class cx cy w h`; COCO uses pixel `x y w h`, with category IDs 1–8; submissions use pixel top-left `class score x y width height`, with class IDs 0–7.
11. Report consistent detection metrics, including mAP, AP50, AP_small, and inference speed. Keep the validation evaluator's ignore-region and max-detections behavior documented, and distinguish COCO-style metrics from competition-specific scoring if they differ.
12. Test round-trip coordinate transformations, especially after resizing, padding, tiling, augmentation, and returning predictions to original image coordinates.
13. Validate submissions against `sample_submission.csv`: one row per test `image_id`, empty predictions allowed, pixel top-left coordinates, and no more than 500 scored detections per image.
14. Use descriptive experiment names in notebook prose, code, tables, plots, captions, logs, and reports. Name the model and changed setting, such as `standard_anchors`, `smaller_anchors`, or `diou_regression`, and include the data selection when relevant, such as `standard_anchors_stride5` or `smaller_anchors_full`. Use readable labels such as “RetinaNet with smaller anchors (every fifth frame)” in presentation. Do not use opaque numbered run codes as public names. Keep historical checkpoint IDs only at the storage boundary through `hm2.experiment_names`; do not rename existing run directories or change saved checkpoint configurations just to relabel results.
15. Store downloaded completed-experiment JSON, logs, previews and submission metadata on Drive. Restore local records only when needed for rerunning, then delete temporary copies unless the user asks to keep them. Version the research notebook outputs, reusable code and instructions for restoring the records from Drive.

## Notebooks

01. Use Jupyter for notebooks. Store report and research notebooks as `.ipynb` files in `notebooks/`; keep execution-only notebooks in `experiments/`. Use the project's Python 3.14 environment when compatible, or a documented GPU-compatible kernel locally or remotely. Launch a local JupyterLab session from the repository root with `uv run jupyter lab notebooks/` when appropriate.
02. Use dedicated Markdown cells for static explanatory text. Avoid generating narrative paragraphs through `print()` or other output wrappers.
03. If narrative text has to be generated programmatically (e.g., for reporting variables), use `IPython.display.Markdown` with `display()`. Use `IPython.display.Image` for images and `ipywidgets` for interactive controls.
04. Do not use `print()` for output in notebooks. Use cell return values or `IPython.display.display()` for multiple outputs.
05. For slow iterative computations, use `tqdm.notebook.tqdm(iterator)` to display progress bars. Avoid verbose console output.
06. Let notebook cells display their results directly when possible. Avoid verbose console output.
07. Use consistent mathematical notation across code, plots, and explanations.
08. Treat notebooks as the primary presentation and experiment interface. Show data decisions, model configuration, the relevant training/evaluation calls, plots, comparisons, and conclusions there. Move repeated or testable implementation details into `src/` instead of duplicating them across notebooks.
09. Inspect notebook code and metadata for hardware, data, dependency, side effect, and runtime requirements before execution. Prefer an existing configured remote kernel with the required environment when available. Use a local kernel only for verified lightweight CPU notebooks.
10. Do not automatically rerun notebooks requiring a GPU, model training or fine-tuning, or other expensive computation, locally or remotely. Such runs require explicit user authorization and suitable compute.
11. When execution is appropriate, restart the selected kernel and run all cells in order. If suitable compute is unavailable, requirements are unclear, or a run is not authorized, validate notebook schema and code syntax without executing cells. Preserve saved outputs when skipping execution and report that execution was not run, why it was skipped, and which outputs may be stale. Follow `notebooks/README.md` for naming and output conventions, if present.
12. Prefer numbered notebooks such as `00_data.ipynb`, `01_eda.ipynb`, detector-family experiments, and final inference/submission. Adapt names to the repository instead of enforcing a rigid notebook count. Keep numbered files in logical dependency and presentation order, with consecutive prefixes within each directory. Number `notebooks/` and `experiments/` independently. After adding, moving or deleting a numbered file, check the complete sequence and update affected filenames, links and references together. Do not leave gaps from removed notebooks or rename historical experiment IDs and checkpoint directories to match notebook numbering.
13. Keep notebook outputs concise and useful for review. Avoid huge logs, embedded datasets, and unnecessary checkpoint outputs. Make major results understandable without executing expensive training.
14. Write the notebook as a student research narrative in first-person plural: use **We**, never **I**. Follow the concise structure of `00_data.ipynb` and `01_eda.ipynb`: question, assumptions, visible configuration, execution, measured results, and limitations. Keep the experiment matrix and scientific choices visible even when implementation is imported from `src/`.
15. Keep AI working notes, agent handoffs, internal task prompts, permission history, and development-status reports out of notebooks. Do not mention AI artifacts or `.artifacts/` in notebook prose, code, metadata, or outputs. Put platform operation instructions here, and keep only concise reproducibility notes in research notebooks. Do not invent student conclusions or replace the student-authored final report.
16. Use `$...$` for inline Markdown math and `$$...$$` for display math, with display delimiters on their own lines. Do not use `\(...\)` or `\[...\]` in Markdown; this rule does not apply to `.tex` sources.

## Hosted Experiment Execution

Use the existing native notebooks as working examples:
`experiments/00_retinanet_experiments_colab.ipynb` and
`experiments/01_retinanet_experiments_kaggle.ipynb`. Keep the research comparison in
`notebooks/03_retinanet_results.ipynb`. These execution notebooks use the
same experiment identities and durable Drive layout, so either platform can resume
compatible saved work.

For Faster R-CNN, use `experiments/02_faster_rcnn_experiments_kaggle.ipynb` and
`src/hm2/faster_rcnn_experiments.py`. Its research results belong in
`notebooks/04_faster_rcnn_results.ipynb`. Reuse the existing Drive transport and
checkpoint utilities rather than introducing another training loop.

### Shared Python modules

- Use `src/hm2/experiment_setup.py` for `stage_dataset`, `verify_dataset`,
  `dataset_identity`, and `save_experiment_config`. Reuse the existing raw archive,
  preserve prepared metadata byte for byte, verify real image/target pairs and split
  boundaries, and compare the exact recipe before resuming.
- Use `src/hm2/experiment_checkpoint.py` for `save_checkpoint`, `load_checkpoint`,
  `write_json`, and `publish_snapshot`. These functions accept ordinary PyTorch
  models and optimizers; reuse them for later detector families. Keep model-specific
  training and evaluation in the corresponding model module.
- Use `src/hm2/colab_drive.py` for mounted-Drive synchronization and restoration.
  Use `src/hm2/drive_api.py` for the same snapshot layout through authenticated Drive
  API transfers. Reuse these transports rather than adding another storage service.
- Keep a small bootstrap cell for dependencies and making `src/` importable when
  opening a hosted notebook from a project ZIP. Move repeated data preparation and
  checkpoint implementation into these modules. Do not add an experiment-management
  framework or hide all model configuration behind one orchestration call.

### Input bundle and paths

1. Put the current project ZIP and reusable dataset ZIP in
   `My Drive/hm2_retinanet/inputs/`. The current filenames are
   `retinanet_colab_project.zip` and `visdrone_train_val.zip`. The project ZIP contains
   the current `src/hm2/`, configuration, and prepared `data/metadata/`, including
   the source hash manifest `colab_inputs.json`. Include all imported modules and
   verify the ZIP before use. Exclude credentials, `.env`, `.artifacts/`, caches and
   model weights. Local paths on the Mac are not accessible to hosted runtimes.
2. Reuse an existing matching extracted dataset. Otherwise, transfer and verify the
   existing archive and extract onto fast local runtime storage. Check disk capacity
   for the ZIP, extracted images, checkpoints, and outputs before transferring.
   Do not read training images directly through a slow Drive mount.
3. Colab uses `/content/hm2` for code and `/content/visdrone` for images. Kaggle uses
   `/kaggle/temp/hm2` and `/kaggle/temp/visdrone`, with outputs under
   `/kaggle/working/runs/`. Preserve the actual source metadata and frame identifiers
   when changing runtime or detector family.
4. Local outputs use `runs/<experiment_group>/<batch_id>/`, with one directory per
   experiment. Durable outputs use `My Drive/hm2_retinanet/batches/<batch_id>/`.
   The current batch is `retinanet_stride5_full_5epochs_seed2026`. Keep this ID and
   recipe when continuing it; use a new ID for a scientifically different batch.
5. When changing shared modules, update and verify the project bundle and its
   notebook consumers together for the next launch. Do not overwrite a running
   runtime's source tree, replace its checkpoint files, or restart it for cleanup.
6. Create every destination's parent directory before opening a download, temporary
   file or archive. For example, call
   `archive_path.parent.mkdir(parents=True, exist_ok=True)` before opening its
   `.incoming` file. A fresh Kaggle runtime may not have `/kaggle/temp` at all.

### Colab launch

1. Open the Colab entry notebook manually, select a GPU runtime and run setup.
   Use native `google.colab.drive.mount("/content/drive")`; the user completes
   interactive Google authorization. Read the mounted input archives, stage images
   under `/content`, and use the existing mounted-Drive persistence functions.
2. Retain the hosted runtime's compatible CUDA PyTorch/TorchVision pair. Install
   missing auxiliary packages with `uv`; do not apply the local Python 3.14 lock
   to an incompatible hosted interpreter. Record the actual Python, CUDA, GPU and
   library versions. The current Kaggle notebook parses on Python 3.11 or newer;
   the Colab notebook uses Python 3.12 or newer.
3. Inspect the experiment table and restored progress before the training cell.
   Keep synthetic storage and GPU probes out of the training path. One Run All invocation may perform
   setup and then training once authorization is complete. Avoid simultaneous runs.
4. Browser disconnection may leave the runtime running, but session expiry and GPU
   quota can terminate it. Inspect active execution and saved progress before
   restarting. Do not promise unattended completion or use anti-idle workarounds.
5. Colab CLI execution has failed with `RuntimeError: Connection was lost` in this
   environment. Native notebook execution is the established path. Do not resume
   CLI troubleshooting or use Computer Use, browser automation, screenshots or
   VS Code UI automation for Colab access. Keep the existing CLI controller only
   as a separately invoked legacy interface; do not treat it as verified unattended
   execution or choose it automatically for new experiments.

### Kaggle launch with the same Google Drive storage

1. Import the Kaggle entry notebook into a private notebook. Enable Internet and
   GPU. Current experiments use one GPU and batch size 2, even if two T4 devices
   are visible. Changing GPU count or batch size is an explicit experiment change.
2. Enable Google Drive API in a Google Cloud project and create a user OAuth client
   of type **Desktop app**. For an External app in Testing, add the Google account
   owning the experiment folder under Google Auth Platform → Audience → Test users.
   The Desktop client JSON alone does not authorize access.
3. On the Mac, run
   `uv run python scripts/authorize_drive.py --client-json /absolute/path/to/client_secret.json`.
   The user completes consent. The command stores refreshable credentials privately
   in `~/.config/hm2/drive_token.json`. Keep this file out of Git and notebook cells.
4. Copy the entire authorized token JSON into Kaggle Secret `HM2_DRIVE_TOKEN` and
   enable the notebook's access to that secret. The notebook reads it through
   Kaggle's `UserSecretsClient` and authenticates the Drive API. Access tokens
   refresh automatically; External Testing refresh tokens can expire after seven
   days, requiring the local authorization command again.
5. Fetch and verify the inputs required by the selected entry notebook: RetinaNet
   reads code and data archives from Drive; Faster R-CNN reads code from Drive and
   competition data from attached Kaggle inputs. Restore checkpoints from Drive.
   Check write permission and available quota. Verify actual checkpoint transfers
   through the normal save and restore path. Do not use
   `google.colab.drive.mount()` on Kaggle or assume `/kaggle/working` is durable.
6. Execute setup, inspect restored progress, then start the explicit training cell.
   For native background execution, use Save Version → Save & Run All with secret
   access enabled. Stop any interactive training writer before starting that run.
   Never write to the same batch concurrently from two notebooks or platforms.
7. A session deadline bounds one invocation, not the total epoch budget. Leave time
   for a final checkpoint and full validation. On planned pause or interruption,
   rerun setup and training with the same batch ID and recipe. Completed runs skip
   and interrupted runs resume automatically from the last verified durable state.

### Test inference and submission export

1. Prepare test inference and CSV export together with every Kaggle training notebook.
   Run inference on the Kaggle GPU immediately after each completed experiment, using
   the same runtime and saved model settings. Do not default to downloading large weights
   and running full test inference on the Mac. Local inference requires an explicit request.
2. Keep an inference-only mode for completed checkpoints. It must restore the selected
   final or explicitly chosen best checkpoint, skip training, and export every test image.
   Validate the actual test_images.csv and test paths; a train/validation archive is insufficient.
3. Stage only the images needed by the selected mode. Record checkpoint and test metadata
   hashes, class conversion, resize/NMS settings and the selected epoch with each CSV.
   Use sample_submission.csv when present; otherwise state validation against test_images.csv.
4. Write descriptive CSV filenames under `/kaggle/working/submissions/`, and include CSVs
   and export metadata in verified Drive snapshots. Reuse completed exports only when
   checkpoint, test metadata and CSV hashes match. Never publish a partly written CSV.
5. Current entry: `experiments/03_data_experiments_kaggle.ipynb`, with
   `SHOULD_TRAIN = False` and `SHOULD_EXPORT_SUBMISSIONS = True` for the three completed
   Faster R-CNN data experiments. For future training enable both switches. Keep the
   updated `controlled_data_submissions_project.zip` in `hm2_retinanet/inputs/` on Drive.
   Export CSVs without automatically submitting them to the competition.

### Required pipeline for every future experiment

01. Keep the operational pipeline the same for every detector family. Reuse the
    established bootstrap, data staging, preflight checks, training callbacks,
    Drive synchronization, resume and result export. Change only the model-specific
    training/evaluation code and explicit scientific configuration. Do not create
    another setup, checkpoint transport or experiment-management framework per model.
02. Use the established Drive-first input workflow. Download the verified code bundle
    from `My Drive/hm2_retinanet/inputs/` through `HM2_DRIVE_TOKEN` on Kaggle or the
    native Drive mount on Colab. Reuse the existing dataset archive and prepared
    metadata where compatible. Do not introduce manual code/dataset attachments as
    an unstated prerequisite. If an existing experiment requires attached competition
    data, declare that requirement before launch and verify those inputs explicitly.
    Changing the data source must preserve frame identifiers, annotations and splits.
03. Build a small gitignored code ZIP with `src/hm2/` and `config/` at the root, plus
    prepared metadata only when the consumer requires it. Exclude secrets, weights,
    caches and datasets. Verify ZIP integrity, imported-module coverage and byte-for-byte
    agreement with current source files. Updating the ZIP does not update notebook
    cells already uploaded to Kaggle. State which notebook and bundle must be replaced;
    do not request another archive upload for a notebook-only fix.
04. Make setup work in an empty runtime. Create parent directories before opening
    downloads or temporary files; do not assume `/kaggle/temp` exists. Verify GPU,
    compatible dependencies, credentials, write permission, disk and Drive quota.
    Read real image/target pairs and verify expected selection and validation counts.
    Check test images and their row metadata when export is configured; a train/val
    archive does not establish test-data availability. Inspect actual paths before
    diagnosing missing inputs versus unsupported directory layouts.
05. Show effective configuration, data selection, stage, batch ID and output paths
    before training. Environment overrides must be visible. A stage named `full`
    means the complete configured experiment, not necessarily all training frames.
    Explain whether Run All starts training. Preserve existing seeds, scientific
    settings and run identities during operational fixes.
06. Start the requested training run without synthetic GPU or storage probes and
    without a `probe.json` prerequisite. Keep actual checkpoint checksum verification
    and exact-recipe compatibility checks in normal save and restore operations.
    Log effective training controls so a disabled run is visible in hosted logs.
07. Use the same persistence contract everywhere: save complete training state locally
    and atomically, publish a finished snapshot, verify its durable Drive copy, then
    advance `CURRENT.json`. Save periodically, at epoch boundaries, after validation
    and on completion. Stop on persistence failure. Resume compatible state including
    pending validation and skip completed work. Follow the recovery rules below.
08. Test the actual setup cells against an empty temporary filesystem and simulated
    API responses, including download, extraction, snapshot publication
    and restart restoration. Reuse focused regressions in `tests/test_drive_api.py`.
    Schema, syntax and type checks alone do not verify a working hosted setup.
    Report simulated, real API and GPU verification separately; never launch full
    training merely to validate bootstrap or formatting.
09. Keep durable outputs together under
    `My Drive/hm2_retinanet/batches/<batch_id>/<experiment>/`, with local outputs under
    `runs/<experiment_group>/<batch_id>/<experiment>/`. Persist configuration, selection
    manifests, environment, training history, metrics, per-class AP and logs alongside
    checkpoints. Synchronize configured summaries, plots and submission CSVs after
    completion. A teammate uses their own authorized token and Editor access to the
    same Drive folder; credentials stay in Kaggle Secrets, never in notebooks or Git.
10. Confirm completion from saved status, finished final validation and verified final
    checkpoint/result snapshots, including configured exports. Epoch count alone is
    insufficient. Resolve the verified generation through `CURRENT.json`, not folder
    timestamps. Report training time separately from total elapsed time. Conversion
    warnings after a successful run are not evidence of failed training. Distinguish
    conclusions from supplied logs from independently inspected Drive files, and never
    invent metrics or restart a completed run to check its status.

### Automatic checkpoints and recovery

Training entry points must work in a fresh hosted runtime without diagnostic switches
or temporary probe files. Do not insert synthetic GPU or storage probes into the training
path. The controlled-experiment Kaggle notebook currently exports submissions for completed
photometric augmentation, repeat factor sampling and higher input resolution, in that order.
Enable training explicitly for future runs and export after completion on the same Kaggle GPU.
Use one session deadline for the full queue, bind each persistence callback to its own
Drive folder, skip completed training and stop the queue when a run pauses.
Research notebooks keep training disabled. Preserve normal checkpoint integrity checks.

1. Save the complete state: model, optimizer, scheduler, AMP scaler, exact config,
   completed epoch, global optimizer step, image exposures, history and RNG states.
   Preserve the sampler order and consumed-batch cursor where supported. Restore
   the same trainable-parameter policy before loading optimizer state; a fresh
   randomly initialized model can have a different frozen-backbone policy.
2. Save locally first using atomic replacement, then publish an immutable snapshot.
   Checkpoint approximately every 10–15 minutes, at epoch boundaries, after
   validation and on completion. Preserve each experiment's selection protocol:
   RetinaNet keeps `latest.pt` and `best.pt` by highest full-validation COCO mAP;
   the current Faster R-CNN baseline reports the final epoch from `latest.pt` and
   does not select an epoch on validation. Do not add best-epoch selection to it.
3. Synchronize completed files synchronously to Drive. Verify destination hashes,
   write `manifest.json` last in its generation, and publish `CURRENT.json` only
   after verification. Retain the preceding good generation until replacement is
   confirmed. Reuse unchanged files rather than repeatedly uploading model bytes.
4. A persistence failure stops training. Never rely only on `/content`,
   `/kaggle/temp` or `/kaggle/working`, and never copy a checkpoint being written.
   Save metrics, per-class AP, logs, environment and timing with the model state;
   persist comparison tables and plots after each completed experiment.
5. On restart, validate experiment ID, architecture, anchors, dataset-selection hash
   and optimizer configuration before restoring. Resume the existing epoch and
   scheduler budget, finish pending validation, and skip fully completed runs.
   If the current generation is corrupt, recover a verified retained predecessor
   and report the rollback. Never silently return to pretrained initialization.
6. Work since the last durable save may repeat. Record any rollback or runtime
   change, and do not claim bit-for-bit CUDA reproducibility or guaranteed session
   duration. Preserve the specified validation coverage and epoch budget.
7. Verify refactors locally with focused CPU checkpoint/restore tests and notebook
   schema/syntax checks. Do not launch training, overwrite live states or rerun GPU
   notebooks merely to check formatting or documentation.

## Plotting

1. Use Plotly Express for quick visualizations where possible. Only if Plotly Express does not support the required functionality, use Plotly's `graph_objects` API.
2. Use Plotly for charts; do not add another plotting library or custom style without a concrete need. Use Pillow or OpenCV to inspect images and draw bounding-box overlays. Framework-native training plots are acceptable when relevant.
3. Always label axes, legends, and titles. Use clear and descriptive labels. Avoid generic labels such as "x", "y", or "value". Axes, legends and titles should be capitalized and use sentence case. Avoid all caps, camel case, or snake case.

## Writing

1. Write concise, technical explanations without unnecessary repetition.
2. Avoid dash punctuation in prose.
3. Keep mathematical symbols and terminology consistent.
4. Prefer clear descriptions of assumptions, decisions, and results over generic commentary. Follow the assignment reporting rules: the student authors the final report; AI assistance is for code and review, not an AI-written final report.

## Git Conventions

Use Conventional Commits with these prefixes:

`feat:`, `fix:`, `docs:`, `exp:`, `refactor:`, `test:`, `chore:`.

Commit subjects should describe the change. Commit bodies should explain **why the change was made**, including important reasoning and tradeoffs, rather than repeating the diff.

For `exp:` commits, include the measured result, evaluation setup, and location of detailed experiment records. Never invent metrics.

Do not combine unrelated changes into one commit. Do not create commits unless explicitly instructed.

## Artifacts and Research Notes

01. Reserve `.artifacts/` exclusively for AI working notes, temporary investigation outputs, and agent-generated reports. It is not a project data or runtime output directory.
02. Store useful intermediate findings when they help explain decisions, reproduce experiments, or continue work in a later session.
03. Do not reference or depend on `.artifacts/` in project code, notebooks, or runtime configuration. Do not store datasets, prepared data, metadata caches, or image previews there. Store project data and derived metadata under `data/`, such as `data/metadata/` and `data/previews/`, and keep them excluded from Git.
04. Organize artifacts by task or experiment. Use descriptive filenames and avoid unnecessary files.
05. Prefer updating an existing artifact over creating multiple overlapping reports.
06. Do not create documentation or artifacts for trivial changes.
07. Record relevant commands, configurations, seeds, metrics, and conclusions when reporting experimental results.
08. Clearly distinguish measured results, hypotheses, assumptions, and unverified observations.
09. Keep `.artifacts/` excluded from Git. Do not stage or commit its contents.
10. Do not store credentials, tokens, private information, or unnecessary copies of large datasets in artifacts.
11. Important validated findings intended for the project history should be promoted to tracked documentation when explicitly requested.
12. Summarize significant findings and reference relevant artifact paths in the final response.

## Code Changes and Cleanup

01. All substantial code changes must pass the relevant test suite, formatting, linting, and static type checks.
02. Preserve existing functionality unless the task explicitly requires changing it.
03. When replacing an implementation, remove obsolete functions, classes, imports, configuration, and references that are no longer used.
04. Do not leave commented-out implementations, redundant wrappers, or compatibility code that serves no purpose.
05. Do not write large numbers of tests. Keep the test suite small and focused on critical behavior and important regressions. Update affected tests when behavior or interfaces change. Exhaustive coverage of every function, branch, or component is not required. Do not add tests solely to increase coverage or mirror implementation details; reason about correctness before deciding what needs testing.
06. Prefer modifying existing code over introducing parallel implementations of the same behavior.
07. Keep changes minimal, readable, idiomatic, and consistent with project conventions.
08. Do not perform unrelated cleanup outside the task scope.
09. Before removing code, verify that it is no longer referenced by relevant callers, tests, or supported interfaces.
10. Do not suppress type errors, disable tests, or weaken quality checks to make changes pass.
11. If verification fails, investigate and fix issues introduced by the change. Report pre-existing failures separately.
12. Do not describe a task as complete when required checks are failing or have not been executed. State the remaining blockers explicitly.
13. For substantial changes, optionally record implementation decisions, removed legacy code, verification results, and unresolved issues in `.artifacts/`.
