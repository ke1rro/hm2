# AGENTS.md

## Project Context

This repository is for the UCU AI 2026 object detection homework using VisDrone2019-VID
video frames. Predict bounding boxes and one of eight classes: person, bicycle, car, van,
truck, tricycle, bus, and motor. The main deliverable is a reproducible, notebook-first
comparison of detector families followed by a justified final prediction pipeline.

Use `notebooks/` for the visible research narrative and experiment results. Keep reusable
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

## Notebooks

01. Use Jupyter for notebooks. Store them as `.ipynb` files in `notebooks/`. Use the project's Python 3.14 environment when compatible, or a documented GPU-compatible kernel locally or remotely. Launch a local JupyterLab session from the repository root with `uv run jupyter lab notebooks/` when appropriate.
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
12. Prefer numbered notebooks such as `00_data.ipynb`, `01_eda.ipynb`, detector-family experiments, and final inference/submission. Adapt names to the repository instead of enforcing a rigid notebook count.
13. Keep notebook outputs concise and useful for review. Avoid huge logs, embedded datasets, and unnecessary checkpoint outputs. Make major results understandable without executing expensive training.

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
