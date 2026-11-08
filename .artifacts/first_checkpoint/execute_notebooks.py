import logging
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

for name in sys.argv[1:] or ["00_data.ipynb", "01_eda.ipynb"]:
    path = ROOT / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
    LOGGER.info("Executing %s with a fresh CPU kernel from notebooks/", name)
    client = NotebookClient(
        notebook, timeout=300, kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    LOGGER.info("Executed %s successfully; %.2f MiB", name, path.stat().st_size / 1024**2)
