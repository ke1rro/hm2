import hashlib
import json
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from googleapiclient.discovery import Resource
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from hm2.colab_drive import restore_checkpoint_from_drive, verify_drive_checkpoint
from hm2.experiment_checkpoint import load_checkpoint, publish_snapshot
from hm2.retinanet import make_optimizer

LOGGER = logging.getLogger(__name__)
FILE_FIELDS = "id,name,mimeType,size,md5Checksum,sha256Checksum"


def verify_checksum(path: Path, metadata: dict) -> None:
    """Check downloaded or uploaded bytes against a Drive binary object's checksum.

    Args:
        path: Complete local file to verify.
        metadata: Drive metadata containing SHA256 or MD5 and optionally size.
    """
    algorithm = "sha256" if metadata.get("sha256Checksum") else "md5"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, algorithm).hexdigest()
    if digest != metadata.get(f"{algorithm}Checksum") or (
        "size" in metadata and path.stat().st_size != int(metadata["size"])
    ):
        raise ValueError(f"Drive transfer checksum mismatch: {path.name}")


class DriveApi:
    """Authenticated binary file transport for the existing Drive snapshot layout.

    Args:
        service: Google Drive v3 service using refreshable user OAuth credentials.
    """

    def __init__(self, service: Resource) -> None:
        self.files = getattr(service, "files")()

    def children(self, parent: str) -> dict[str, dict]:
        """List untrashed children, rejecting ambiguous duplicate filenames.

        Args:
            parent: Drive folder identifier.

        Returns:
            Child metadata indexed by its unique filename.
        """
        result = {}
        page_token = None
        while True:
            page = self.files.list(
                q=f"'{parent}' in parents and trashed = false",
                fields=f"nextPageToken,files({FILE_FIELDS})",
                pageSize=1000,
                pageToken=page_token,
            ).execute(num_retries=3)
            for item in page.get("files", []):
                if item["name"] in result:
                    raise ValueError(f"Duplicate Drive name in folder {parent}: {item['name']}")
                result[item["name"]] = item
            page_token = page.get("nextPageToken")
            if not page_token:
                return result

    def folder(self, parent: str, name: str) -> str:
        """Reuse a named folder or create it under the specified parent.

        Args:
            parent: Parent Drive folder identifier.
            name: Unique child folder name.

        Returns:
            Existing or newly created folder identifier.
        """
        children = self.children(parent)
        if name in children:
            if children[name]["mimeType"] != "application/vnd.google-apps.folder":
                raise ValueError(f"Expected a Drive folder: {name}")
            return children[name]["id"]
        return self.files.create(
            body={
                "name": name,
                "parents": [parent],
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        ).execute(num_retries=0)["id"]

    def read_json(self, file_id: str) -> dict:
        """Read a small JSON object directly from Drive.

        Args:
            file_id: Binary JSON file identifier.

        Returns:
            Decoded JSON contents.
        """
        return json.loads(self.files.get_media(fileId=file_id).execute(num_retries=3))

    def upload(self, source: Path, parent: str) -> dict:
        """Upload completed bytes resumably and verify the provider's resulting checksum.

        Args:
            source: Complete local file, never a file being written by training.
            parent: Destination Drive folder identifier.

        Returns:
            Verified remote binary metadata.
        """
        existing = self.children(parent).get(source.name)
        if existing:
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            if existing.get("md5Checksum") == digest:
                return existing
        media = MediaFileUpload(str(source), chunksize=16 * 1024**2, resumable=True)
        if existing:
            request = self.files.update(fileId=existing["id"], media_body=media, fields=FILE_FIELDS)
        else:
            request = self.files.create(
                body={"name": source.name, "parents": [parent]},
                media_body=media,
                fields=FILE_FIELDS,
            )
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=3)
        verify_checksum(source, response)
        return response

    def download(self, file_id: str, destination: Path, sha256: str | None = None) -> None:
        """Download, verify and atomically replace a local file, retaining any good old copy.

        Args:
            file_id: Source Drive binary identifier.
            destination: Local final pathname.
            sha256: Optional expected SHA256 from a committed experiment manifest.
        """
        metadata = self.files.get(fileId=file_id, fields=FILE_FIELDS).execute(num_retries=3)
        if sha256:
            metadata = {**metadata, "sha256Checksum": sha256}
        if destination.is_file():
            with destination.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256" if sha256 else "md5").hexdigest()
            if digest == (sha256 or metadata.get("md5Checksum")):
                return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".incoming")
        LOGGER.info("Download %s (%.1f MiB)", destination.name, int(metadata["size"]) / 1024**2)
        with temporary.open("wb") as stream:
            transfer = MediaIoBaseDownload(
                stream, self.files.get_media(fileId=file_id), chunksize=16 * 1024**2
            )
            has_completed = False
            while not has_completed:
                _, has_completed = transfer.next_chunk(num_retries=3)
        verify_checksum(temporary, metadata)
        temporary.replace(destination)

    def copy(self, file_id: str, parent: str, name: str) -> dict:
        """Copy unchanged bytes on Drive without transferring them through the GPU runtime.

        Args:
            file_id: Existing verified binary identifier.
            parent: New snapshot folder identifier.
            name: Destination filename.

        Returns:
            New binary object's metadata for verification.
        """
        return self.files.copy(
            fileId=file_id, body={"name": name, "parents": [parent]}, fields=FILE_FIELDS
        ).execute(num_retries=0)

    def discard(self, file_id: str) -> None:
        """Delete an expired generated snapshot after its replacement and predecessor are durable.

        Args:
            file_id: Expired generation folder identifier.
        """
        self.files.delete(fileId=file_id).execute(num_retries=3)


def _upload_generation(api: DriveApi, source: Path, parent: str, manifest: dict) -> dict:
    children = api.children(parent)
    previous = api.read_json(children["CURRENT.json"]["id"]) if "CURRENT.json" in children else {}
    if previous == manifest:
        return manifest
    generations = api.folder(parent, "generations")
    folders = api.children(generations)
    previous_files = {}
    if previous and previous["generation"] in folders:
        previous_files = api.children(folders[previous["generation"]]["id"])
    generation = api.folder(generations, manifest["generation"])
    for name, digest in manifest["files"].items():
        target = generation
        for part in Path(name).parts[:-1]:
            target = api.folder(target, part)
        existing = api.children(target).get(Path(name).name)
        if existing:
            api.upload(source / name, target)
        elif name in previous_files and previous["files"].get(name) == digest:
            metadata = api.copy(previous_files[name]["id"], target, Path(name).name)
            verify_checksum(source / name, metadata)
        else:
            api.upload(source / name, target)
    with TemporaryDirectory(prefix="hm2-drive-api-") as temporary_name:
        temporary = Path(temporary_name)
        encoded = json.dumps(manifest, indent=2) + "\n"
        (temporary / "manifest.json").write_text(encoded)
        api.upload(temporary / "manifest.json", generation)
        pointer = api.children(parent).get("CURRENT.json")
        actual = api.read_json(pointer["id"]) if pointer else {}
        if actual != previous:
            raise RuntimeError(
                "Drive checkpoint changed concurrently. Stop the other writer before resuming."
            )
        (temporary / "CURRENT.json").write_text(encoded)
        pointer = api.upload(temporary / "CURRENT.json", parent)
        if api.read_json(pointer["id"]) != manifest:
            raise ValueError("Drive checkpoint pointer failed read-back verification.")
    retained = {manifest["generation"], previous.get("generation")}
    for name, folder in api.children(generations).items():
        if name.isdigit() and name not in retained:
            api.discard(folder["id"])
    LOGGER.info("Verified Drive API snapshot %s in folder %s", manifest["generation"], parent)
    return manifest


def sync_checkpoint_to_drive_api(api: DriveApi, local: Path, folder_id: str) -> dict:
    """Persist an existing runner's completed snapshot using the Colab-compatible Drive layout.

    Args:
        api: Authenticated Drive file transport.
        local: Local experiment output with sync/ready.json.
        folder_id: Existing or new experiment folder on the same Google Drive.

    Returns:
        Committed generation and verified file hashes.
    """
    manifest = json.loads((local / "sync/ready.json").read_text())
    return _upload_generation(api, local / "sync" / manifest["generation"], folder_id, manifest)


def restore_checkpoint_from_drive_api(
    api: DriveApi, folder_id: str, local: Path, config: dict, cache: Path
) -> dict:
    """Download the committed Drive state and reuse existing training-state restoration.

    Try the retained predecessor if the current generation is unreadable. Verify its
    complete manifest before repairing the pointer. Persist any newer compatible local
    progress through the API before a caller can skip a completed experiment.

    Args:
        api: Authenticated Drive file transport.
        folder_id: Experiment folder containing the same CURRENT.json used by Colab.
        local: Writable experiment output directory.
        config: Exact requested recipe checked before training-state restoration.
        cache: Local mirror for checksum verification and the existing restore utility.

    Returns:
        Restored epoch, cursor, counters and history, or an empty mapping for a new run.
    """
    children = api.children(folder_id)
    if "CURRENT.json" not in children:
        return {}
    pointer = api.read_json(children["CURRENT.json"]["id"])
    generations = api.children(children["generations"]["id"])
    candidates = [pointer["generation"]] + sorted(
        (name for name in generations if name.isdigit() and name != pointer["generation"]),
        reverse=True,
    )
    has_restored = False
    for generation in candidates:
        try:
            files = api.children(generations[generation]["id"])
            if "manifest.json" not in files:
                continue
            manifest = api.read_json(files["manifest.json"]["id"])
            if manifest["generation"] != generation:
                raise ValueError("Drive manifest generation does not match its directory.")
            if generation == pointer["generation"] and manifest != pointer:
                raise ValueError("Drive CURRENT.json differs from the committed manifest.")
            directory = cache / "generations" / generation
            for name, digest in manifest["files"].items():
                destination = directory / name
                if not destination.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(
                        "Drive manifest contains a path outside the snapshot directory."
                    )
                api.download(files[name]["id"], destination, digest)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            has_restored = True
            break
        except (OSError, ValueError, KeyError) as error:
            LOGGER.warning("Reject unreadable Drive API generation %s: %s", generation, error)
    if not has_restored:
        raise ValueError("No verified Drive API checkpoint remains.")
    progress = restore_checkpoint_from_drive(cache, local, config)
    if manifest != pointer:
        _upload_generation(api, directory, folder_id, manifest)
        pointer = manifest
        LOGGER.warning(
            "Recovered prior Drive generation %s; later work may repeat.", manifest["generation"]
        )
    with (local / "latest.pt").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != pointer["files"]["latest.pt"]:
        publish_snapshot(local)
        sync_checkpoint_to_drive_api(api, local, folder_id)
        LOGGER.warning(
            "Persisted recovered/local progress; work after its saved cursor may repeat."
        )
    return progress


def sync_results_to_drive_api(api: DriveApi, local: Path, batch_folder_id: str) -> dict:
    """Save summary tables, plots and previews alongside the original Colab batch.

    Args:
        api: Authenticated Drive file transport.
        local: Batch root containing generated comparison outputs.
        batch_folder_id: Existing Drive batch folder identifier.

    Returns:
        Committed summary generation and checksums.
    """
    files = [path for path in local.iterdir() if path.suffix in {".json", ".csv", ".html"}]
    files.extend((local / "previews").glob("*.jpg"))
    manifest = {"generation": str(time.time_ns()), "files": {}}
    for path in files:
        with path.open("rb") as stream:
            manifest["files"][path.relative_to(local).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return _upload_generation(api, local, api.folder(batch_folder_id, "summary"), manifest)


def verify_drive_api_checkpoint(api: DriveApi, local: Path, folder_id: str) -> dict:
    """Exercise checkpoint serialization, API upload, download, integrity and state restoration.

    Args:
        api: Authenticated Drive file transport.
        local: Temporary local directory for a tiny CPU model's checkpoint.
        folder_id: Drive probe folder, separate from all real experiment states.

    Returns:
        Verified remote probe checksum and step.
    """
    verify_drive_checkpoint(local / "write", local / "copy")
    source = local / "write/checkpoint.pt"
    uploaded = api.upload(source, folder_id)
    downloaded = local / "read/checkpoint.pt"
    downloaded.unlink(missing_ok=True)
    api.download(uploaded["id"], downloaded)
    model = torch.nn.Linear(2, 1)
    optimizer, scheduler = make_optimizer(model, 0.01, 4, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = load_checkpoint(
        downloaded, model, optimizer, scheduler, scaler, {"purpose": "drive_checkpoint_probe"}
    )
    original = torch.load(source, weights_only=True)
    if (
        state["step"] != 1
        or not all(
            torch.equal(model.state_dict()[key], value) for key, value in original["model"].items()
        )
        or scheduler.state_dict() != original["scheduler"]
        or not optimizer.state
        or scaler.state_dict() != original["scaler"]
    ):
        raise ValueError("Drive API checkpoint restoration failed.")
    return {"verified": True, "step": state["step"], "file_id": uploaded["id"]}
