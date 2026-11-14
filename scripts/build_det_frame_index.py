import argparse
import io
import json
import logging
import re
import sys
import urllib.request
import zipfile
from collections.abc import Buffer
from pathlib import Path

LOGGER = logging.getLogger(__name__)
ARCHIVE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}.zip"
ARCHIVES = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}
IMAGE_NAME = re.compile(r"/images/(\d{7})_(\d{5})_d_\d+\.jpg$")


class _RemoteFile(io.RawIOBase):
    def __init__(self, url: str) -> None:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self._url = response.geturl()
            self._size = int(response.headers["Content-Length"])
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        self._position = {0: offset, 1: self._position + offset, 2: self._size + offset}[whence]
        return self._position

    def readinto(self, buffer: Buffer, /) -> int:
        view = memoryview(buffer).cast("B")
        if self._position >= self._size:
            return 0
        end = min(self._position + len(view), self._size) - 1
        request = urllib.request.Request(
            self._url, headers={"Range": f"bytes={self._position}-{end}"}
        )
        with urllib.request.urlopen(request) as response:
            data = response.read()
        view[: len(data)] = data
        self._position += len(data)
        return len(data)


def list_archive_images(url: str) -> list[str]:
    """Read the image names of a remote ZIP archive from its central directory only.

    Args:
        url: HTTP URL of a ZIP archive that supports range requests.

    Returns:
        Archive member names.
    """
    remote = io.BufferedReader(_RemoteFile(url), buffer_size=1 << 20)
    with zipfile.ZipFile(remote) as archive:
        return archive.namelist()


def build_frame_index() -> dict:
    """Index VisDrone2019-DET images by source video and frame number for every split.

    DET image names have the form VVVVVVV_FFFFF_d_NNNNNNN.jpg, where VVVVVVV is the raw
    VisDrone video and FFFFF the frame inside it. VID clips use the same video number in
    uav0000VVV_SSSSS_v, so the index locates DET images relative to every VID clip.

    Returns:
        Source URLs and sorted frame numbers keyed by video and DET split.
    """
    videos: dict[str, dict[str, list[int]]] = {}
    sources = {}
    for split, name in ARCHIVES.items():
        url = ARCHIVE_URL.format(name=name)
        names = list_archive_images(url)
        matches = [match for match in map(IMAGE_NAME.search, names) if match]
        LOGGER.info("%s: %d members, %d images", name, len(names), len(matches))
        sources[split] = {"url": url, "images": len(matches)}
        for match in matches:
            videos.setdefault(match.group(1), {}).setdefault(split, []).append(int(match.group(2)))
    return {
        "sources": sources,
        "videos": {
            video: {split: sorted(frames) for split, frames in sorted(splits.items())}
            for video, splits in sorted(videos.items())
        },
    }


def main() -> None:
    """Write the VisDrone-DET frame index used to find validation clips unseen by pretraining."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "config/visdrone_det_frame_index.json"
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    index = build_frame_index()
    arguments.output.write_text(json.dumps(index, indent=1) + "\n")
    LOGGER.info("Indexed %d DET videos into %s", len(index["videos"]), arguments.output)


if __name__ == "__main__":
    main()
