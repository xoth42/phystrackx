"""
Project files for the Qt spike: the setup for one video (trim, axes, ruler, tracked boxes), as
JSON with a .phystrack extension, written next to the video.

Format 2 is this app's own. The Tk app's files are format 1 and refuse version 2 with a message,
so the two apps never read each other's files wrongly. (Sharing files between them is a decision
for the migration; see the TODO.)
"""

import json
import os
import tempfile

FORMAT = "phystrackx-qt"
VERSION = 2
EXTENSION = ".phystrack"
# Where projects go when the video's folder is read-only (as the Tk app's projects/ folder)
PROJECTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")


def default_folder(video_path: str) -> str:
    """Next to the video, so a trial's files stay together; projects/ if that folder is read-only."""
    folder = os.path.dirname(os.path.abspath(video_path))
    if os.access(folder, os.W_OK):
        return folder
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    return PROJECTS_DIR


def new_project_path(video_path: str) -> str:
    """A path for a new project for this video, next to it, that does not overwrite a file."""
    return output_path(video_path, "", EXTENSION)


def output_path(video_path: str, suffix: str, extension: str) -> str:
    """A path next to the video for something saved from it, named after it ("Air Trial 1" +
    suffix + extension), that does not overwrite a file: " 2", " 3"... are added when needed."""
    folder = default_folder(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0] + suffix
    path = os.path.join(folder, stem + extension)
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem} {n}{extension}")
        n += 1
    return path


def save(path: str, state: dict) -> None:
    """Writes the project atomically, so a failed save never leaves a half-written file."""
    data = {"format": FORMAT, "version": VERSION, **state}
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load(path: str) -> dict:
    """
    Reads a project file and checks it.

    Raises:
        ValueError: If the file is not a project of this app's format and version, or names no video.
        OSError: If it cannot be read.
    """
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{os.path.basename(path)} is not a project file ({e}).") from e
    if data.get("format") != FORMAT:
        raise ValueError(f"{os.path.basename(path)} is not a project from this app. Projects "
                         "from the Tk app cannot be opened here.")
    if data.get("version") != VERSION:
        raise ValueError(f"{os.path.basename(path)} is project version {data.get('version')!r}; "
                         f"this app reads version {VERSION}.")
    if "video" not in data:
        raise ValueError(f"{os.path.basename(path)} does not name a video.")
    return data
