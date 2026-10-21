"""
proxyvideo.py

Generate a low-resolution proxy video for fast frame seeking.

Author: Isam Balghari
"""

import hashlib
import logging
import os
import re
import subprocess
from typing import Callable, Optional

from core import find_ffmpeg


# Encoding quality of the proxy. CRF 28 caused visible blocking around sharp colour edges.
PROXY_VERSION = "crf18-v2"
PROXY_CRF = "18"


def _cache_key(videopath: str, width: int, height: int) -> str:
    """Short key that changes when the source file or the proxy size changes."""
    st = os.stat(videopath)
    # PROXY_VERSION changes whenever the encoding settings change, so old proxies are re-made
    raw = f"{PROXY_VERSION}|{os.path.abspath(videopath)}|{st.st_size}|{st.st_mtime_ns}|{width}x{height}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def source_is_plain(videopath: str) -> bool:
    """
    True when the video can be read as it is: even width and height, and no rotation. Such a file
    already holds the pixels a full-resolution copy would have, so the copy need not be encoded.
    Read from ffmpeg's own report, so no ffprobe is needed.
    """
    try:
        ffmpeg = find_ffmpeg()
        report = subprocess.run([ffmpeg, "-hide_banner", "-i", videopath], capture_output=True,
                                text=True).stderr
    except (FileNotFoundError, OSError):
        return False
    if "rotation of" in report or re.search(r"\brotate\s*:", report):
        return False
    match = re.search(r"Stream #\d+:\d+.*Video:.*?,\s*(\d+)x(\d+)", report)
    if not match:
        return False
    width, height = int(match.group(1)), int(match.group(2))
    return width % 2 == 0 and height % 2 == 0


def proxyvideo(videopath: str, width: int = 1000, height: int = -2,
               writepath: Optional[str] = None, overwrite: bool = False,
               on_progress: Optional[Callable[[float], None]] = None,
               total_frames: Optional[int] = None) -> Optional[str]:
    """
    Create a low-resolution proxy video using ffmpeg.
    Automatically bakes in rotation metadata and outputs as a standard .mp4.

    Args:
        videopath: Path to the original high-res video.
        width: Target width for the proxy video (e.g., 1000).
        height: Target height (e.g., -2 preserves aspect ratio).
        writepath: Optional output path. If None, auto-generated from the source
            file and the target size.
        overwrite: Whether to overwrite an existing proxy video.
        on_progress: Optional callback receiving the encoded fraction (0.0 to 1.0).
        total_frames: Frame count of the source, needed for on_progress.

    Returns:
        Path to the generated proxy video or None if failed.
    """
    logger = logging.getLogger(__name__)

    if not os.path.isfile(videopath):
        logger.error(f"Input video not found: {videopath}")
        return None

    # Force the proxy output to be an .mp4 container.
    # This destroys the proprietary .MOV metadata and bakes the rotation
    # directly into the physical pixels for OpenCV.
    if writepath is None:
        base, _ = os.path.splitext(os.path.basename(videopath))
        key = _cache_key(videopath, width, height)
        writepath = os.path.join("./temp", f"{base}_{key}_proxy.mp4")
    else:
        # Ensure even a user-provided writepath ends in .mp4
        base, _ = os.path.splitext(writepath)
        writepath = f"{base}.mp4"

    if os.path.exists(writepath) and not overwrite:
        logger.info(f"Proxy already exists: {writepath}")
        if on_progress:
            on_progress(1.0)
        return writepath

    os.makedirs(os.path.dirname(writepath) or ".", exist_ok=True)

    # Encode to a partial file and rename on success, so a cancelled or failed
    # encode is never picked up later as a valid cached proxy.
    partial = f"{os.path.splitext(writepath)[0]}.partial.mp4"

    try:
        ffmpeg = find_ffmpeg()
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    resolution = f"{width}:{height}"

    # By default, FFmpeg applies -autorotate before video filters.
    # So it will rotate the portrait video upright, THEN apply the scale filter.
    # -progress pipe:1 writes machine-readable "frame=N" lines to stdout for the progress bar.
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats",
        "-i", videopath,
        "-vf", f"scale={resolution}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", PROXY_CRF,
        "-an", # Strips audio to speed up proxy generation
        partial
    ]

    logger.info(f"Creating proxy video: {writepath}")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in proc.stdout:
        if on_progress and total_frames and line.startswith("frame="):
            try:
                done = int(line.split("=", 1)[1])
            except ValueError:
                continue
            on_progress(min(done / total_frames, 1.0))

    stderr = proc.stderr.read()
    returncode = proc.wait()

    if returncode != 0:
        logger.error(f"FFmpeg failed with exit code {returncode}")
        # This will log the actual error text FFmpeg prints if it crashes
        logger.error(f"FFmpeg Error Output:\n{stderr}")
        if os.path.exists(partial):
            os.remove(partial)
        return None

    os.replace(partial, writepath)
    if on_progress:
        on_progress(1.0)
    logger.info(f"Proxy created: {writepath}")
    return writepath
