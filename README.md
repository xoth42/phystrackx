# PhysTrackX Qt spike

**Experimental. Not the app.** This folder is a throwaway spike to find out how a Qt
(PySide6, `QGraphicsView` with an OpenGL viewport) version of PhysTrackX feels. The real app is the
Tk one in the parent folder; nothing here changes it.

Notes for anyone (person or agent) working in the parent repo:

- `qt/` is its **own git repository** and is listed in the parent's `.git/info/exclude`, so the
  parent repo does not see it. Do not `git add` it from the parent.
- It has its **own virtual environment, outside iCloud**: `~/.venvs/phystrackx-qt` (see Setup).
  The parent's `.venv` is untouched. Do not put a Qt venv in this iCloud folder: iCloud sets the
  macOS "hidden" flag on files in dot-folders, Qt skips hidden plugin files, and the app then
  fails with "no Qt platform plugin could be initialized".
- It **imports the parent's `media` and `core` packages read-only** (video reading and proxy
  encoding). It never writes into the parent: proxies go to `qt/cache/`.

## Setup

```sh
uv venv --python 3.12 ~/.venvs/phystrackx-qt
uv pip install --python ~/.venvs/phystrackx-qt/bin/python -r requirements.txt
```

`opencv-python-headless` is pinned to the parent's OpenCV version: the 5.0 macOS wheel has no
FFmpeg backend, so `media.VideoReader` (which asks for FFmpeg) cannot open videos with it. The
parent's `opencv-python` (not headless) bundles its own Qt and clashes with PySide6.

## Run

```sh
cd qt
~/.venvs/phystrackx-qt/bin/python run.py [optional/video.mp4]
```

## What is in it

A window with a left sidebar of three buttons, the video view, and a seek bar.

- **Import Video**: makes a proxy with the parent's `media.proxyvideo` (cached in `qt/cache/`).
- **Trim**: choose start and end frames; the seek bar is limited to that range. Only a range, not
  a re-encoded clip as in the Tk app.
- **Place Axis**: click to place the origin. Drag a line to move the axes, drag the round handle
  at the end of the x axis to rotate them.
- **View**: pinch or Cmd+scroll to zoom, scroll to pan, double-click the picture to fit it.
- **Seek**: drag the bar, or Left and Right arrows for one frame.

Overlays are children of the picture item, so they are in frame pixels and follow zoom and pan
without any coordinate code (in the Tk app, `ViewTransform` does this by hand).
