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

A window with a left sidebar of buttons, the video view, and a seek bar.

- **Import Video**: the first frame shows at once (about 0.3 s), read from the source. The proxy
  is made in the background by the parent's `media.proxyvideo` (cached in `qt/cache/`), and
  reading switches to it when it is ready (about 10 s for a 39 s phone clip).
- **Trim**: choose start and end frames; the seek bar is limited to that range. Only a range, not
  a re-encoded clip as in the Tk app.
- **Place Axis**: click to place the origin. Drag a line to move the axes, drag the round handle
  at the end of the x axis to rotate them.
- **Mark Object**: drag a box around each object; Escape when done.
- **Track**: runs the parent's `MLTracker` (NanoTrack, as the Tk app) from the current frame to the
  end of the trim, on a worker thread with its own reader. The view follows live and scrubbing
  stays free. Press again to stop.
- **View**: pinch or Cmd+scroll to zoom, scroll to pan, double-click the picture to fit it.
- **Seek**: drag the bar, or Left and Right arrows for one frame.

Overlays are children of the picture item, so they are in frame pixels and follow zoom and pan
without any coordinate code (in the Tk app, `ViewTransform` does this by hand).

## Tests

```sh
~/.venvs/phystrackx-qt/bin/python tests/test_tracking.py     # PHYSTRACK_QT_SHOW=1 to watch
```

`tests/harness.py` runs Qt with no window (the `offscreen` platform) and a CPU viewport, and
`render_picture()` draws the picture with its overlays at frame pixels. That is how tests check
overlays: Qt cannot capture an OpenGL view on macOS (`grab()` gives black, offscreen gives garbage),
and checking frame pixels does not depend on the window, zoom or screen.

## Findings for the migration

- **OpenCV and Qt do not clash on macOS or Windows.** `opencv-python` only points Qt at its own
  plugins on Linux (see `cv2/config-3.py`); on macOS it uses Cocoa. Switching to
  `opencv-python-headless` is one line in requirements and only matters for Linux. Pin 4.11.x:
  the 5.0 macOS wheel has no FFmpeg backend.
- **The parent's package `__init__` files import Tk.** `experiments/__init__.py` imports `Rigid`
  (customtkinter, the Tk plugins, OCR), so importing the tracker pulls in Tk.
  `phystrack_qt/parent_imports.py` works around it; the fix in the migration is to make those
  `__init__` files import lazily, and to move `Crop`/`Filters` maths out of `gui/plugins`.
- **Keep Qt virtual environments out of iCloud** (see above).
- **Load time is decoding, not encoding.** Decoding the 1080p HEVC phone clip alone is about
  70% of the proxy time, and the hardware encoder is no faster on an Intel Mac. Showing the
  source at once, as here, removes the wait instead.
- **Tracking does not need to block the UI.** A second reader for the worker means the user can
  scrub during a run; the Tk app shares one reader and has to lock scrubbing.
