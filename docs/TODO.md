# TODO

## Last done (newest first)
- **Zoom and pan broke overlay positions and scales (fixed).** The zoomed picture was drawn from
  a whole-pixel crop stretched to a fixed size, while ruler, axes, boxes and tracked points use the
  exact transform. Rounding the crop could drop a pixel and change the scale by about 2%, so
  overlays drifted by up to a dozen pixels at high zoom. Enlarged pictures are now drawn through an
  exact sub-pixel map (`Video._place`), for both the proxy and the full-resolution copy.
  `tests/test_overlay_geometry.py` checks it to 1 px across zoom, pan, window size and frame
  offset (and was confirmed to fail when the drawing is shifted); `test_overlay_alignment.py`
  checks a real rotated clip.
- **Tracker benchmark** (`tests/test_tracker_scenarios.py`): ten scenarios from a white screen to
  camouflage with look-alikes and occluders, with the tracker given the circle's bounding square.
  `tests/benchmark_report.py` builds a page with a player and error chart per scenario.
- **Too zoomed on start (fixed).** The window resize redrew the picture while the proxy was still
  encoding, when the reader still pointed at the source file. That cached a source-size frame,
  which stayed after the reader switched to the proxy, so the picture was drawn at the wrong size.
  A resize is now laid out only after loading ends, and the frame cache is cleared when a load
  completes. `tests/test_start_view.py` loads a rotated phone-style clip, waits for the first frame,
  and checks the drawn picture against the video area.
- **Tests keep out of Open Recent:** the harness points the recent list at the run folder, and a
  check guards it. The test entries that had reached `~/.phystrackx/recent.json` were removed.
- **Auto Ruler is fast now (about 0.03 s, was 8 s with Segment Anything).** It finds the bar by
  colour (Lab a and b) against the wall beside it, and no longer loads Segment Anything. On the
  Air Trial 1 frame the ends land within about 2 px of the Segment Anything result. Segment
  Anything stays in the tracker's assist and the batch plan.
- **File menu** in the menu bar: Open Video, Open Project (Cmd+O), Open Recent (the last 8
  projects, kept in `~/.phystrackx/recent.json`), Save Project (Cmd+S). Opening another file asks
  whether to save the current project first.
- **Clear Data** in the track strip: during a run it stops the run and discards the results when
  it ends; otherwise it discards them at once. The boxes and the video stay, so tracking starts
  again from the beginning.
- **Window resize:** the video area follows the window. The picture is fitted to it, and the
  ruler and axes keep their place on the picture (they are stored relative to the picture now).
  Track rectangles were already relative to the picture.
- **Dark text** on the cyan strips (zoom, X, Y, rotation, crop and track status, the two check
  boxes).
- **`--video` with a project file** opens that project.
- **Startup crash fixed:** a failed video load raised NameError (the error variable was read
  after its except block). The error is now kept in its own variable.
- **Tests run in parallel** (four at a time, each with its own run folder), about a minute of wall
  time for the whole suite. `core.abspath` resolves resources from the project folder, so the
  working directory no longer matters.
- **Plots:**
  - Fit results and the legend sit in a panel beside the plot, so nothing covers the data.
  - Plots without a fit have a legend that is placed automatically and can be dragged.
  - Every plot has a **Save plot** button. The dialog starts in the project's folder, with a name
    from the video and the plot, and the buttons and sliders are left out of the image.
- **New projects** are created next to the video (or in `projects/` if that folder is read-only).
- **Axis tool:** drag the axes by the origin or either line to move them, and the rotation bar
  stays as before. The axis items are moved instead of recreated, which removes the lag. Opening the
  tool on placed axes allows dragging at once, and Escape (or starting another tool) puts them back.
- **Cmd 0** (Ctrl 0 on other systems) returns to the whole picture, like Reset View.
- **Crash on Track (fixed):**
  - The tracking thread read Tk variables (the filter choice) on every frame. It now gets a
    snapshot taken on the UI thread.
  - Python's garbage collector, running on the tracking thread, deleted Tk fonts there, and both
    of these crashed Tk at random (segmentation fault or bus error). Collection now runs only on the
    UI thread (`core/crash.py`).
- **Global error handling** (`core/crash.py`):
  - Exceptions on any thread and inside Tk callbacks are logged with their traceback, versions,
    and the open project, to the terminal, the session log, and `logs/crash.log`.
  - Crashes in C code print every thread's Python stack (faulthandler).
- **Two-engine tracking** (the "Assist (Segment Anything, slow)" checkbox, off by default): the
  fast tracker runs on every frame. Segment Anything re-finds the object only on hard frames, when
  pixels change in a ring around the ball (a hand moving beside it) or the tracker's confidence
  drops, with a cooldown between uses (`experiments/rigid/assist.py`).
- **Auto Ruler** uses Segment Anything when the model is installed. It runs on a worker thread, and
  the button shows "Finding..." while it works.
- **Speed:**
  - Several objects are tracked in parallel threads (3 objects: 27 → 17 ms per frame).
  - Pan and zoom: decoded frames are cached, and a burst of scroll events draws once.
  - The full-resolution copy is skipped for unrotated sources (the file is read directly).
    Rotated phone videos still get the copy (tested).
- Initial drop metric on the time plots, and a legend on every plot placed automatically.
- Quadratic drag models in the fit menu, and Undo for Auto Ruler.
- Zoomed views (trimmed clips too) read the full-resolution copy.
- Starting any tool cancels the one before it.

## Open
- **Tracker weaknesses found by the benchmark:** a small, fast ball (the dropped ball, up to about
  18 px per frame) is lost after the first bounce (first miss at frame 39), and a fast ball on
  camouflage is lost when large shapes pass in front of it (frame 109). Both are marked
  `known_weak` in `tests/test_tracker_scenarios.py` with a floor. Try the Segment Anything assist
  on them, and a search for the ball near its last position when the tracker's score drops.
- **Half-pixel convention:** tracked points are plotted at pixel indices, but the picture treats a
  pixel as spanning its whole cell, so points sit half a frame pixel up and left of the cell's
  centre (about 3 canvas px at 6x zoom). Decide on one convention.
- **Window resize, not yet covered:** the seek bar keeps its starting width, so it does not
  stretch with the window, and geometry shapes (triangle, line) are not moved with the picture.
- **Auto Ruler on other footage:** tuned on one frame of Air Trial 1 (colour distance 8, from the
  wall's a and b). Check it on the other trials, and on a ruler that is paler than this one.
- **Batch trials** (large): see [BATCH_TRIALS_PLAN.md](BATCH_TRIALS_PLAN.md). It covers a batch
  runner that reuses setup between trials, automatic trim, an experiment reference (labelled ruler,
  axes, and ball patches found again by template matching plus Segment Anything), a review grid, and
  per-trial folders with summary tables. It needs answers to the questions at the end of the plan.
- **Sub-pixel tracking:** `MLTracker.update_tracker` rounds the tracked box to whole pixels
  (`int(...)`), which throws away precision, most of all at proxy size. Keep the floats.
- **Assist on real footage:** on the synthetic clip the fast tracker stays on the ball without
  help, so the test shows the mechanism works, not that it rescues a failing run. Tune the ring
  threshold (`AssistSettings.ring_change`, 4.0) on a real hand-release video.
- **Window redraw cost:** about 55 ms per window update on this Mac, even for small changes. That
  is Tk's own drawing, so pan and zoom can't go faster than that. Coalescing keeps them from
  falling behind.
- **Check on real data:** the zoom quality, the initial drop, the legend placement, and the crash
  fix, on the user's own project.

## Notes
- Worker threads (load, trim, track, Auto Ruler) must never touch Tk: no widgets and no Tk
  variables. Read what they need on the UI thread and pass plain values.
- The plot summary (`gui/components/plot/plot.py`, `_summary`) lists parameters per model, so new
  models need their own lines there.
