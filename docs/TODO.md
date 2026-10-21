# TODO

## Last done (newest first)
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
