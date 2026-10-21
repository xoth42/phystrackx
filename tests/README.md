# PhysTrackX tests

Run everything from the repo root:

    uv run python -m tests.run_all

Run one module:

    uv run python -m tests.run_all test_rect_editing

## What is covered

| Module | Needs a window | Covers |
|---|---|---|
| `test_view_math` | no | zoom/pan transform, round trips, window bounds, zoom keeps the point under the pointer, pan follows the fingers |
| `test_project_io` | no | project file write/read, version and missing-video errors |
| `test_cli` | no | `--help`, `--video`/`--project` validation, `--download-models` present |
| `test_models` | no | model file detection; no download when files exist |
| `test_rect_editing` | yes | draw at zoom, circle shape, move, resize handle, tiny-click discard |
| `test_roundtrip` | yes (2 processes) | save a full setup, reopen it, compare every part |
| `test_trim_crop_zoom` | yes | trim crop size, zoom strip for axes and scale, strip hides on apply |
| `test_fit` | no | damped oscillator fit recovers known parameters; peak spacing and fit period agree; a range uses only its own data; the plot slider updates the fit |
| `test_crop_fit` | yes | applied crop fills the view at its ratio; overlays stay in frame pixels; clearing returns to 1:1 |
| `test_saved_data` | no | saved CSV round trip; save dialog starts in the saved folder; project keeps the data entry; plot from the reloaded CSV |
| `test_axis_geometry` | no | axis line keeps its angle at every zoom; endpoint lies on the canvas edge |
| `test_units` | yes (ruler only) | typed "50 cm" parses to value and unit; the ruler shows, saves and restores the unit; plot and fit labels carry the unit |
| `test_assist` | no | pixel-change gate quiet for a lone ball, fires for a hand beside it; hard engine cooldown; a run with the hard engine keeps points on the ball; real Segment Anything picks the ball, not the hand |
| `test_crash` | no (one check uses a window) | worker-thread exceptions reach the crash log with traceback and versions; a C-level crash still prints the Python stack; garbage collection runs only on the UI thread |
| `test_mode_switch` | yes | starting the axis or the ruler cancels an unfinished one of the other |
| `test_navigation` | yes | trackpad and wheel pan in both directions, Cmd +/- zoom around the pointer, Cmd 0 back to the whole picture, a scroll burst draws once, axes drag by the origin and keep their angle, window title shows project and video |
| `test_trim_again` | yes | a second trim (with crop) is recorded in full-video frames; a failed trim keeps the current clip |
| `test_tracking` | yes | live frames, refused second run, stop keeps results, manual positions |
| `test_tracker_accuracy` | no | tracker follows a synthetic ball with known path; pipeline keeps points on the ball at native size, when downscaled, and when tracking the full-resolution copy at a downscaled view; full-size region maps the crop |

## Notes

- The test clip is synthetic (`testsrc2`, 12 s, 540x960), made with ffmpeg into the system
  temp folder, so the tests do not depend on personal videos.
- Tests that open the app run in their own process. Put a new app test in its own module.
- Message boxes are recorded in `harness.DIALOGS` instead of shown.
- Network tests (downloading the models) only run with `PHYSTRACK_TEST_NETWORK=1`.
- Tcl/Tk: on macOS, `harness.setup_env()` points Tcl/Tk at the uv-managed Python if the venv
  cannot find them.

## Adding a test

1. Add a function `test_<something>()` to the module that covers the area.
2. Use `check(name, condition, detail)` for each expectation.
3. Add it to the module's `if __name__ == "__main__":` list.
4. For a new area, copy an existing `test_*.py` and add a row to the table above.
