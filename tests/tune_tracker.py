"""
Tunes the blob tracker's settings (experiments/rigid/blobtracker.py, BlobSettings) against both
tracker benchmarks (test_tracker_scenarios and test_tracker_muddy), with Optuna, while a page in
the browser shows the run live and lets you pause, resume or stop it.

    uv run python -m tests.tune_tracker                  # search until stopped from the page
    uv run python -m tests.tune_tracker --minutes 20     # ...or for 20 minutes at most
    uv run python -m tests.tune_tracker --score          # only score the defaults (or --settings FILE)

The page is at http://127.0.0.1:8765 (opened in the browser at the start; --port, --no-browser).

- Holdout: five scenarios (HOLDOUT) never take part in the search. "Show holdout" on the page
  scores the current best settings, and the defaults, on them, to see whether a gain carries over
  to footage the search did not see.
- Pruning: each trial runs the training scenarios in groups, the hardest first (by the defaults'
  scores), and Optuna's median pruner stops a trial early when its first groups are already worse
  than the median of earlier trials at the same point.
- Two trials run at once, each spreading its scenarios over the worker processes.
- The frames of every scenario are drawn once and kept in temp/tracker_frames (grey, as the
  tracker uses them; redrawn when the drawn footage changes).
- At the end the best settings are saved to temp/tuned_blob_settings.json, and the study to
  temp/tune_study.log (an Optuna journal: --resume continues it).

A trial's score is accuracy + speed - regression (combine()):
- accuracy: the mean over the scenarios of
      the fraction of visible frames on the ball
      - 0.5 if the ball is never found again after a cover, or 0.05 per frame over the recovery limit
      - 0.02 x the median error in ball radii (up to one radius)
      + 0.05 if it meets its scenario's target (the tests' thresholds)
  so 1.05 is perfect;
- speed: 0.04 per doubling of speed against the defaults (negative when slower), within -0.12..+0.08;
- regression: for each scenario that scores more than 0.02 below the defaults, that drop again
  (averaged over the scenarios), so trading a working scenario for another costs extra.

To watch a run started elsewhere (or with --no-page): `uv run python -m tests.tune_tracker --view`
hosts the same page from temp/tune_state.json, and its buttons reach the run through
temp/tune_control.txt. The best settings and the top five trials are saved at every new best.

The benchmarks are synthetic and few: keep changes that make sense and that hold on the holdout,
and check them on real clips.
"""

import argparse
import hashlib
import http.server
import json
import os
import threading
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from tests import harness as h

CACHE = os.path.join(h.REPO, "temp", "tracker_frames")
OUT = os.path.join(h.REPO, "temp", "tuned_blob_settings.json")
JOURNAL = os.path.join(h.REPO, "temp", "tune_study.log")
TOP = os.path.join(h.REPO, "temp", "tuned_top5.json")
STATE = os.path.join(h.REPO, "temp", "tune_state.json")      # Read by --view
CONTROL = os.path.join(h.REPO, "temp", "tune_control.txt")   # Written by --view

# Kept out of the search: one of each kind of difficulty
HOLDOUT = ["glass tube, glare, fog and scratches", "tiny steel ball, filthy lens",
           "falls behind a clamp arm, filthy lens", "paper wall, look-alike ball",
           "camouflage, fine pattern, look-alikes"]

# The search space: name -> ("float", low, high, log) | ("int", low, high) | ("choice", options)
# mass_lo and mass_hi make up mass_range
SPACE = {
    "feature": ("choice", ["contrast", "dog"]), "motion": ("choice", ["acceleration", "velocity"]),
    "surround": ("float", 1.0, 5.0, False), "spread": ("float", 0.2, 1.2, False),
    "patch": ("float", 0.4, 1.0, False), "shift": ("int", 1, 4),
    "template_score": ("float", 0.6, 0.95, False), "strong": ("float", 0.4, 0.85, False),
    "least": ("float", 0.1, 0.6, False), "near_least": ("float", -0.2, 0.4, False),
    "mass_lo": ("float", 0.1, 0.6, False), "mass_hi": ("float", 1.5, 6.0, False),
    "gate": ("float", 2.0, 8.0, False), "w_far": ("float", 0.2, 3.0, True),
    "w_unlike": ("float", 0.3, 5.0, True), "w_mass": ("float", 0.05, 3.0, True),
    "base": ("float", 1.5, 6.0, False), "max_reach": ("float", 5.0, 30.0, False),
    "wide": ("float", 0.8, 3.0, False), "jerk": ("float", 0.05, 2.0, True),
    "start_vel": ("float", 0.3, 4.0, True), "noise": ("float", 0.2, 3.0, True),
    "noise_unlike": ("float", 0.5, 20.0, True), "noise_weak": ("float", 0.3, 4.0, True),
    "learn_above": ("float", 0.5, 0.95, False), "learn_picture": ("float", 0.05, 0.8, False),
    "learn_mass": ("float", 0.05, 0.6, False),
}


# -------------------------------------------------------------------------------------------------
# The frames, drawn once
def _scenarios():
    from tests import test_tracker_muddy as m, test_tracker_scenarios as s
    return [("muddy", i, scn) for i, scn in enumerate(m.build_scenarios())] + \
           [("clean", i, scn) for i, scn in enumerate(s.build_scenarios())]


def _stamp() -> str:
    """A fingerprint of what the scenarios draw (two frames of each, and their paths), so an edit
    to the test files that does not change the footage keeps the cache."""
    from tests.test_tracker_scenarios import N
    digest = hashlib.sha256()
    for kind, i, scn in _scenarios():
        for t in (1, N // 2):
            digest.update(scn.frame(t).tobytes())
            digest.update(np.float64(scn.truth(t)).tobytes())
    return digest.hexdigest()[:16]


def _draw(arg):
    h.setup_env()
    import cv2
    kind, i = arg
    scn = next(scn for k, j, scn in _scenarios() if k == kind and j == i)
    from tests.test_tracker_scenarios import N
    gray = np.stack([cv2.cvtColor(scn.frame(t), cv2.COLOR_BGR2GRAY) for t in range(N)])
    cover = [scn.cover(t) if hasattr(scn, "cover") else 0.0 for t in range(N)]
    base = os.path.join(CACHE, f"{kind}{i:02d}")
    np.save(base + ".npy", gray)
    with open(base + ".json", "w") as f:
        json.dump({"name": scn.name, "kind": kind, "tier": scn.tier, "radius": scn.radius,
                   "truth": [list(map(float, scn.truth(t))) for t in range(N)], "cover": cover,
                   "target": list(scn.target)}, f)
    return base


def ensure_frames(jobs: int) -> list:
    """The cached scenarios' paths (without extension), drawing them when missing or stale."""
    os.makedirs(CACHE, exist_ok=True)
    stamp_file = os.path.join(CACHE, "stamp")
    stamp = _stamp()
    keys = [(kind, i) for kind, i, _ in _scenarios()]
    paths = [os.path.join(CACHE, f"{kind}{i:02d}") for kind, i in keys]
    current = os.path.exists(stamp_file) and open(stamp_file).read() == stamp
    if not current or not all(os.path.exists(p + ".npy") for p in paths):
        print(f"drawing {len(keys)} scenarios into {os.path.relpath(CACHE, h.REPO)} ...", flush=True)
        with ProcessPoolExecutor(jobs) as ex:
            list(ex.map(_draw, keys))
        with open(stamp_file, "w") as f:
            f.write(stamp)
    return paths


def physical_cores() -> int:
    """Cores without hyperthreads: two workers on one core slow each other unevenly, which would
    make the speed reward noisy."""
    try:
        import subprocess
        return max(1, int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True,
                                         text=True, check=True).stdout))
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def _name(path) -> str:
    with open(path + ".json") as f:
        return json.load(f)["name"]


# -------------------------------------------------------------------------------------------------
# One scenario, one setting
def run_one(arg):
    """Tracks one cached scenario with the given settings; returns its scores."""
    path, values = arg
    h.setup_env()
    import cv2
    cv2.setNumThreads(1)  # One thread per worker: the workers already fill the cores
    from core import PixelRect
    from experiments.rigid.blobtracker import BlobSettings, BlobTracker
    from tests.test_tracker_muddy import RECOVERY_LIMIT, analyse
    gray = np.load(path + ".npy", mmap_mode="r")
    with open(path + ".json") as f:
        meta = json.load(f)
    truth = np.array(meta["truth"])
    n = len(truth)
    tracker = BlobTracker(BlobSettings.from_dict(values))
    r = int(round(meta["radius"]))
    x0, y0 = truth[0]
    tracker.initialize_tracker(np.asarray(gray[0]), PixelRect(int(round(x0)) - r, int(round(y0)) - r, 2 * r, 2 * r))
    centre = np.zeros((n, 2))
    centre[0] = truth[0]
    lost = np.zeros(n, bool)
    start = time.process_time()  # This process's CPU time: not inflated by other work on the machine
    for t in range(1, n):
        ok, _ = tracker.update_tracker(np.asarray(gray[t]))
        if ok:
            centre[t] = tracker.centre
        else:
            centre[t], lost[t] = centre[t - 1], True
    ms = 1000 * (time.process_time() - start) / (n - 1)
    error = np.hypot(*(centre - truth).T)
    scn = SimpleNamespace(radius=meta["radius"], cover=lambda t: meta["cover"][t])
    res = analyse(scn, SimpleNamespace(error=error, lost=lost))
    score = res.on_ball - 0.02 * min(res.median / meta["radius"], 1.0)
    limit = meta["target"][3] if len(meta["target"]) > 3 else RECOVERY_LIMIT
    if res.recovery is not None:
        score -= 0.5 if res.recovery == float("inf") else 0.05 * max(0.0, res.recovery - limit)
    med, p95, on_ball = meta["target"][:3]
    meets = (res.median <= med and res.p95 <= p95 and res.on_ball >= on_ball
             and (res.recovery is None or res.recovery <= limit))
    score += 0.05 * meets
    rec = None if res.recovery is None else (-1 if res.recovery == float("inf") else int(res.recovery))
    return {"name": meta["name"], "kind": meta["kind"], "score": score, "on_ball": res.on_ball,
            "median": res.median, "recovery": rec, "meets": bool(meets), "ms": ms}


SPEED_WEIGHT = 0.04                # Score per doubling of speed against the defaults
SPEED_LIMITS = (-0.12, 0.08)       # ...kept within these
REGRESSION_WEIGHT = 1.0            # Extra loss per point a scenario drops below the defaults' score
REGRESSION_TOLERANCE = 0.02        # ...ignoring drops this small (timing and rounding noise)


def combine(rows, reference: dict, reference_ms: float) -> dict:
    """The trial score from per-scenario rows: accuracy (the mean scenario score), plus a reward
    for being faster than the defaults, minus an extra penalty for every scenario that does worse
    than the defaults did (`reference`: scenario name -> the defaults' row)."""
    accuracy = float(np.mean([r["score"] for r in rows]))
    ms = float(np.mean([r["ms"] for r in rows]))
    speed = float(np.clip(SPEED_WEIGHT * np.log2(reference_ms / max(ms, 1e-3)), *SPEED_LIMITS))
    drops = [max(0.0, reference[r["name"]]["score"] - r["score"] - REGRESSION_TOLERANCE)
             for r in rows if r["name"] in reference]
    regression = REGRESSION_WEIGHT * float(np.mean(drops)) if drops else 0.0
    worse = [r["name"] for r in rows if r["name"] in reference
             and reference[r["name"]]["score"] - r["score"] > REGRESSION_TOLERANCE]
    return {"value": accuracy + speed - regression, "accuracy": accuracy, "speed": speed,
            "regression": regression, "ms": ms, "worse": worse,
            "meets": sum(r["meets"] for r in rows)}


def evaluate(pool, paths, values: dict) -> tuple[float, list]:
    rows = list(pool.map(run_one, [(p, values) for p in paths]))
    return float(np.mean([row["score"] for row in rows])), rows


def show(rows, total) -> None:
    for row in rows:
        rec = "" if row["recovery"] is None else (
            "  back never" if row["recovery"] < 0 else f"  back after {row['recovery']}")
        print(f"  {'ok ' if row['meets'] else '   '}{row['kind']:5} {row['name'][:40]:40} on-ball {row['on_ball']:4.0%}"
              f"  median {row['median']:6.1f}{rec:16} {row['ms']:5.1f} ms")
    print(f"  score {total:.4f}  meets target {sum(r['meets'] for r in rows)}/{len(rows)}  "
          f"mean {np.mean([r['ms'] for r in rows]):.1f} ms/frame")


# -------------------------------------------------------------------------------------------------
# Settings <-> Optuna parameters
def suggest(trial) -> dict:
    from experiments.rigid.blobtracker import BlobSettings
    values = BlobSettings().to_dict()
    picked = {}
    for name, spec in SPACE.items():
        if spec[0] == "choice":
            picked[name] = trial.suggest_categorical(name, spec[1])
        elif spec[0] == "int":
            picked[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            picked[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
    values.update({k: v for k, v in picked.items() if k in values})
    values["mass_range"] = (picked["mass_lo"], picked["mass_hi"])
    return values


def as_params(values: dict) -> dict:
    """Settings as Optuna parameters (to queue a known setting as a trial)."""
    params = {k: values[k] for k in SPACE if k in values}
    params["mass_lo"], params["mass_hi"] = values["mass_range"]
    return params


def changes(values: dict) -> dict:
    from experiments.rigid.blobtracker import BlobSettings
    default = BlobSettings().to_dict()
    out = {}
    for k, v in values.items():
        d = default.get(k)
        if isinstance(v, (list, tuple)):
            v, d = list(v), list(d)
        if v != d:
            out[k] = {"from": d, "to": v}
    return out


# -------------------------------------------------------------------------------------------------
class Run:
    """The search's state, shared between the trials, the controls and the page."""

    def __init__(self, pool, train, holdout, groups, reference, reference_ms):
        self.pool, self.train, self.holdout, self.groups = pool, train, holdout, groups
        self.reference, self.reference_ms = reference, reference_ms  # The defaults, per scenario
        self.lock = threading.Lock()
        self.go = threading.Event()
        self.go.set()
        self.stopping = False
        self.finished = False
        self.started = time.time()
        self.trials = []            # {n, state, value, accuracy, speed, regression, ms, meets, step, seconds}
        self.best = None            # The best trial's entry, plus its settings and rows
        self.top = []               # The five best finished trials (with settings and rows)
        self.baseline = None        # combine() of the defaults on the training scenarios, plus rows
        self.show_holdout = False
        self.holdout_results = {}   # trial number (or "default") -> {value, rows}
        self.holdout_busy = False
        self.importance = []        # [(setting, importance)], refreshed every few minutes
        self.study = None

    # The controls
    def control(self, action):
        if action == "pause":
            self.go.clear()
        elif action == "resume":
            self.go.set()
        elif action == "stop":
            self.stopping = True  # The trials see it and end; after_trial then stops the study
            self.go.set()
        elif action == "holdout":
            self.show_holdout = not self.show_holdout
            if self.show_holdout:
                threading.Thread(target=self.score_holdout, daemon=True).start()

    def wait(self):
        """Blocks while paused; True when the run is stopping."""
        while not self.go.wait(0.25):
            pass
        return self.stopping

    # Holdout, on request (display only: it never feeds the search)
    def score_holdout(self):
        if self.holdout_busy:
            return
        self.holdout_busy = True
        try:
            from experiments.rigid.blobtracker import BlobSettings
            if "default" not in self.holdout_results:
                value, rows = evaluate(self.pool, self.holdout, BlobSettings().to_dict())
                self.holdout_results["default"] = {"value": value, "rows": rows}
            while self.show_holdout:
                best = self.best
                if best is None or best["n"] in self.holdout_results:
                    break
                value, rows = evaluate(self.pool, self.holdout, best["values"])
                self.holdout_results[best["n"]] = {"value": value, "rows": rows}
        finally:
            self.holdout_busy = False

    # One trial
    def objective(self, trial):
        import optuna
        if self.wait():
            raise optuna.TrialPruned()
        values = suggest(trial)
        start = time.time()
        rows = []
        for step, group in enumerate(self.groups):
            if self.wait():
                raise optuna.TrialPruned()
            rows += list(self.pool.map(run_one, [(p, values) for p in group]))
            parts = combine(rows, self.reference, self.reference_ms)
            trial.report(parts["value"], step)
            if trial.should_prune():
                self._record(trial.number, "pruned", parts, step + 1, start)
                raise optuna.TrialPruned()
        parts = combine(rows, self.reference, self.reference_ms)
        entry = self._record(trial.number, "complete", parts, len(self.groups), start)
        with self.lock:
            full = dict(entry, values=values, rows=rows, changes=changes(values))
            self.top = sorted(self.top + [full], key=lambda t: -t["value"])[:5]
            if self.best is None or parts["value"] > self.best["value"]:
                self.best = full
                print(f"  trial {trial.number}: new best {parts['value']:.4f} (accuracy "
                      f"{parts['accuracy']:.4f}, speed {parts['speed']:+.3f}, regression "
                      f"-{parts['regression']:.3f}, {parts['ms']:.1f} ms), meets "
                      f"{parts['meets']}/{len(rows)}", flush=True)
                self.save()
                if self.show_holdout:
                    threading.Thread(target=self.score_holdout, daemon=True).start()
        return parts["value"]

    def _record(self, n, state, parts, step, start) -> dict:
        entry = {"n": n, "state": state, "step": step, "seconds": time.time() - start,
                 **{k: parts[k] for k in ("value", "accuracy", "speed", "regression", "ms", "meets", "worse")}}
        with self.lock:
            self.trials.append(entry)
        return entry

    def save(self):
        """The best settings and the top five, written at every new best."""
        with open(OUT, "w") as f:
            json.dump(self.best["values"], f, indent=1)
        with open(TOP, "w") as f:
            json.dump([{k: t[k] for k in ("n", "value", "accuracy", "speed", "regression", "ms",
                                          "meets", "worse", "values")} for t in self.top], f, indent=1)

    def after_trial(self, study, trial):
        """Optuna callback: the study can only be stopped from inside its own loop."""
        if self.stopping:
            study.stop()

    def refresh_importance(self):
        """Which settings matter most so far (Optuna's fANOVA), every two minutes."""
        import optuna
        while not self.finished:
            time.sleep(120)
            try:
                done = [t for t in self.study.get_trials(deepcopy=False)
                        if t.state == optuna.trial.TrialState.COMPLETE]
                if len(done) >= 20:
                    found = optuna.importance.get_param_importances(self.study)
                    self.importance = [(k, float(v)) for k, v in list(found.items())[:12]]
            except Exception as exc:  # Importance is a nicety; never stop the run for it
                print(f"  (importance not computed: {exc})", flush=True)

    # What the page shows
    def state(self) -> dict:
        with self.lock:
            trials = list(self.trials)
            best, top = self.best, list(self.top)
        done = [t for t in trials if t["state"] == "complete"]
        status = ("finished" if self.finished else "stopping" if self.stopping
                  else "paused" if not self.go.is_set() else "running")
        slim = lambda t: {k: v for k, v in t.items() if k not in ("values", "rows")}
        return {
            "status": status, "elapsed": time.time() - self.started, "updated": time.time(),
            "trials": trials, "complete": len(done), "pruned": len(trials) - len(done),
            "groups": [len(g) for g in self.groups], "train_count": len(self.train),
            "weights": {"speed": SPEED_WEIGHT, "regression": REGRESSION_WEIGHT},
            "baseline": None if self.baseline is None else slim(self.baseline),
            "baseline_rows": None if self.baseline is None else self.baseline["rows"],
            "best": None if best is None else dict(slim(best), rows=best["rows"]),
            "top": [slim(t) for t in top],
            "importance": self.importance,
            "holdout_names": [_name(p) for p in self.holdout],
            "show_holdout": self.show_holdout, "holdout_busy": self.holdout_busy,
            "holdout": {str(k): v for k, v in self.holdout_results.items()},
        }

    def mirror(self):
        """Writes the page's state to STATE every two seconds, and applies the actions a separate
        viewer (--view) leaves in CONTROL, so the run can be watched from another terminal."""
        seen = os.path.getmtime(CONTROL) if os.path.exists(CONTROL) else 0.0
        while True:
            try:
                tmp = STATE + ".part"
                with open(tmp, "w") as f:
                    json.dump(self.state(), f)
                os.replace(tmp, STATE)
                if os.path.exists(CONTROL) and os.path.getmtime(CONTROL) > seen:
                    seen = os.path.getmtime(CONTROL)
                    with open(CONTROL) as f:
                        for line in f.read().split():
                            self.control(line)
                    open(CONTROL, "w").close()
                    seen = os.path.getmtime(CONTROL)
            except Exception as exc:
                print(f"  (state file not written: {exc})", flush=True)
            if self.finished:
                break
            time.sleep(2)


def serve(port: int, get_state, on_control):
    """The page, its state and its buttons, on 127.0.0.1 only."""
    page = PAGE.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, kind: str):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(json.dumps(get_state()).encode(), "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path.startswith("/control/"):
                on_control(self.path.rsplit("/", 1)[-1])
            self._send(b"{}", "application/json")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def view(port: int, browser: bool) -> None:
    """Hosts the page for a run in another process: reads STATE, writes actions to CONTROL."""
    def get_state():
        try:
            with open(STATE) as f:
                state = json.load(f)
            if state["status"] != "finished" and time.time() - state.get("updated", 0) > 15:
                state["status"] = "not responding"
            return state
        except (OSError, ValueError):
            return {"status": "waiting for a run", "elapsed": 0, "trials": [], "complete": 0, "pruned": 0,
                    "baseline": None, "best": None, "top": [], "importance": [], "holdout": {},
                    "holdout_names": [], "show_holdout": False}

    def on_control(action):
        with open(CONTROL, "a") as f:
            f.write(action + "\n")

    serve(port, get_state, on_control)
    url = f"http://127.0.0.1:{port}"
    print(f"viewing the tuning run at {url} (Ctrl+C to close; the run carries on)", flush=True)
    if browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


# -------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score", action="store_true", help="only score the settings, no search")
    parser.add_argument("--view", action="store_true", help="host the page for a run started elsewhere")
    parser.add_argument("--settings", help="a JSON file of settings (to score, or to start from)")
    parser.add_argument("--minutes", type=float, help="stop after this long (default: when stopped)")
    parser.add_argument("--jobs", type=int, default=physical_cores(), help="worker processes (default: physical cores)")
    parser.add_argument("--parallel", type=int, default=2, help="trials run at once")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-page", action="store_true", help="no page from the run itself (use --view)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue the study in temp/tune_study.log")
    args = parser.parse_args()
    if args.view:
        view(args.port, not args.no_browser)
        return
    h.setup_env()
    from experiments.rigid.blobtracker import BlobSettings
    start_values = BlobSettings().to_dict()
    if args.settings:
        with open(args.settings) as f:
            start_values = BlobSettings.from_dict(json.load(f)).to_dict()
    paths = ensure_frames(args.jobs)
    holdout = [p for p in paths if _name(p) in HOLDOUT]
    train = [p for p in paths if p not in holdout]
    with ProcessPoolExecutor(args.jobs) as pool:
        if args.score:
            print("training scenarios:")
            show(*reversed(evaluate(pool, train, start_values)))
            print("holdout scenarios:")
            show(*reversed(evaluate(pool, holdout, start_values)))
            return
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # The defaults: the reference for speed and for regressions, and the order of the
        # scenarios (hardest first, so the pruner sees the ones that tell settings apart early).
        # Timed twice and the faster kept, as the first run also warms the workers up
        _, rows = evaluate(pool, train, BlobSettings().to_dict())
        _, again = evaluate(pool, train, BlobSettings().to_dict())
        rows = [a if a["ms"] <= b["ms"] else b for a, b in zip(rows, again)]
        reference = {r["name"]: r for r in rows}
        reference_ms = float(np.mean([r["ms"] for r in rows]))
        order = sorted(range(len(train)), key=lambda i: rows[i]["score"])
        groups = [[train[i] for i in order[k:k + args.jobs]] for k in range(0, len(order), args.jobs)]
        run = Run(pool, train, holdout, groups, reference, reference_ms)
        run.baseline = dict(combine(rows, reference, reference_ms), rows=rows)
        print(f"defaults on the {len(train)} training scenarios: accuracy {run.baseline['accuracy']:.4f}, "
              f"{reference_ms:.1f} ms/frame, meets {run.baseline['meets']}/{len(rows)}", flush=True)
        if not args.resume and os.path.exists(JOURNAL):
            os.remove(JOURNAL)
        study = optuna.create_study(
            study_name="blob-tracker", direction="maximize", load_if_exists=True,
            storage=JournalStorage(JournalFileBackend(JOURNAL)),
            sampler=optuna.samplers.TPESampler(seed=1, multivariate=True, n_startup_trials=12),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
        run.study = study
        if not args.resume:
            study.enqueue_trial(as_params(start_values))
        threading.Thread(target=run.mirror, daemon=True).start()
        threading.Thread(target=run.refresh_importance, daemon=True).start()
        server = None
        if not args.no_page:
            server = serve(args.port, run.state, run.control)
            url = f"http://127.0.0.1:{args.port}"
            print(f"live page: {url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
        try:
            study.optimize(run.objective, n_jobs=args.parallel,
                           timeout=None if args.minutes is None else args.minutes * 60,
                           catch=(Exception,), callbacks=[run.after_trial])
        except KeyboardInterrupt:
            run.stopping = True
        best = run.best
        if best is not None:
            try:
                found = optuna.importance.get_param_importances(study)
                run.importance = [(k, float(v)) for k, v in list(found.items())[:12]]
            except Exception:
                pass
            run.save()
            print(f"\nbest: trial {best['n']}, score {best['value']:.4f} (accuracy {best['accuracy']:.4f}, "
                  f"speed {best['speed']:+.3f}, regression -{best['regression']:.3f}, {best['ms']:.1f} ms/frame; "
                  f"defaults: accuracy {run.baseline['accuracy']:.4f}, {reference_ms:.1f} ms/frame)")
            print(f"saved {os.path.relpath(OUT, h.REPO)} and {os.path.relpath(TOP, h.REPO)}")
            for k, v in best["changes"].items():
                print(f"  {k}: {v['from']} -> {v['to']}")
            print("training scenarios:")
            show(best["rows"], best["accuracy"])
            print("holdout scenarios (best, then defaults):")
            run.holdout_results[best["n"]] = dict(zip(("value", "rows"), evaluate(pool, holdout, best["values"])))
            run.holdout_results["default"] = dict(zip(("value", "rows"), evaluate(pool, holdout, BlobSettings().to_dict())))
            show(run.holdout_results[best["n"]]["rows"], run.holdout_results[best["n"]]["value"])
            show(run.holdout_results["default"]["rows"], run.holdout_results["default"]["value"])
            run.show_holdout = True
        run.finished = True
        with open(STATE, "w") as f:
            json.dump(run.state(), f)
        if server is not None:
            print(f"the page stays up for 10 minutes ({url}); Ctrl+C to close it now")
            try:
                time.sleep(600)
            except KeyboardInterrupt:
                pass
            server.shutdown()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracker Tuning</title>
<style>
/* Layout: a control bar, then the trials plot, then the best settings and per-scenario scores */
:root {
  --bg: #f3f5f2; --surface: #ffffff; --ink: #17211c; --muted: #5a665f; --line: #d5dcd6;
  --accent: #2749c4; --pruned: #9aa49e; --best: #0b7a57; --hold: #c25a00; --warn: #a8480a;
  --good-bg: #dcf1e7; --warn-bg: #fbe6d3;
  --mono: ui-monospace, "SF Mono", Menlo, monospace; --body: system-ui, -apple-system, sans-serif;
}
@media (prefers-color-scheme: dark) { :root {
  --bg: #0f1512; --surface: #171e1a; --ink: #e5ebe7; --muted: #99a59e; --line: #29332e;
  --accent: #86a4ff; --pruned: #59625d; --best: #55d4a5; --hold: #f2a560; --warn: #f2a560;
  --good-bg: #12342a; --warn-bg: #3b2615; color-scheme: dark; } }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 var(--body); margin: 0;
  padding: 20px max(16px, env(safe-area-inset-left, 0px)) 40px; }
.wrap { max-width: 1100px; margin: 0 auto; display: grid; gap: 18px; }
h1 { font-size: 26px; margin: 0; }
.bar { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: center; }
.status { font-family: var(--mono); font-size: 13px; color: var(--muted); }
.status b { color: var(--ink); text-transform: uppercase; letter-spacing: .06em; }
button { font: inherit; font-weight: 600; border: 0; border-radius: 6px; padding: 7px 14px; cursor: pointer;
  background: var(--ink); color: var(--bg); }
button.alt { background: var(--surface); color: var(--ink); border: 1px solid var(--line); }
button.on { background: var(--hold); color: #fff; }
button.danger { background: var(--warn); color: #fff; }
button:disabled { opacity: .45; cursor: default; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-width: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--muted); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
canvas { width: 100%; height: 300px; display: block; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 18px; }
h2 { font-size: 15px; margin: 0 0 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; padding: 5px 6px; border-bottom: 1px solid var(--line); }
td { padding: 5px 6px; border-bottom: 1px solid var(--line); }
td.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.up { color: var(--best); } .down { color: var(--warn); }
.pill { font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 99px; }
.pill.ok { background: var(--good-bg); color: var(--best); } .pill.no { background: var(--warn-bg); color: var(--warn); }
.scroll { overflow-x: auto; }
.note { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
.imp { display: inline-block; height: 8px; border-radius: 4px; background: var(--accent); vertical-align: middle; max-width: 70%; }
</style></head><body>
<div class="wrap">
  <div class="bar">
    <h1>Tracker tuning</h1>
    <span class="status" id="status">connecting…</span>
  </div>
  <div class="bar">
    <button id="pause">Pause</button>
    <button class="alt" id="holdout">Show holdout</button>
    <button class="danger" id="stop">Stop</button>
    <span class="status" id="hint"></span>
  </div>
  <div class="status" id="parts"></div>
  <div class="panel">
    <canvas id="plot" aria-label="score of each trial"></canvas>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>finished trial</span>
      <span><i style="background:var(--pruned)"></i>pruned (score on the scenarios it ran)</span>
      <span><i style="background:var(--best)"></i>best so far</span>
      <span><i style="background:var(--muted)"></i>defaults (dashed)</span>
      <span id="leg-hold" hidden><i style="background:var(--hold)"></i>holdout score of a best (dashed: defaults on holdout)</span>
    </div>
  </div>
  <div class="cols">
    <div class="panel"><h2 id="best-title">Best settings</h2><div class="scroll"><table id="changes"></table></div>
      <p class="note">Only settings that differ from the defaults are listed.</p></div>
    <div class="panel"><h2>Training scenarios: defaults → best</h2><div class="scroll"><table id="train"></table></div></div>
    <div class="panel"><h2>Top five trials</h2><div class="scroll"><table id="top"></table></div>
      <p class="note">Speed is the reward for being faster than the defaults (negative: slower). Regression is the extra penalty for scenarios that do worse than with the defaults.</p></div>
    <div class="panel"><h2>Which settings matter</h2><div class="scroll"><table id="importance"></table></div>
      <p class="note">Optuna's estimate of how much of the score's spread each setting explains.</p></div>
    <div class="panel" id="hold-panel" hidden><h2>Holdout scenarios: defaults → best</h2><div class="scroll"><table id="hold"></table></div>
      <p class="note" id="hold-note"></p></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let S = null, stopArmed = false;
const post = a => fetch('/control/' + a, {method: 'POST'}).then(tick);
$('pause').onclick = () => post(S && S.status === 'paused' ? 'resume' : 'pause');
$('holdout').onclick = () => post('holdout');
$('stop').onclick = () => {
  if (!stopArmed) { stopArmed = true; $('stop').textContent = 'Click again to stop'; setTimeout(() => { stopArmed = false; $('stop').textContent = 'Stop'; }, 3000); return; }
  post('stop');
};
const fmt = (v, d = 3) => v == null ? '–' : v.toFixed(d);
const mins = s => Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
const back = r => r.recovery == null ? '' : r.recovery < 0 ? 'never' : r.recovery;

function rowsTable(el, before, after) {
  let h = '<tr><th>Scenario</th><th class="n">On ball</th><th class="n">Median px</th><th class="n">Back</th><th>Target</th></tr>';
  after.forEach((r, i) => {
    const b = before ? before.find(x => x.name === r.name) : null;
    const d = b ? r.on_ball - b.on_ball : 0;
    h += '<tr><td>' + r.name + '</td><td class="n">' + (b ? Math.round(b.on_ball * 100) + '% → ' : '') +
      '<span class="' + (d > 0.005 ? 'up' : d < -0.005 ? 'down' : '') + '">' + Math.round(r.on_ball * 100) + '%</span></td>' +
      '<td class="n">' + (b ? b.median.toFixed(1) + ' → ' : '') + r.median.toFixed(1) + '</td>' +
      '<td class="n">' + (b && b.recovery != null ? back(b) + ' → ' : '') + back(r) + '</td>' +
      '<td><span class="pill ' + (r.meets ? 'ok' : 'no') + '">' + (r.meets ? 'meets' : 'below') + '</span></td></tr>';
  });
  el.innerHTML = h;
}

function plot() {
  const c = $('plot'), ctx = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 300;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const T = S.trials, L = 46, R = 12, Tp = 10, B = 26, w = W - L - R, hh = H - Tp - B;
  const hold = S.show_holdout ? S.holdout : {};
  const vals = T.map(t => t.value).concat(S.baseline ? [S.baseline.value] : []).concat(Object.values(hold).map(x => x.value));
  if (!vals.length) return;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const done = T.filter(t => t.state === 'complete').map(t => t.value);
  if (done.length > 4) lo = Math.max(lo, Math.min(...done) - 0.05);   // pruned trials can sit far below
  lo = Math.floor(lo * 20) / 20; hi = Math.ceil(hi * 20) / 20 + 0.0001;
  const nmax = Math.max(10, ...T.map(t => t.n + 1));
  const X = n => L + n / nmax * w, Y = v => Tp + hh - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * hh;
  ctx.font = '11px ' + css('--mono'); ctx.fillStyle = css('--muted'); ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
  for (let v = lo; v <= hi + 1e-9; v += (hi - lo) > 0.4 ? 0.1 : 0.05) {
    ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + w, Y(v)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(v.toFixed(2), L - 6, Y(v));
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const step = Math.max(1, Math.ceil(nmax / 10 / 5) * 5);
  for (let n = 0; n <= nmax; n += step) ctx.fillText(n, X(n), Tp + hh + 6);
  const dashed = (v, col) => { if (v == null) return; ctx.setLineDash([5, 4]); ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + w, Y(v)); ctx.stroke(); ctx.setLineDash([]); };
  dashed(S.baseline && S.baseline.value, css('--muted'));
  if (S.show_holdout && hold.default) dashed(hold.default.value, css('--hold'));
  T.forEach(t => {
    ctx.beginPath(); ctx.arc(X(t.n), Y(t.value), t.state === 'complete' ? 3.5 : 2.5, 0, 6.3);
    ctx.fillStyle = t.state === 'complete' ? css('--accent') : css('--pruned'); ctx.fill();
  });
  // Best so far
  const sorted = T.filter(t => t.state === 'complete').sort((a, b) => a.n - b.n);
  let best = -Infinity; ctx.beginPath(); let started = false;
  sorted.forEach(t => { if (t.value > best) { best = t.value; const x = X(t.n), y = Y(best);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); } });
  if (started) { ctx.lineTo(X(nmax), Y(best)); ctx.strokeStyle = css('--best'); ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1; }
  // Holdout scores of the bests
  Object.entries(hold).forEach(([k, v]) => { if (k === 'default') return;
    const x = X(+k), y = Y(v.value); ctx.fillStyle = css('--hold');
    ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y); ctx.closePath(); ctx.fill(); });
  ctx.fillStyle = css('--muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('score per trial: accuracy + speed reward − regression penalty', L + 4, Tp);
}

function render() {
  const st = S.status;
  $('status').innerHTML = '<b>' + st + '</b> · ' + mins(S.elapsed) + ' · ' + S.complete + ' finished, ' + S.pruned +
    ' pruned · ' + (S.elapsed > 0 ? ((S.complete + S.pruned) / S.elapsed * 60).toFixed(1) : '0') + ' trials/min · best ' +
    fmt(S.best && S.best.value) + ' (defaults ' + fmt(S.baseline && S.baseline.value) + ')';
  $('parts').innerHTML = S.best ? parts(S.best) + (S.baseline ? '<span class="status"> · defaults: ' + parts(S.baseline) + '</span>' : '') : '';
  $('top').innerHTML = '<tr><th class="n">Trial</th><th class="n">Score</th><th class="n">Accuracy</th><th class="n">Speed</th><th class="n">Regression</th><th class="n">ms/frame</th><th class="n">Meets</th><th>Worse than defaults</th></tr>' +
    (S.top || []).map(t => '<tr><td class="n">' + t.n + '</td><td class="n">' + fmt(t.value) + '</td><td class="n">' + fmt(t.accuracy) +
      '</td><td class="n ' + (t.speed > 0 ? 'up' : t.speed < 0 ? 'down' : '') + '">' + (t.speed >= 0 ? '+' : '') + fmt(t.speed) +
      '</td><td class="n ' + (t.regression > 0 ? 'down' : '') + '">−' + fmt(t.regression) + '</td><td class="n">' + t.ms.toFixed(1) +
      '</td><td class="n">' + t.meets + '</td><td>' + (t.worse.length ? t.worse.join(', ') : 'none') + '</td></tr>').join('');
  $('importance').innerHTML = S.importance && S.importance.length
    ? S.importance.map(([k, v]) => '<tr><td>' + k + '</td><td class="n" style="width:55%"><span class="imp" style="width:' + Math.round(v * 100) + '%"></span> ' + (v * 100).toFixed(0) + '%</td></tr>').join('')
    : '<tr><td>Computed every two minutes once 20 trials have finished.</td></tr>';
  $('pause').textContent = st === 'paused' ? 'Resume' : 'Pause';
  $('pause').disabled = $('stop').disabled = !['running', 'paused'].includes(st);
  $('holdout').textContent = S.show_holdout ? 'Hide holdout' : 'Show holdout';
  $('holdout').classList.toggle('on', S.show_holdout);
  $('hint').textContent = st === 'paused' ? 'Trials already running finish their current group, then wait.' :
    st === 'stopping' ? 'Finishing the trials in progress…' : st === 'finished' ? 'Done: best settings saved to temp/tuned_blob_settings.json' : '';
  $('leg-hold').hidden = !S.show_holdout;
  if (S.best) {
    $('best-title').textContent = 'Best settings · trial ' + S.best.n + ' · score ' + fmt(S.best.value);
    const ch = Object.entries(S.best.changes);
    $('changes').innerHTML = '<tr><th>Setting</th><th class="n">Default</th><th class="n">Best</th></tr>' +
      (ch.length ? ch.map(([k, v]) => '<tr><td>' + k + '</td><td class="n">' + show(v.from) + '</td><td class="n">' + show(v.to) + '</td></tr>').join('')
        : '<tr><td colspan="3">The defaults are still the best.</td></tr>');
    rowsTable($('train'), S.baseline_rows, S.best.rows);
  }
  $('hold-panel').hidden = !S.show_holdout;
  if (S.show_holdout) {
    const hb = S.best && S.holdout[String(S.best.n)], hd = S.holdout['default'];
    $('hold-note').textContent = (hb && hd) ? 'Holdout score: defaults ' + fmt(hd.value) + ' → best ' + fmt(hb.value) + '. These scenarios never take part in the search.'
      : 'Scoring the holdout scenarios…';
    if (hb) rowsTable($('hold'), hd ? hd.rows : null, hb.rows);
    else $('hold').innerHTML = '<tr><td>' + S.holdout_names.join(', ') + '</td></tr>';
  }
  plot();
}
const parts = t => 'accuracy ' + fmt(t.accuracy) + ', speed ' + (t.speed >= 0 ? '+' : '') + fmt(t.speed) + ', regression −' + fmt(t.regression) + ', ' + t.ms.toFixed(1) + ' ms/frame, meets ' + t.meets;
const show = v => Array.isArray(v) ? v.map(x => typeof x === 'number' ? +x.toFixed(3) : x).join('–') : typeof v === 'number' ? +v.toFixed(3) : v;
function tick() {
  return fetch('/state').then(r => r.json()).then(s => { S = s; render(); })
    .catch(() => { $('status').innerHTML = '<b>tuner not running</b> · the last results stay on the page'; });
}
setInterval(tick, 1000); tick();
window.addEventListener('resize', () => S && plot());
</script></body></html>
"""


if __name__ == "__main__":
    main()
