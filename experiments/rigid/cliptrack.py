"""
cliptrack.py

Tracks one object through a whole clip, with any of the engines, behind one call:

    result = track_frames(frames, rect, engine="blob", settings={...})

`frames` is any sequence of grey or BGR images (a list, an array, a memory-mapped file), `rect` the
object's box on frame 0. The result has a centre for every frame, and a status that says where
it came from: measured, filled in (the object was hidden and its position comes from the frames
around), or lost.

Engines:
- "blob": the blob tracker (blobtracker.py), frame by frame, as it runs live. Given the clip, it
  also gets the clip's still background (background.py) when its settings use one.

No GUI here. Author: Isam Balghari
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from core import PixelRect

LOST, MEASURED, FILLED = 0, 1, 2


@dataclass
class ClipTrack:
    centre: np.ndarray       # (n, 2) sub-pixel centre per frame (the last known one where lost)
    status: np.ndarray       # (n,) LOST, MEASURED or FILLED
    score: np.ndarray        # (n,) how sure the engine is (0..1; 0 where lost)
    ms_per_frame: float      # The engine's CPU time per frame, everything included
    stopped_at: int | None = None  # The frame a time budget stopped it at (the rest counts as lost)

    @property
    def lost(self) -> np.ndarray:
        return self.status == LOST


def track_frames(frames, rect: PixelRect, engine: str = "blob", settings: dict | None = None,
                 budget: float | None = None, outline=None) -> ClipTrack:
    """Tracks the object selected by `rect` on frame 0 through `frames`. With a `budget` (ms per
    frame), a run that is slower than that on average after 10 frames stops there. With the
    setting "physics" true, the track is refined by the ball's motion (physics.refine) at the end.
    `outline` is the object's outline on frame 0 when the selection gives one (a polygon of (x, y)
    points or a boolean mask): its centroid is the centre and its area the size, and `rect` may
    then be None."""
    values = settings or {}
    if outline is not None:
        from experiments.rigid.blobtracker import outline_circle
        (cx, cy), rad = outline_circle(outline)
        k = int(round(rad))
        rect = PixelRect(int(round(cx)) - k, int(round(cy)) - k, 2 * k, 2 * k)
    sel = (rect, outline)     # The selection, as the engines take it
    if engine == "blob":
        out = _track_blob(frames, sel, values, budget)
    elif engine == "offline":
        out = _track_offline(frames, sel, values, budget)
    else:
        raise ValueError(f"unknown tracking engine: {engine}")
    if values.get("physics"):
        from experiments.rigid.physics import PhysicsSettings, refine
        start = time.process_time()
        out.centre, out.status = refine(out.centre, out.status, _radius(sel),
                                        PhysicsSettings.from_dict(values))
        out.ms_per_frame += 1000 * (time.process_time() - start) / max(1, len(frames) - 1)
    return out


def _track_blob(frames, sel, values, budget) -> ClipTrack:
    """The blob tracker frame by frame. With a background, a first track that shows the ball
    resting in one place long enough to be in the background (a ghost there would hide it) is
    followed by a second, on the background taken again without the ball."""
    from experiments.rigid.blobtracker import BlobSettings
    s = BlobSettings.from_dict(values)
    start = time.process_time()  # This process's CPU time: not inflated by other work on the machine
    if s.background_mix <= 0:
        return _blob_pass(frames, sel, s, None, budget, start)
    first, background, again = _background_passes(frames, sel, s, budget, start)
    if not again:
        return first
    return _blob_pass(frames, sel, s, background, budget, start)


def _background_passes(frames, sel, s, budget, start):
    """The clip's background, and a first blob-tracker pass on it: (first pass, background, whether
    the background was taken again without the ball's resting place)."""
    from experiments.rigid.background import estimate_background, rested, sample_frames
    background = estimate_background(frames)
    first = _blob_pass(frames, sel, s, background, budget, start)
    if background is None or first.stopped_at is not None:
        return first, background, False
    r = _radius(sel)
    picks = sample_frames(len(frames))
    points = np.where(first.lost[picks][:, None], np.nan, first.centre[picks])
    if not rested(points, r, s.ghost_share):
        return first, background, False
    exclude = {int(i): (*first.centre[i], 2 * r) for i in picks if not first.lost[i]}
    return first, estimate_background(frames, exclude=exclude), True


def _blob_pass(frames, sel, s, background, budget, start) -> ClipTrack:
    from experiments.rigid.blobtracker import BlobTracker
    n = len(frames)
    centre, status, score = np.zeros((n, 2)), np.zeros(n, int), np.zeros(n)
    tracker = BlobTracker(s)
    tracker.background = background
    tracker.initialize_tracker(np.asarray(frames[0]), *sel)
    centre[0], status[0], score[0] = tracker.centre, MEASURED, 1.0
    stopped = None
    for t in range(1, n):
        ok, _ = tracker.update_tracker(np.asarray(frames[t]))
        if ok:
            centre[t], status[t], score[t] = tracker.centre, MEASURED, tracker.score()
        else:
            centre[t] = centre[t - 1]
        if budget and t >= 10 and 1000 * (time.process_time() - start) / t > budget:
            stopped = t
            centre[t + 1:] = centre[t]
            break
    ms = 1000 * (time.process_time() - start) / (stopped or n - 1)
    return ClipTrack(centre, status, score, ms, stopped)


# -------------------------------------------------------------------------------------------------
# Offline: candidates in every frame, then the best path through all of them
@dataclass
class LinkSettings:
    # Candidates (lengths in ball radii)
    candidates: int = 12          # Strongest peaks of the whole frame kept per frame,
    floor: float = 4.0            # ...if they have this many times the mass of a typical patch (noise)
    scale_radius: float = 4.0     # Whole frames are searched shrunk so the ball is this many pixels
    online: bool = True           # The blob tracker's own sighting is a candidate too,
    online_bonus: float = 0.0     # ...cheaper by this much
    online_ref: bool = True       # Likeness also against the online tracker's learned picture
    # The path
    max_gap: int = 45             # Longest stretch without the ball that a path may jump (frames)
    miss: float = 1.2             # Cost of a frame without the ball
    w_motion: float = 1.0         # Cost of a step: w_motion * (distance from the path's prediction / tolerance)^2
    tol: float = 0.5              # Tolerance of a one-frame step (ball radii),
    tol_gap: float = 0.3          # plus this per frame jumped,
    tol_gap2: float = 0.02        # plus this per frame jumped squared (the motion may have changed),
    tol_speed: float = 0.2        # plus this fraction of the path's speed (px)
    follow_v: float = 0.6         # How much of a step's surprise goes into the path's velocity,
    follow_a: float = 0.15        # and into its acceleration
    max_step: float = 12.0        # Steps costlier than this are not taken
    # A kick: the motion changed at once (a bounce, a push, the release). Such a step costs `kick`
    # plus w_motion * (distance from the last position / reach)^2, the reach being the tolerance
    # plus the path's speed and kick_reach ball radii per frame; the velocity starts again from the
    # step, and the acceleration (gravity) carries on
    kick: float = 2.5
    kick_reach: float = 2.0

    @classmethod
    def from_dict(cls, values: dict) -> "LinkSettings":
        from dataclasses import fields
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


def _shrink(image, scale):
    return image if image is None or scale >= 1 else cv2.resize(image, None, fx=scale, fy=scale,
                                                                interpolation=cv2.INTER_AREA)


class _Frame:
    """A frame's feature image, shrunk by `scale` (so whole frames are cheap to search).
    `background` is the clip's background already shrunk by `scale`."""

    def __init__(self, gray, background, r, s, scale):
        from experiments.rigid.blobtracker import feature_image
        gray = _shrink(gray, scale)
        self.scale = scale
        self.image = feature_image(gray, r * scale, s, background)
        self.k = k = max(1, int(round(r * scale)))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        self.disc = (xx * xx + yy * yy <= k * k).astype(np.float32)
        self.xx, self.yy = xx, yy

    def mass_at(self, p) -> float:
        """The feature summed over the ball's disc at frame position p."""
        k, img = self.k, self.image
        x, y = (int(round((v + 0.5) * self.scale - 0.5)) for v in p)
        if x < k or y < k or x >= img.shape[1] - k or y >= img.shape[0] - k:
            return 0.0
        return float((img[y - k:y + k + 1, x - k:x + k + 1] * self.disc).sum())

    def peaks(self, count, floor):
        """The `count` strongest ball-sized peaks with at least `floor` times the mass of a typical
        patch of the image (noise): rows of (x, y, mass) in frame pixels."""
        image, k = self.image, self.k
        size = 2 * k + 1
        least = floor * float(np.median(image)) * float(self.disc.sum())
        peak = (image >= cv2.dilate(image, np.ones((size, size), np.uint8))) & (image > 0)
        peak[:k], peak[-k:], peak[:, :k], peak[:, -k:] = False, False, False, False
        ys, xs = np.nonzero(peak)
        if not len(xs):
            return np.zeros((0, 3))
        order = np.argsort(-image[ys, xs])[:count]
        out = []
        for x, y in zip(xs[order].tolist(), ys[order].tolist()):
            part = image[y - k:y + k + 1, x - k:x + k + 1] * self.disc
            mass = float(part.sum())
            if mass <= least or mass <= 0:
                continue
            out.append((x + float((part * self.xx).sum()) / mass, y + float((part * self.yy).sum()) / mass, mass))
        out = np.array(out).reshape(-1, 3)
        out[:, :2] = (out[:, :2] + 0.5) / self.scale - 0.5
        return out


def _track_offline(frames, sel, values, budget) -> ClipTrack:
    """Every frame gives a few candidates: the strongest ball-like peaks of the whole frame, and
    the online blob tracker's sighting. A candidate costs more the less it looks like the ball
    (its picture against the ball's on frame 0 and against the online tracker's learned picture)
    and the more its blob mass differs. The chosen path starts at the selection and runs through
    one candidate per frame, or jumps frames where the ball is hidden at a cost per frame; a step
    costs more the farther it lands from where the path's own motion (position, velocity,
    acceleration, followed along the path) puts the ball. Dynamic programming finds the cheapest
    path over the whole clip (each candidate keeps its cheapest way in, as in the Viterbi
    algorithm), so a cover or a look-alike is judged by what comes before and after it. Jumped
    frames are filled in between the path's two sides.

    With a background: when the path has the ball resting in one place long enough to be in the
    background, the background is taken again without it and the clip tracked again."""
    from experiments.rigid.background import estimate_background, rested, sample_frames
    from experiments.rigid.blobtracker import BlobSettings
    s, L = BlobSettings.from_dict(values), LinkSettings.from_dict(values)
    start = time.process_time()
    if s.background_mix <= 0:
        return _offline_pass(frames, sel, s, L, None, budget, start)
    background = estimate_background(frames)
    out = _offline_pass(frames, sel, s, L, background, budget, start)
    if background is None or out.stopped_at is not None:
        return out
    r = _radius(sel)
    picks = sample_frames(len(frames))
    seen = out.status[picks] == MEASURED
    if not rested(np.where(seen[:, None], out.centre[picks], np.nan), r, s.ghost_share):
        return out
    exclude = {int(i): (*out.centre[i], 2 * r) for i, ok in zip(picks, seen) if ok}
    return _offline_pass(frames, sel, s, L, estimate_background(frames, exclude=exclude), budget, start)


def _offline_pass(frames, sel, s, L, background, budget, start) -> ClipTrack:
    from experiments.rigid.blobtracker import BlobTracker
    n = len(frames)
    online = BlobTracker(s)
    online.background = background
    first = np.asarray(frames[0])
    online.initialize_tracker(first, *sel)
    r = online.r
    scale = min(1.0, L.scale_radius / r)
    small_bg = _shrink(background, scale)
    gray0 = online._gray(first)
    ref0 = online.ref.copy() if online.ref is not None else None
    # The ball's blob mass at the scale searched, at the selection
    begin = np.array(online.centre) - online.offset   # Candidates are blob centres; the offset comes last
    mass0 = _Frame(gray0, small_bg, r, s, scale).mass_at(begin) or None

    def look(gray, p):
        """How much the picture at p looks like the ball: the better of frame 0's and the online
        tracker's learned picture."""
        m = online._match(gray, p) if L.online_ref or ref0 is None else -1.0
        if ref0 is not None:
            current, online.ref = online.ref, ref0
            m = max(m, online._match(gray, p))
            online.ref = current
        return m

    # Per frame: candidate positions, their own cost, their likeness
    P, U, M, O = [begin[None, :]], [np.zeros(1)], [np.ones(1)], [np.ones(1, bool)]
    stopped = None
    for t in range(1, n):
        frame = np.asarray(frames[t])
        gray = online._gray(frame)
        image = _Frame(gray, small_bg, r, s, scale)
        found = image.peaks(L.candidates, L.floor)
        pos, mass, bonus = list(found[:, :2]), list(found[:, 2]), [0.0] * len(found)
        ok = False
        if L.online:
            ok, _ = online.update_tracker(frame)
            if ok:
                c = np.array(online.centre)
                keep = [i for i, p in enumerate(pos) if np.hypot(*(p - c)) > r]  # The sighting replaces peaks on it
                pos, mass, bonus = [pos[i] for i in keep], [mass[i] for i in keep], [bonus[i] for i in keep]
                pos.append(c - online.offset)
                mass.append(image.mass_at(c - online.offset))
                bonus.append(L.online_bonus)
        is_online = np.zeros(len(pos), bool)
        if ok:
            is_online[-1] = True    # The sighting was added last
        pos = np.array(pos).reshape(-1, 2)
        likeness = np.array([look(gray, p) for p in pos])
        cost = s.w_unlike * (1.0 - np.clip(likeness, -1, 1)) - np.array(bonus)
        if mass0:
            ratio = np.maximum(np.array(mass), 1e-6) / mass0
            cost += s.w_mass * (np.log(ratio) / np.log(3.0)) ** 2
        P.append(pos), U.append(cost), M.append(likeness), O.append(is_online)
        if budget and t >= 10 and 1000 * (time.process_time() - start) / t > budget:
            stopped = t
            break
    used = len(P)

    # The cheapest path (forward pass): for every candidate, its cheapest way in and the motion
    # state (velocity, acceleration) of that path
    C = [np.zeros(1)] + [np.full(len(p), np.inf) for p in P[1:]]
    V = [np.zeros((1, 2))] + [np.zeros((len(p), 2)) for p in P[1:]]
    A = [np.zeros((1, 2))] + [np.zeros((len(p), 2)) for p in P[1:]]
    back = [None] + [np.full((len(p), 2), -1) for p in P[1:]]
    for t in range(1, used):
        if not len(P[t]):
            continue
        frames_from = [u for u in range(max(0, t - L.max_gap), t) if np.isfinite(C[u]).any()]
        if not frames_from:
            continue
        pp = np.concatenate([P[u] for u in frames_from])
        vv = np.concatenate([V[u] for u in frames_from])
        aa = np.concatenate([A[u] for u in frames_from])
        cc = np.concatenate([C[u] for u in frames_from])
        gap = np.concatenate([np.full(len(P[u]), t - u) for u in frames_from]).astype(float)
        ids = np.concatenate([np.column_stack([np.full(len(P[u]), u), np.arange(len(P[u]))]) for u in frames_from])
        pred = pp + vv * gap[:, None] + 0.5 * aa * (gap ** 2)[:, None]
        jumped = gap - 1
        tol = r * (L.tol + L.tol_gap * jumped + L.tol_gap2 * jumped ** 2) + L.tol_speed * np.hypot(*vv.T) * gap
        d = np.hypot(pred[:, None, 0] - P[t][None, :, 0], pred[:, None, 1] - P[t][None, :, 1])
        step = L.w_motion * (d / tol[:, None]) ** 2
        # ...or a kick, judged by the distance from the last position only
        reach = r * L.tol + (np.hypot(*vv.T) + L.kick_reach * r) * gap
        d0 = np.hypot(pp[:, None, 0] - P[t][None, :, 0], pp[:, None, 1] - P[t][None, :, 1])
        kick = L.kick + L.w_motion * (d0 / reach[:, None]) ** 2
        kicked = kick < step
        step = np.where(kicked, kick, step)
        step[step > L.max_step] = np.inf
        total = (cc + L.miss * jumped)[:, None] + step + U[t][None, :]
        best = np.argmin(total, axis=0)
        cols = np.arange(len(P[t]))
        C[t] = total[best, cols]
        back[t] = ids[best]
        g = gap[best][:, None]
        surprise = P[t] - pred[best]
        smooth_v = vv[best] + aa[best] * g + L.follow_v * surprise / g
        V[t] = np.where(kicked[best, cols][:, None], (P[t] - pp[best]) / g, smooth_v)
        A[t] = np.where(kicked[best, cols][:, None], aa[best], aa[best] + L.follow_a * surprise / g ** 2)

    # The path's end: the cheapest, counting the frames after it as missed
    ends = [(C[t][k] + L.miss * (n - 1 - t), t, k) for t in range(used) for k in range(len(P[t]))
            if np.isfinite(C[t][k])]
    _, t, k = min(ends)
    chosen = {}
    while t > 0:
        chosen[t] = k
        t, k = back[t][k]
    chosen[0] = 0

    centre, status, score = np.zeros((n, 2)), np.zeros(n, int), np.zeros(n)
    times = sorted(chosen)
    for t in times:
        k = chosen[t]
        centre[t], status[t], score[t] = P[t][k], MEASURED, float(np.clip(M[t][k], 0, 1))
        if t and not O[t][k]:
            # A peak of the shrunk frame: its blob at full size, for a sub-pixel centre
            found = online._blobs(online._gray(np.asarray(frames[t])), P[t][k], r)
            if len(found):
                j = int(np.argmin(np.hypot(*(found[:, :2] - P[t][k]).T)))
                if np.hypot(*(found[j, :2] - P[t][k])) <= r:
                    centre[t] = found[j, :2]
    status[0], score[0] = MEASURED, 1.0
    for a, b in zip(times, times[1:]):   # Jumped frames: a cubic between the two sides' motion
        if b - a > 1:
            va = V[a][chosen[a]]
            vb = V[b][chosen[b]]
            for t in range(a + 1, b):
                centre[t] = _hermite(centre[a], va, centre[b], vb, (t - a) / (b - a), b - a)
                status[t] = FILLED
    last = times[-1]
    centre[last + 1:] = centre[last]
    centre += online.offset
    ms = 1000 * (time.process_time() - start) / (stopped or n - 1)
    LAST_RUN.update(P=P, U=U, M=M, C=C, chosen=chosen)  # For inspecting a run
    return ClipTrack(centre, status, score, ms, stopped)


LAST_RUN: dict = {}


def _radius(sel) -> float:
    """The object's radius from the selection (rect, outline)."""
    rect, outline = sel
    if outline is not None:
        from experiments.rigid.blobtracker import outline_circle
        return outline_circle(outline)[1]
    return max(rect.width, rect.height) / 2.0


def _hermite(p0, v0, p1, v1, u, span):
    """Cubic from p0 (velocity v0, per frame) to p1 (velocity v1) at fraction u of `span` frames."""
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * span * v0 + h01 * p1 + h11 * span * v1
