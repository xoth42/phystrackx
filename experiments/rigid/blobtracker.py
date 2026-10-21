"""
blobtracker.py

A ball tracker for lab footage: trackpy finds ball-like blobs, and a motion model says which one is
the ball.

- Finding: only a window around the predicted position is searched (a few ms per frame). The
  window is turned into a contrast image (how much each pixel differs from its surroundings, so
  a shiny ball's bright and dark parts both count), blurred by half the ball's radius (so the
  whole ball is one peak, centred on it), and trackpy.locate finds the ball-sized blobs in it,
  with sub-pixel centres and their mass. A ball with too little contrast to stand out as a blob is
  found by matching its picture (template matching) instead.
- Choosing: each candidate is scored by how far it is from the prediction, measured against how
  sure the prediction is; how much it looks like the ball (normalised correlation with the inside
  of the ball, so a veil or fog on the lens does not matter); and how close its mass is to the
  ball's. A candidate far from the prediction must clearly look like the ball.
- Motion: a Kalman filter on position, velocity and acceleration. When the ball is not seen, the
  prediction carries on and its uncertainty grows, so the search widens by itself; a weak
  sighting (partly covered, merged with something) is a noisy measurement that barely changes
  the motion. When nothing clearly like the ball is near the prediction (it bounced, or was
  pushed), the area all around the last sighting is searched for it.

This is trackpy's linking idea (a search range, memory of a vanished particle, a motion
predictor) for one object, done frame by frame so tracking can be shown live and stopped.
Same interface as MLTracker. No GUI here. Author: Isam Balghari
"""

import cv2
import numpy as np
import trackpy as tp

from core import PixelRect

tp.quiet()

STRONG = 0.6     # A match at least this good is clearly the ball
GATE = 16.0      # Squared distance from the prediction, in standard deviations (4 sigma)


class BlobTracker:
    def __init__(self, min_match: float = 0.35, mass_range=(0.3, 3.0), jerk: float = 0.4) -> None:
        self.min_match = min_match    # Least likeness for a candidate away from the prediction
        self.mass_range = mass_range  # Allowed blob mass, as multiples of the ball's
        self.jerk = jerk              # How fast the acceleration may change (px / frame^3)
        self.centre = None            # Sub-pixel centre (x, y) of the last update
        self._score = 0.0

    # ---------------------------------------------------------------------------------------------
    def initialize_tracker(self, frame, rect: PixelRect) -> None:
        self.r = max(rect.width, rect.height) / 2.0
        self.d = int(2 * round(self.r) + 1) | 1                # trackpy wants an odd diameter
        self.half = max(2, int(0.7 * self.r))                  # Appearance patch: inside the ball
        self.base = max(3.0 * self.r, 12.0)                    # Smallest search radius
        centre = np.array([rect.xmin + rect.width / 2.0, rect.ymin + rect.height / 2.0])
        gray = self._gray(frame)
        self.ref = self._patch(gray, centre)
        if self.ref is not None and float(self.ref.std()) < 4.0:
            # A plain ball looks flat inside: use its edge and a little of what is around it
            self.half = int(np.ceil(self.r)) + 1
            self.ref = self._patch(gray, centre)
        self.mass, self.found_count = None, 0
        found = self._blobs(gray, centre, self.base)
        if len(found):
            k = int(np.argmin(np.hypot(found[:, 0] - centre[0], found[:, 1] - centre[1])))
            if np.hypot(*(found[k, :2] - centre)) <= self.r:
                centre, self.mass, self.found_count = found[k, :2].copy(), float(found[k, 2]), 1
                self.ref = self._patch(gray, centre)
        # Kalman filter: state (x, y, vx, vy, ax, ay), one frame per step
        self.x = np.array([centre[0], centre[1], 0, 0, 0, 0], float)
        self.P = np.diag([(self.r / 2) ** 2] * 2 + [self.r ** 2] * 2 + [0.5 ** 2] * 2)
        f = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])
        q = np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]]) * jerk_sq(self.jerk)
        self.F, self.Q = np.kron(f, np.eye(2)), np.kron(q, np.eye(2))
        self.H = np.hstack([np.eye(2), np.zeros((2, 4))])
        self.seen, self.missing = centre.copy(), 0
        self.centre = (float(centre[0]), float(centre[1]))
        self._score = 1.0

    def score(self) -> float:
        """How much the last sighting looked like the ball (0 when it was not found)."""
        return self._score

    def update_tracker(self, frame) -> tuple[bool, PixelRect]:
        gray = self._gray(frame)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        pred = self.x[:2].copy()
        speed = float(np.hypot(*self.x[2:4]))
        S = self.H @ self.P @ self.H.T + np.eye(2) * 0.7 ** 2
        sigma = float(np.sqrt(np.linalg.eigvalsh(S).max()))
        reach = min(np.sqrt(GATE) * sigma + self.r, 4 * self.base + speed)
        found = self._choose(gray, pred, pred, reach, np.linalg.inv(S), near=self.r)
        strong = found is not None and (found[1] >= STRONG or self.half < 3)  # (a tiny ball's few
        if not strong and self.found_count:                                    # pixels never match well)
            # Nothing clearly the ball near the prediction. It may have bounced or been pushed:
            # look all around the last sighting, as far as it could have gone since, for something
            # that clearly looks like the ball
            wide = self.base + 1.5 * speed * (self.missing + 1)
            other = self._choose(gray, self.seen, pred, wide, None, near=0.0, least=STRONG)
            if other is not None:
                found, strong = other, True
        if found is None:
            self.missing += 1
            self._score = 0.0
            self.centre = (float(pred[0]), float(pred[1]))
            return False, PixelRect(0, 0, 0, 0)
        best, match, mass = found
        # A weak sighting is a noisy measurement: it barely moves the motion
        noise = 0.7 + 6.0 * (1.0 - min(match, 1.0)) if strong else self.r
        if strong and np.hypot(*(best - pred)) > reach:
            # Found by the wide search: start the motion again from here
            self.x[:2], self.x[2:4], self.x[4:] = best, (best - self.seen) / (self.missing + 1), 0
            self.P = np.diag([1.0] * 2 + [self.r ** 2] * 2 + [0.5 ** 2] * 2)
        else:
            S = self.H @ self.P @ self.H.T + np.eye(2) * noise ** 2
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ (best - pred)
            self.P = (np.eye(6) - K @ self.H) @ self.P
        if strong:
            self.seen, self.missing = best.copy(), 0
            self.found_count += 1
            # Learn how the ball looks where it is clearly found (a hand or a veil on the first
            # frame fades out of the reference; a cover does not get into it)
            if mass is not None:
                self.mass = mass if self.mass is None else 0.8 * self.mass + 0.2 * mass
            patch = self._patch(gray, best)
            if patch is not None and (self.ref is None or match > 0.7 or self.found_count < 3):
                self.ref = patch if self.ref is None else 0.7 * self.ref + 0.3 * patch
        else:
            self.missing += 1
        self._score = match
        self.centre = (float(best[0]), float(best[1]))
        r = int(round(self.r))
        return True, PixelRect(int(round(best[0])) - r, int(round(best[1])) - r, 2 * r, 2 * r)

    # ---------------------------------------------------------------------------------------------
    def _choose(self, gray, around, pred, reach, s_inv, near, least=None):
        """The best candidate within `reach` of `around`, as (centre, match, mass), or None.
        The cost adds the distance from the prediction (in standard deviations when `s_inv` is
        given, else as a fraction of `reach`), how unlike the ball it looks, and how far its mass
        is from the ball's. Within `near` of the prediction a weak likeness is enough; farther
        out it must clearly look like the ball (at least `least`, if given)."""
        cands = [(x, y, mass) for x, y, mass in self._blobs(gray, around, reach)]
        if near > 0:  # Not in the wide search: a picture match far away is too easily something else
            peak = self._template_peak(gray, pred, min(reach, max(self.r, 0.5 * reach)))
            if peak is not None:
                mass = self._mass_at(gray, peak)
                # A picture match alone is weak evidence (a chrome ball's horizon looks like any
                # edge), so its blob mass must also be close to the ball's
                if self.mass is None or 0.5 * self.mass <= mass <= 2.0 * self.mass:
                    cands.append((peak[0], peak[1], mass))
        best, best_cost = None, np.inf
        for x, y, mass in cands:
            if np.hypot(x - around[0], y - around[1]) > reach:
                continue
            nu = np.array([x - pred[0], y - pred[1]])
            dist = float(np.hypot(*nu))
            if self.mass is not None and not (
                    self.mass_range[0] * self.mass <= mass <= self.mass_range[1] * self.mass):
                continue
            match = self._match(gray, (x, y))
            if self.found_count == 0 and dist > self.r and match < STRONG:
                continue  # Not found yet: the ball is near where it was selected
            if match < (least if least is not None else 0.1 if dist <= near else self.min_match):
                continue
            if s_inv is not None:
                d2 = float(nu @ s_inv @ nu)
                if d2 > GATE and dist > near:
                    continue
                far = d2 / GATE
            else:
                far = dist / max(reach, 1.0)
            off = np.log(mass / self.mass) / np.log(3.0) if self.mass and mass > 0 else 0.0
            cost = far + 1.5 * (1.0 - match) + 0.5 * off * off
            if cost < best_cost:
                best, best_cost = (np.array([x, y]), match, mass), cost
        return best

    @staticmethod
    def _gray(frame) -> np.ndarray:
        return (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame).astype(np.float32)

    def _contrast(self, gray, centre, reach):
        """The blurred contrast image of a window around `centre`, and the window's corner."""
        h, w = gray.shape
        sigma = 1.5 * self.r
        pad = int(reach + 2 * sigma + self.d)
        x0, y0 = max(0, int(centre[0]) - pad), max(0, int(centre[1]) - pad)
        x1, y1 = min(w, int(centre[0]) + pad + 1), min(h, int(centre[1]) + pad + 1)
        if x1 - x0 < 2 * self.d or y1 - y0 < 2 * self.d:
            return None, x0, y0
        win = gray[y0:y1, x0:x1]
        contrast = np.abs(win - cv2.GaussianBlur(win, (0, 0), sigma))
        # A shiny ball is bright above and dark below its middle: spread the contrast over about
        # half its radius so the whole ball is one peak, centred on the ball
        return cv2.GaussianBlur(contrast, (0, 0), max(1.0, 0.5 * self.r)), x0, y0

    def _blobs(self, gray, centre, reach) -> np.ndarray:
        """Ball-sized blobs within `reach` of `centre`, as rows of (x, y, mass) in frame pixels."""
        contrast, x0, y0 = self._contrast(gray, centre, reach)
        if contrast is None:
            return np.zeros((0, 3))
        found = tp.locate(contrast, self.d, separation=self.d, preprocess=False,
                          characterize=False, engine="python")
        if not len(found):
            return np.zeros((0, 3))
        return np.column_stack([found["x"].to_numpy() + x0, found["y"].to_numpy() + y0,
                                found["mass"].to_numpy()])

    def _mass_at(self, gray, centre) -> float:
        """The blob mass at `centre`, as trackpy.locate measures it (contrast summed over a disc)."""
        contrast, x0, y0 = self._contrast(gray, centre, 0)
        if contrast is None:
            return 0.0
        k = self.d // 2
        yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
        cx, cy = int(round(centre[0])) - x0, int(round(centre[1])) - y0
        part = contrast[max(0, cy - k):cy + k + 1, max(0, cx - k):cx + k + 1]
        disc = (xx * xx + yy * yy <= k * k)[:part.shape[0], :part.shape[1]]
        return float(part[disc].sum())

    def _patch(self, gray, centre) -> np.ndarray | None:
        k = self.half
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        if cx - k < 0 or cy - k < 0 or cx + k + 1 > gray.shape[1] or cy + k + 1 > gray.shape[0]:
            return None
        return gray[cy - k:cy + k + 1, cx - k:cx + k + 1].copy()

    def _match(self, gray, centre) -> float:
        """Normalised correlation between the ball's picture and the picture at `centre`,
        allowing two pixels of shift either way."""
        if self.ref is None:
            return 1.0
        k = self.half + 2
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        if cx - k < 0 or cy - k < 0 or cx + k + 1 > gray.shape[1] or cy + k + 1 > gray.shape[0]:
            return 0.0
        area = gray[cy - k:cy + k + 1, cx - k:cx + k + 1]
        if float(area.std()) < 2.0 or float(self.ref.std()) < 2.0:
            return 0.0  # A flat patch matches anything: it says nothing
        return float(cv2.matchTemplate(area, self.ref, cv2.TM_CCOEFF_NORMED).max())

    def _template_peak(self, gray, around, reach):
        """Where the ball's picture matches best within `reach` of `around`, if it matches well."""
        if self.ref is None or self.half < 3 or float(self.ref.std()) < 2.0:
            return None  # A tiny or flat picture matches too much to be a finder on its own
        k = self.half
        h, w = gray.shape
        x0, y0 = max(0, int(around[0] - reach) - k), max(0, int(around[1] - reach) - k)
        x1, y1 = min(w, int(around[0] + reach) + k + 1), min(h, int(around[1] + reach) + k + 1)
        if x1 - x0 <= 2 * k + 1 or y1 - y0 <= 2 * k + 1:
            return None
        res = cv2.matchTemplate(gray[y0:y1, x0:x1], self.ref, cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(res)
        if score < 0.8:
            return None
        return x0 + px + k, y0 + py + k


def jerk_sq(jerk: float) -> float:
    return jerk * jerk
