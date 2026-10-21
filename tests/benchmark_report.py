"""
Builds a page that shows the tracker benchmark: for every scenario in test_tracker_scenarios, a
player that steps through the clip with the true ball position and the tracker's box drawn on
it, an error-over-time chart, and the scores against the scenario's target.

    uv run python -m tests.benchmark_report                 # writes temp/benchmark_report.html
    uv run python -m tests.benchmark_report some/place.html

The page is one file with the frames embedded, so it opens anywhere. Frames are every third one,
scaled down; the chart and scores use every frame.
"""

import base64
import json
import os
import sys

import cv2

from tests import harness as h
h.setup_env()

from tests.test_tracker_scenarios import N, W, H, build_scenarios, score, trace  # noqa: E402

STEP = 3                 # One frame in three is embedded
SHOT = (400, 300)        # Embedded frame size
BLURBS = {
    "white, clean": "A dark red ball loops across a white screen. Nothing else is in the picture.",
    "white, small and fast": "A small ball (7 px radius) bounces off the edges at 9 px per frame.",
    "white, still shapes": "Five coloured shapes stand on the white screen, and the ball passes by them.",
    "paper wall, moving shapes": "An off-white wall with shading and camera noise. Five shapes drift around.",
    "paper wall, look-alike ball": "The same wall, with two decoy balls of the same colour moving across the ball's path.",
    "noise background": "Random grey pixels everywhere, a dark ball, and shapes drifting past.",
    "camouflage background": "Green, brown and tan blobs. The ball is painted in the camouflage's own green.",
    "camouflage, fine pattern, look-alikes": "Small camouflage blobs about the ball's size, three same-coloured decoys, noise and flicker.",
    "camouflage, fast, occluders": "A fast ball (13 px per frame) on camouflage, two decoys, and large shapes that pass in front of it.",
    "dropped by a hand": "A small dark ball held by a hand, then dropped. It falls and bounces, and the hand moves away.",
}


def collect() -> list:
    out = []
    for scn in build_scenarios():
        print(f"running: {scn.name}", flush=True)
        tr = trace(scn)
        res = score(scn, tr)
        target = scn.target
        meets = (res.median <= target[0] and res.p95 <= target[1] and res.on_ball >= target[2]
                 and res.lost <= scn.max_lost)
        frames = []
        for t in range(0, N, STEP):
            small = cv2.resize(scn.frame(t), SHOT, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
            frames.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
        out.append({
            "name": scn.name, "tier": scn.tier, "weak": scn.known_weak, "meets": bool(meets),
            "blurb": BLURBS.get(scn.name, ""), "radius": scn.radius,
            "stats": {"median": res.median, "p95": res.p95, "onBall": res.on_ball, "lost": res.lost,
                      "worst": res.worst, "firstMiss": res.first_miss},
            "target": {"median": target[0], "p95": target[1], "onBall": target[2]},
            "floor": scn.floor_on_ball if scn.known_weak else None,
            "truth": [[round(float(x), 1), round(float(y), 1)] for x, y in tr.truth],
            "box": [[round(float(v), 1) for v in b] for b in tr.box],
            "error": [round(float(e), 2) for e in tr.error],
            "frames": frames,
        })
    return out


PAGE = r"""<title>Tracker Benchmark</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>
/* Layout: a header with the legend, one summary table, then a grid of players, one per scenario.
   Overlay colours are fixed (lime ring, magenta box) so they read on white, noise and camouflage. */
:root {
  --bg: #f2f4f1; --surface: #ffffff; --ink: #16201c; --muted: #58645e; --line: #d6dcd7;
  --accent: #2749c4; --good: #0b7a57; --good-bg: #dcf1e7; --warn: #a8480a; --warn-bg: #fbe6d3;
  --track: #e6ebe7;
  --display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --body: "Source Sans 3", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1411; --surface: #161d19; --ink: #e6ece8; --muted: #98a49d; --line: #28322d;
  --accent: #86a4ff; --good: #55d4a5; --good-bg: #12342a; --warn: #f2a560; --warn-bg: #3b2615;
  --track: #222b26; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1411; --surface: #161d19; --ink: #e6ece8; --muted: #98a49d; --line: #28322d;
  --accent: #86a4ff; --good: #55d4a5; --good-bg: #12342a; --warn: #f2a560; --warn-bg: #3b2615;
  --track: #222b26; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.5;
  padding-block: 28px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.wrap { max-width: 1120px; margin: 0 auto; display: grid; gap: 28px; }
.eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font-family: var(--display); font-weight: 600; font-size: clamp(38px, 7vw, 60px); line-height: 1; margin: 6px 0 10px; text-wrap: balance; }
.lede { max-width: 62ch; color: var(--muted); margin: 0; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 16px; align-items: center; font-size: 14px; }
.legend span { display: inline-flex; gap: 8px; align-items: center; }
.sw-ring { width: 14px; height: 14px; border-radius: 50%; border: 2px solid #7cff00; box-shadow: 0 0 0 1.5px #000; }
.sw-box { width: 15px; height: 15px; border: 2px solid #ff2bd6; box-shadow: 0 0 0 1.5px #000; }
button { font: inherit; cursor: pointer; }
.btn { background: var(--ink); color: var(--bg); border: 0; border-radius: 6px; padding: 7px 14px; font-weight: 600; }
.btn:focus-visible, .slider:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; min-width: 640px; font-size: 14px; }
th { text-align: left; font-weight: 600; color: var(--muted); font-size: 12px; letter-spacing: .05em; text-transform: uppercase; padding: 10px 12px; border-bottom: 1px solid var(--line); }
td { padding: 9px 12px; border-bottom: 1px solid var(--line); }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.tier { font-family: var(--mono); font-size: 12px; color: var(--muted); }
.bar { display: inline-block; vertical-align: middle; width: 70px; height: 7px; background: var(--track); border-radius: 4px; margin-right: 8px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--good); }
.bar.bad i { background: var(--warn); }
.pill { display: inline-block; font-size: 12px; font-weight: 600; padding: 2px 9px; border-radius: 99px; white-space: nowrap; }
.pill.ok { background: var(--good-bg); color: var(--good); }
.pill.weak { background: var(--warn-bg); color: var(--warn); }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 440px), 1fr)); gap: 18px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.card header { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: baseline; justify-content: space-between; }
.card h2 { font-family: var(--display); font-weight: 600; font-size: 22px; line-height: 1.1; margin: 0; }
.blurb { color: var(--muted); font-size: 14px; margin: 0; }
canvas { display: block; width: 100%; height: auto; border-radius: 4px; }
canvas.view { background: #888; aspect-ratio: 4 / 3; }
.controls { display: flex; align-items: center; gap: 10px; }
.controls .btn { padding: 4px 12px; min-width: 66px; }
.slider { flex: 1; min-width: 0; accent-color: var(--accent); }
.now { font-family: var(--mono); font-size: 12px; color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
.stats div { min-width: 0; }
.stats b { display: block; font-family: var(--mono); font-weight: 500; font-size: 15px; font-variant-numeric: tabular-nums; }
.stats span { font-size: 11px; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; }
.over { color: var(--warn); }
.note { font-size: 13px; color: var(--muted); margin: 0; }
@media (max-width: 520px) { .stats { grid-template-columns: repeat(3, 1fr); } }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">NanoTrack · <span id="count"></span> scenarios · 120 frames each · 640 × 480</div>
    <h1>Tracker benchmark</h1>
    <p class="lede">Each clip is drawn with a known ball path. The ball is selected on the first frame with the circle tool's bounding square, then tracked. The tracker's centre is compared with the true centre on every frame.</p>
    <div class="legend">
      <span><i class="sw-ring"></i> true ball position</span>
      <span><i class="sw-box"></i> tracker's box</span>
      <span>Error is in pixels. A frame is on the ball when the error is under the ball's radius.</span>
      <button class="btn" id="playall">Play all</button>
    </div>
  </header>
  <section class="tablewrap"><table id="summary"></table></section>
  <section class="cards" id="cards"></section>
</div>
<script>
const DATA = __DATA__;
const N = 120, STEP = 3, SRC_W = 640;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 2) => v.toFixed(d);
document.getElementById('count').textContent = DATA.length;

// Summary table
const sum = document.getElementById('summary');
sum.innerHTML = '<thead><tr><th>Scenario</th><th>Tier</th><th class="num">Median px</th><th class="num">95th % px</th>' +
  '<th>On the ball</th><th class="num">First miss</th><th>Result</th></tr></thead><tbody>' +
  DATA.map(s => {
    const st = s.stats, bad = st.onBall < s.target.onBall;
    const result = s.meets ? '<span class="pill ok">meets target</span>'
      : '<span class="pill weak">' + (s.weak ? 'known weakness' : 'below target') + '</span>';
    return '<tr><td>' + s.name + '</td><td class="tier">' + s.tier + '</td>' +
      '<td class="num' + (st.median > s.target.median ? ' over' : '') + '">' + fmt(st.median) + '</td>' +
      '<td class="num' + (st.p95 > s.target.p95 ? ' over' : '') + '">' + fmt(st.p95) + '</td>' +
      '<td><span class="bar' + (bad ? ' bad' : '') + '"><i style="width:' + (st.onBall * 100) + '%"></i></span>' +
      '<span class="num" style="font-family:var(--mono)">' + Math.round(st.onBall * 100) + '%</span></td>' +
      '<td class="num">' + (st.firstMiss === null ? '–' : 'frame ' + st.firstMiss) + '</td><td>' + result + '</td></tr>';
  }).join('') + '</tbody>';

// Players
const cards = document.getElementById('cards');
const players = DATA.map((s, k) => {
  s.imgs = s.frames.map(src => { const im = new Image(); im.onload = () => draw(p); im.src = src; return im; });
  const el = document.createElement('article');
  el.className = 'card';
  const st = s.stats;
  const nowrap = (cls, label, value) => '<div><b class="' + cls + '">' + value + '</b><span>' + label + '</span></div>';
  const status = s.meets ? '<span class="pill ok">meets target</span>'
    : '<span class="pill weak">' + (s.weak ? 'known weakness' : 'below target') + '</span>';
  el.innerHTML =
    '<header><h2>' + s.name + '</h2>' + status + '</header>' +
    '<p class="blurb">' + s.blurb + '</p>' +
    '<canvas class="view" width="400" height="300" aria-label="' + s.name + ' tracking playback"></canvas>' +
    '<div class="controls"><button class="btn play">Play</button>' +
    '<input class="slider" type="range" min="0" max="' + (s.frames.length - 1) + '" value="0" aria-label="frame">' +
    '<span class="now"></span></div>' +
    '<canvas class="chart" width="400" height="110" aria-label="error over time"></canvas>' +
    '<div class="stats">' +
      nowrap(st.median > s.target.median ? 'over' : '', 'median', fmt(st.median) + ' px') +
      nowrap(st.p95 > s.target.p95 ? 'over' : '', '95th %', fmt(st.p95) + ' px') +
      nowrap(st.onBall < s.target.onBall ? 'over' : '', 'on ball', Math.round(st.onBall * 100) + '%') +
      nowrap(st.lost ? 'over' : '', 'lost', st.lost) +
      nowrap('', 'worst', fmt(st.worst, 1) + ' px') + '</div>' +
    '<p class="note">Target for the ' + s.tier + ' tier: median ≤ ' + s.target.median + ' px, 95th % ≤ ' +
      s.target.p95 + ' px, on the ball ≥ ' + Math.round(s.target.onBall * 100) + '%' +
      (s.weak ? '. Known weakness: it must stay on the ball on at least ' + Math.round(s.floor * 100) + '% of frames.' : '.') + '</p>';
  cards.appendChild(el);
  const p = { s, el, view: el.querySelector('.view'), chart: el.querySelector('.chart'), slider: el.querySelector('.slider'),
              now: el.querySelector('.now'), btn: el.querySelector('.play'), i: 0, playing: false };
  // Rest on the interesting moment: just after the first miss, or the middle of the clip
  const rest = s.stats.firstMiss === null ? 60 : Math.min(N - 1, s.stats.firstMiss + 6);
  p.i = Math.round(rest / STEP);
  p.slider.value = p.i;
  p.slider.addEventListener('input', () => { p.i = +p.slider.value; draw(p); });
  p.btn.addEventListener('click', () => { p.playing = !p.playing; p.btn.textContent = p.playing ? 'Pause' : 'Play'; });
  return p;
});

function trail(ctx, pts, t, k, color, width) {
  ctx.beginPath();
  for (let u = Math.max(0, t - 14); u <= t; u++) {
    const x = pts[u][0] * k, y = pts[u][1] * k;
    u === Math.max(0, t - 14) ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.lineWidth = width + 2; ctx.strokeStyle = '#000'; ctx.stroke();
  ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke();
}

function draw(p) {
  const { s, view, chart } = p;
  const t = Math.min(N - 1, p.i * STEP);
  const ctx = view.getContext('2d'), k = view.width / SRC_W;
  const im = s.imgs[p.i];
  if (im.complete && im.naturalWidth) ctx.drawImage(im, 0, 0, view.width, view.height);
  const centre = s.box.map(b => [b[0] + b[2] / 2, b[1] + b[3] / 2]);
  trail(ctx, s.truth, t, k, '#7cff00', 1.5);
  trail(ctx, centre, t, k, '#ff2bd6', 1.5);
  ctx.beginPath(); ctx.arc(s.truth[t][0] * k, s.truth[t][1] * k, s.radius * k, 0, 6.2832);
  ctx.lineWidth = 4; ctx.strokeStyle = '#000'; ctx.stroke(); ctx.lineWidth = 2; ctx.strokeStyle = '#7cff00'; ctx.stroke();
  const b = s.box[t];
  ctx.lineWidth = 4; ctx.strokeStyle = '#000'; ctx.strokeRect(b[0] * k, b[1] * k, b[2] * k, b[3] * k);
  ctx.lineWidth = 2; ctx.strokeStyle = '#ff2bd6'; ctx.strokeRect(b[0] * k, b[1] * k, b[2] * k, b[3] * k);
  p.now.textContent = 'frame ' + t + ' · error ' + fmt(s.error[t], 1) + ' px';
  p.slider.value = p.i;
  drawChart(p, t);
}

function drawChart(p, t) {
  const { s, chart } = p, ctx = chart.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 3), cssW = chart.clientWidth || 400, cssH = 110;
  if (chart.width !== Math.round(cssW * dpr)) { chart.width = Math.round(cssW * dpr); chart.height = cssH * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  const L = 34, R = 8, T = 8, B = 20, w = cssW - L - R, h = cssH - T - B;
  const maxErr = Math.max(...s.error), cap = Math.max(1.6 * s.radius, Math.min(maxErr * 1.1, 3 * s.radius));
  const X = u => L + (u / (N - 1)) * w, Y = e => T + h - Math.min(e, cap) / cap * h;
  ctx.font = '10px ' + css('--mono'); ctx.fillStyle = css('--muted'); ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
  // Grid and axis labels
  [0, s.radius, cap].forEach(v => {
    ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + w, Y(v)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(v === cap && v !== s.radius ? fmt(v, 0) : fmt(v, v < 10 ? 1 : 0), L - 5, Y(v));
  });
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  [0, 30, 60, 90, 119].forEach(u => ctx.fillText(u, X(u), T + h + 5));
  // Frames where the tracker is off the ball
  ctx.fillStyle = css('--warn'); ctx.globalAlpha = .18;
  for (let u = 1; u < N; u++) if (s.error[u] > s.radius) ctx.fillRect(X(u) - w / N / 2, T, w / N, h);
  ctx.globalAlpha = 1;
  // Ball radius
  ctx.setLineDash([4, 3]); ctx.strokeStyle = css('--muted'); ctx.beginPath(); ctx.moveTo(L, Y(s.radius)); ctx.lineTo(L + w, Y(s.radius)); ctx.stroke(); ctx.setLineDash([]);
  // Error
  ctx.beginPath(); s.error.forEach((e, u) => u ? ctx.lineTo(X(u), Y(e)) : ctx.moveTo(X(u), Y(e)));
  ctx.strokeStyle = css('--accent'); ctx.lineWidth = 1.6; ctx.stroke();
  // Now
  ctx.strokeStyle = css('--ink'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(t), T); ctx.lineTo(X(t), T + h); ctx.stroke();
  ctx.fillStyle = css('--ink'); ctx.beginPath(); ctx.arc(X(t), Y(s.error[t]), 3, 0, 6.2832); ctx.fill();
  ctx.fillStyle = css('--muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('error (px) per frame · dashed: ball radius' + (maxErr > cap ? ' · clipped at ' + fmt(cap, 0) : ''), L + 4, T + 1);
}

// Playback: about 10 steps a second (30 frames a second of the clip)
let timer = null;
function tick() { players.forEach(p => { if (p.playing) { p.i = (p.i + 1) % p.s.frames.length; draw(p); } }); }
function ensureTimer() { if (!timer) timer = setInterval(tick, 100); }
ensureTimer();
const playall = document.getElementById('playall');
playall.addEventListener('click', () => {
  const on = playall.textContent === 'Play all';
  players.forEach(p => { p.playing = on; p.btn.textContent = on ? 'Pause' : 'Play'; });
  playall.textContent = on ? 'Pause all' : 'Play all';
});
// Redraw on resize and theme change (the charts read the theme's colours)
const redraw = () => players.forEach(draw);
window.addEventListener('resize', redraw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
redraw();
</script>
"""


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(h.REPO, "temp", "benchmark_report.html")
    data = collect()
    html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
