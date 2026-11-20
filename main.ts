// gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn
// rate at the case's target.
//
// Model. The rate is a ratio, earners over tracked players, and the two sides
// move for different reasons — but fitting them apart extrapolates each side's own
// curvature and the errors do not cancel, so the forecast overshoots. Instead the
// log-odds of the rate itself is fit: logit(r) against ln(1 + effective play
// hours since release), over the case's reads, weighted toward recent reads and
// toward reads with more players tracked. Effective play hours count weekend hours
// at 1.5x: players play more on Saturday and Sunday, and a plain clock axis makes
// a fit over Friday-Saturday understate the progress a horizon containing the
// whole weekend will see. The extrapolated move past the last read is damped
// (rates saturate; a log-time line does not), and earners are kept non-decreasing
// (a forecast rate may fall by dilution, but never below what the last earners
// count implies). The spread is the fit's prediction error at the target plus a
// refresh-shock term for the jumps Exophase's discrete refreshes cause, floored
// and capped so a flat history can never produce a degenerate zero-width band.
//
// Deterministic: pure arithmetic over the case file, no randomness, no clock, no
// network. Runs in milliseconds.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: 2026-11-19T00:00:00Z, the origin of the time axis. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** Weekend play-rate multiplier (Saturday/Sunday hours count this much). */
const WKND = 1.5;
/** How much of the fitted trend's move past the last read is kept (saturation). */
const DAMP = 0.8;
/** Refresh / regime shock, in logit units, added in quadrature to every spread. */
const SHOCK = 0.35;
/** Spread floor and cap, in logit units. */
const SPREAD_FLOOR = 0.4;
const SPREAD_CAP = 1.2;
/** Recency half-life for read weights, in hours. */
const HALF_LIFE_H = 4;

interface Trophy { id: number }
interface Read { at: string; players_tracked: number; rates: Array<number | null> }
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/**
 * Effective play hours from release to an instant: clock hours with weekend
 * (Saturday/Sunday UTC) hours weighted by WKND. Integrated in 15-minute steps;
 * the release was a Thursday, so the warping is exact on quarter hours.
 */
function playHours(iso: string): number {
  const t = Date.parse(iso);
  let h = 0;
  const step = 0.25 * 3600e3;
  for (let u = RELEASE_MS; u < t; u += step) {
    const d = new Date(u + step / 2).getUTCDay();
    h += ((d === 0 || d === 6) ? WKND : 1) * Math.min(step, t - u) / 3600e3;
  }
  return h;
}
/** x = ln(1 + effective play hours since release). */
const xOf = (iso: string): number => Math.log1p(Math.max(0, playHours(iso)));

interface Fit { a: number; b: number; sigma: number; n: number; xBar: number; sxx: number; xLast: number; yLast: number }

/**
 * Weighted least squares of y on x. Returns the line, the residual scale, and
 * the fitted last level. Null when degenerate (fewer than 2 distinct x values
 * or no spread in x).
 */
function fitLine(xs: number[], ys: number[], ws: number[]): Fit | null {
  let sw = 0, sx = 0, sy = 0;
  for (let k = 0; k < xs.length; k++) { sw += ws[k]; sx += ws[k] * xs[k]; sy += ws[k] * ys[k]; }
  if (sw <= 0) return null;
  const xBar = sx / sw, yBar = sy / sw;
  let sxx = 0, sxy = 0;
  for (let k = 0; k < xs.length; k++) { sxx += ws[k] * (xs[k] - xBar) ** 2; sxy += ws[k] * (xs[k] - xBar) * (ys[k] - yBar); }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx, a = yBar - b * xBar;
  let rss = 0;
  for (let k = 0; k < xs.length; k++) rss += ws[k] * (ys[k] - (a + b * xs[k])) ** 2;
  const n = xs.length;
  const sigma = Math.sqrt(rss / Math.max(1, n - 2));
  const xLast = xs[xs.length - 1];
  return { a, b, sigma, n, xBar, sxx, xLast, yLast: a + b * xLast };
}

/** Prediction standard error of a fit at xStar (logit units). */
function predSe(f: Fit, xStar: number): number {
  const lev = 1 + 1 / Math.max(1, f.n) + (xStar - f.xBar) ** 2 / f.sxx;
  return f.sigma * Math.sqrt(Math.max(0, lev));
}

/** One trophy's forecast row. */
function forecastRow(c: Case, i: number, xStar: number): { id: number; q: number[] } {
  const last = c.history[c.history.length - 1];
  const lastRate = last.rates[i] ?? 0;

  // A trophy everyone already has stays (almost) there: the rate can only dip a
  // little as brand-new players join before earning it.
  if (lastRate >= 99.99) {
    return { id: c.trophies[i].id, q: [99.9, 99.97, 99.99, 100, 100] };
  }

  // Distinct reads only: Exophase refreshes in steps, so long runs of identical
  // reads carry no new information and would fake a tight fit.
  const reads: Read[] = [];
  for (const h of c.history) {
    const prev = reads[reads.length - 1];
    if (prev && prev.players_tracked === h.players_tracked && prev.rates[i] === h.rates[i]) continue;
    reads.push(h);
  }
  const tLast = Date.parse(last.at);

  // Fit logit(rate) against weekend-warped log-time, recency-weighted, and
  // weighted toward reads with more players tracked (a rate over more players
  // is a more precise rate).
  const xs: number[] = [], ys: number[] = [], ws: number[] = [];
  for (const h of reads) {
    const r = h.rates[i];
    if (r === null || r === undefined) continue;
    xs.push(xOf(h.at));
    ys.push(logit(r));
    ws.push(Math.sqrt(h.players_tracked / Math.max(1, last.players_tracked)) * Math.exp(-(tLast - Date.parse(h.at)) / (HALF_LIFE_H * 3600e3)));
  }
  const f = xs.length >= 3 ? fitLine(xs, ys, ws) : null;

  // Forecast tracked players at the target (log-time linear: adoption is close
  // to a power law), only to bound the rate below by non-decreasing earners.
  const xn: number[] = [], yn: number[] = [], wn: number[] = [];
  for (const h of reads) {
    if (h.players_tracked <= 0) continue;
    xn.push(xOf(h.at));
    yn.push(Math.log(h.players_tracked));
    wn.push(Math.exp(-(tLast - Date.parse(h.at)) / (HALF_LIFE_H * 3600e3)));
  }
  const fN = xn.length >= 3 ? fitLine(xn, yn, wn) : null;
  const nStar = fN ? Math.exp(fN.a + fN.b * xStar) : last.players_tracked;

  let med: number, spread: number;
  if (f) {
    // Median: move from the fitted last level by the damped trend.
    const l = f.yLast + DAMP * f.b * (xStar - f.xLast);
    // Earners cannot fall: the rate may drop by dilution, but not below what
    // the last earners count implies at the forecast tracked count.
    const eLast = (lastRate / 100) * last.players_tracked;
    const rFloor = 100 * (eLast / Math.max(1, nStar));
    med = clamp(invLogit(l));
    med = Math.max(med, rFloor);
    // Spread: the fit's prediction error at the target plus the refresh shock.
    spread = Math.sqrt(predSe(f, xStar) ** 2 + SHOCK * SHOCK);
  } else {
    // Too little variation to fit: stay at the last rate, admit it widely.
    med = clamp(lastRate);
    spread = 0.9;
  }
  spread = Math.min(SPREAD_CAP, Math.max(SPREAD_FLOOR, spread));

  const l = logit(med);
  const q = Z.map((z) => {
    const v = round4(invLogit(l + z * spread));
    return Math.min(100, Math.max(0, v));
  });
  // Enforce non-decreasing (rounding can tie adjacent values, never invert).
  for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
  return { id: c.trophies[i].id, q };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const xStar = Math.log1p(Math.max(0, playHours(c.target)));
  const trophies = c.trophies.map((_, i) => forecastRow(c, i, xStar));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
