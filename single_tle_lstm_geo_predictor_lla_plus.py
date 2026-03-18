#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import ufo4_harmonic_azel_tf as hm

try:
    import ruptures as rpt
except Exception:
    rpt = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("expmod", ROOT / "pred_tle_single_90d_tf_experiment.py")
exp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exp
spec.loader.exec_module(exp)
core = exp.core


@dataclass
class Episode:
    tle_name: str
    unix: np.ndarray
    baseline_azel: np.ndarray
    true_azel: np.ndarray
    forecast_mask: np.ndarray
    static_features: np.ndarray


@dataclass
class LlaResidualPack:
    tle_name: str
    unix: np.ndarray
    baseline_azel: np.ndarray
    true_azel: np.ndarray
    pred_azel: np.ndarray
    baseline_lla: np.ndarray
    pred_lla_geo: np.ndarray
    true_lla: np.ndarray
    forecast_mask: np.ndarray
    static_features: np.ndarray


MINUTES_PER_DAY = 24 * 60
MINUTES_PER_YEAR = 365 * MINUTES_PER_DAY


def parse_int_csv(text: str) -> list[int]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def parse_float_csv(text: str) -> list[float]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def parse_str_csv(text: str) -> list[str]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out


def parse_tle_stem_flexible(stem: str) -> dt.datetime:
    base = str(stem)
    if "_" in base:
        head, tail = base.rsplit("_", 1)
        if tail.isdigit():
            base = head
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d-%H-%M-%S"):
        try:
            return dt.datetime.strptime(base, fmt)
        except ValueError:
            pass
    parsed = core.parse_tle_datetime_from_stem(base)
    if parsed is not None:
        return parsed
    raise ValueError(f"Cannot parse JST datetime from TLE stem: {stem}")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if core.tf is not None:
        core.tf.keras.utils.set_random_seed(int(seed))


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "count_below_0_1": 0,
            "mean_max_abs_error": float("nan"),
            "median_max_abs_error": float("nan"),
            "max_max_abs_error": float("nan"),
            "min_max_abs_error": float("nan"),
        }
    return {
        "count": int(arr.size),
        "count_below_0_1": int(np.sum(arr < 0.1)),
        "mean_max_abs_error": float(np.mean(arr)),
        "median_max_abs_error": float(np.median(arr)),
        "max_max_abs_error": float(np.max(arr)),
        "min_max_abs_error": float(np.min(arr)),
    }


def compute_teacher_mask(unix: np.ndarray, teacher_force_days: float) -> np.ndarray:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    if u.size == 0 or float(teacher_force_days) <= 0.0:
        return np.zeros((u.shape[0],), dtype=bool)
    end_unix = float(u[0]) + float(teacher_force_days) * float(exp.SOLAR_DAY_SECONDS)
    return np.asarray(u <= end_unix, dtype=bool)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = int(max(1, int(window)))
    if v.size == 0 or w <= 1:
        return v.astype(np.float64)
    if w % 2 == 0:
        w += 1
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    return np.convolve(v, kernel, mode="same").astype(np.float64)


def segment_by_pelt_median(
    values: np.ndarray,
    model: str,
    pen: float,
    min_size: int,
    jump: int,
    max_points: int = 8192,
) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if rpt is None or v.size < max(8, int(min_size) * 2):
        return v
    stride = int(max(1, math.ceil(float(v.size) / float(max(64, int(max_points))))))
    vd = np.asarray(v[::stride], dtype=np.float64) if stride > 1 else np.asarray(v, dtype=np.float64)
    if vd.size < max(8, int(min_size) * 2):
        return v
    sig = vd.reshape(-1, 1)
    try:
        algo = rpt.Pelt(
            model=str(model),
            min_size=int(max(2, int(min_size))),
            jump=int(max(1, int(jump))),
        ).fit(sig)
        bkps = algo.predict(pen=float(max(1.0e-9, float(pen))))
    except Exception:
        return v
    if not bkps:
        return v
    out_d = np.asarray(vd, dtype=np.float64).copy()
    start = 0
    for end in bkps:
        end_i = int(end)
        if end_i <= start:
            continue
        seg = np.asarray(vd[start:end_i], dtype=np.float64)
        if seg.size > 0:
            out_d[start:end_i] = float(np.median(seg))
        start = end_i
    if stride <= 1:
        return out_d
    idx_old = np.arange(out_d.size, dtype=np.float64) * float(stride)
    vals_old = np.asarray(out_d, dtype=np.float64)
    if idx_old[-1] < float(v.size - 1):
        idx_old = np.concatenate([idx_old, np.asarray([float(v.size - 1)], dtype=np.float64)], axis=0)
        vals_old = np.concatenate([vals_old, np.asarray([vals_old[-1]], dtype=np.float64)], axis=0)
    idx_new = np.arange(v.size, dtype=np.float64)
    return np.interp(idx_new, idx_old, vals_old).astype(np.float64)


def build_periodic_drift_prior_trig(
    unix: np.ndarray,
    base_azel: np.ndarray,
    fit_days: float,
    harmonics: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    base = np.asarray(base_azel, dtype=np.float64)
    trig_base = hm.azel_to_trig4(base)
    n = int(u.shape[0])
    if n == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            0.0,
            0.0,
        )

    fit_mask = compute_teacher_mask(u, teacher_force_days=float(fit_days))
    t_rel_day = (u - float(u[0])) / float(exp.SOLAR_DAY_SECONDS)
    phase_sid = 2.0 * math.pi * (u - float(u[0])) / float(hm.SECONDS_PER_SIDEREAL_DAY)
    phase_day = 2.0 * math.pi * (u - float(u[0])) / float(exp.SOLAR_DAY_SECONDS)

    cols = [
        np.ones((n,), dtype=np.float64),
        np.asarray(t_rel_day, dtype=np.float64),
        np.asarray(t_rel_day, dtype=np.float64) ** 2.0,
    ]
    hh = int(max(1, int(harmonics)))
    for k in range(1, hh + 1):
        ks = float(k)
        cols.append(np.sin(ks * phase_sid))
        cols.append(np.cos(ks * phase_sid))
        cols.append(np.sin(ks * phase_day))
        cols.append(np.cos(ks * phase_day))
    phi = np.column_stack(cols).astype(np.float64)

    fit_idx = np.where(np.asarray(fit_mask, dtype=bool))[0]
    min_rows = int(phi.shape[1] + 4)
    if fit_idx.size < min_rows:
        fit_idx = np.arange(n, dtype=np.int64)

    coef, *_ = np.linalg.lstsq(phi[fit_idx], trig_base[fit_idx], rcond=None)
    prior = np.asarray(phi @ coef, dtype=np.float64)
    prior = hm.normalize_trig4_np(prior).astype(np.float64)

    warm_idx = np.where(np.asarray(fit_mask, dtype=bool))[0]
    if warm_idx.size == 0:
        warm_idx = np.arange(n, dtype=np.int64)
    err = np.asarray(trig_base[warm_idx] - prior[warm_idx], dtype=np.float64)
    rmse_az = float(np.sqrt(np.mean(np.sum(np.square(err[:, 0:2]), axis=1))))
    rmse_el = float(np.sqrt(np.mean(np.sum(np.square(err[:, 2:4]), axis=1))))
    return prior.astype(np.float32), np.asarray(fit_mask, dtype=bool), rmse_az, rmse_el


def build_cycle_repeat_prior_trig(
    unix: np.ndarray,
    base_azel: np.ndarray,
    fit_days: float,
    min_period_minutes: float,
    max_period_minutes: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    base = np.asarray(base_azel, dtype=np.float64)
    trig_base = hm.azel_to_trig4(base).astype(np.float64)
    n = int(u.shape[0])
    if n == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            float(hm.SECONDS_PER_SIDEREAL_DAY / 60.0),
            0.0,
            0.0,
            0.0,
        )

    fit_mask = compute_teacher_mask(u, teacher_force_days=float(fit_days))
    fit_idx = np.where(np.asarray(fit_mask, dtype=bool))[0]
    if fit_idx.size < 16:
        fit_idx = np.arange(min(n, int(max(16, n))), dtype=np.int64)
    if fit_idx.size == 0:
        fit_idx = np.arange(n, dtype=np.int64)

    if n > 1:
        step_seconds = float(np.median(np.diff(u)))
        if not np.isfinite(step_seconds) or step_seconds <= 0.0:
            step_seconds = float(exp.SOLAR_DAY_SECONDS)
    else:
        step_seconds = float(exp.SOLAR_DAY_SECONDS)

    pmin = int(
        max(
            2,
            round(float(min_period_minutes) * 60.0 / max(1.0, float(step_seconds))),
        )
    )
    pmax = int(
        max(
            pmin + 1,
            round(float(max_period_minutes) * 60.0 / max(1.0, float(step_seconds))),
        )
    )
    pmax = min(pmax, max(3, int(max(3, fit_idx.size - 2))))
    if pmin >= pmax:
        pmin = max(2, pmax - 1)

    fit_trig = np.asarray(trig_base[fit_idx], dtype=np.float64)
    best_lag = int(
        max(
            2,
            round(float(hm.SECONDS_PER_SIDEREAL_DAY) / max(1.0, float(step_seconds))),
        )
    )
    best_score = -1.0e18
    for lag in range(int(pmin), int(pmax) + 1):
        if lag >= fit_trig.shape[0] - 1:
            break
        a = fit_trig[:-lag]
        b = fit_trig[lag:]
        sc = float(np.mean(np.sum(a * b, axis=1)))
        if np.isfinite(sc) and sc > best_score:
            best_score = sc
            best_lag = int(lag)
    period_steps = int(max(2, best_lag))
    period_minutes = float(period_steps) * float(step_seconds) / 60.0

    phase_fit = ((fit_idx - int(fit_idx[0])) % period_steps).astype(np.int32)
    cycle_fit = np.floor((fit_idx - int(fit_idx[0])) / float(period_steps)).astype(np.float64)
    template = np.zeros((period_steps, 4), dtype=np.float64)
    counts = np.zeros((period_steps,), dtype=np.float64)
    for c in range(4):
        np.add.at(template[:, c], phase_fit, fit_trig[:, c])
    np.add.at(counts, phase_fit, 1.0)
    valid = counts > 0.0
    if np.any(valid):
        template[valid] = template[valid] / counts[valid, None]
    else:
        template[:] = np.mean(fit_trig, axis=0, dtype=np.float64)[None, :]

    if not np.all(valid):
        xv = np.where(valid)[0].astype(np.float64)
        if xv.size >= 2:
            xq = np.arange(period_steps, dtype=np.float64)
            for c in range(4):
                template[:, c] = np.interp(xq, xv, template[valid, c], period=float(period_steps))
        else:
            template[:] = np.mean(fit_trig, axis=0, dtype=np.float64)[None, :]

    phase_a = np.asarray(template, dtype=np.float64).copy()
    phase_b = np.zeros((period_steps, 4), dtype=np.float64)
    valid_phase = counts > 0.0
    for ph in range(period_steps):
        use = np.where(phase_fit == int(ph))[0]
        if use.size == 0:
            continue
        k = np.asarray(cycle_fit[use], dtype=np.float64)
        if k.size >= 2 and np.std(k) > 1.0e-9:
            A = np.column_stack([np.ones((k.size,), dtype=np.float64), k]).astype(np.float64)
            coef, *_ = np.linalg.lstsq(A, fit_trig[use], rcond=None)
            phase_a[ph] = np.asarray(coef[0], dtype=np.float64)
            phase_b[ph] = np.asarray(coef[1], dtype=np.float64)
        else:
            phase_a[ph] = np.asarray(np.mean(fit_trig[use], axis=0), dtype=np.float64)
            phase_b[ph] = 0.0

    if not np.all(valid_phase):
        xv = np.where(valid_phase)[0].astype(np.float64)
        if xv.size >= 2:
            xq = np.arange(period_steps, dtype=np.float64)
            for c in range(4):
                phase_a[:, c] = np.interp(xq, xv, phase_a[valid_phase, c], period=float(period_steps))
                phase_b[:, c] = np.interp(xq, xv, phase_b[valid_phase, c], period=float(period_steps))
        else:
            phase_a[:] = np.mean(fit_trig, axis=0, dtype=np.float64)[None, :]
            phase_b[:] = 0.0

    idx_all = np.arange(n, dtype=np.int64)
    phase_all = ((idx_all - int(fit_idx[0])) % period_steps).astype(np.int32)
    cycle_all = np.floor((idx_all - int(fit_idx[0])) / float(period_steps)).astype(np.float64)
    prior = np.asarray(phase_a[phase_all] + phase_b[phase_all] * cycle_all[:, None], dtype=np.float64)

    if fit_idx.size >= 8:
        t = (u - float(u[0])) / float(exp.SOLAR_DAY_SECONDS)
        phi = np.column_stack([np.ones((n,), dtype=np.float64), np.asarray(t, dtype=np.float64)])
        coef, *_ = np.linalg.lstsq(phi[fit_idx], (trig_base - prior)[fit_idx], rcond=None)
        prior = prior + phi @ coef

    prior = hm.normalize_trig4_np(prior).astype(np.float64)
    err = np.asarray(trig_base[fit_idx] - prior[fit_idx], dtype=np.float64)
    rmse_az = float(np.sqrt(np.mean(np.sum(np.square(err[:, 0:2]), axis=1))))
    rmse_el = float(np.sqrt(np.mean(np.sum(np.square(err[:, 2:4]), axis=1))))
    return (
        prior.astype(np.float32),
        np.asarray(fit_mask, dtype=bool),
        float(period_minutes),
        float(best_score),
        float(rmse_az),
        float(rmse_el),
    )


def load_truth_cache(years: Sequence[int], suffix: str = "") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in years:
        p = ROOT / f"{int(year)}_calc_az_el{suffix}.csv"
        if p.exists():
            cache[int(year)] = core.load_orbit_numeric_full(p)
    return cache


def load_truth_sincos_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = None
    try:
        arr = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(1, 7, 8, 9, 10, 11, 12),
            dtype=np.float64,
        )
        arr = np.atleast_2d(arr)
        unix = np.asarray(arr[:, 0], dtype=np.float64)
        sincos = np.asarray(arr[:, 1:7], dtype=np.float64)
        return unix, sincos
    except Exception:
        unix, full = core.load_orbit_numeric_full(path)
        azel = np.asarray(full[:, 3:5], dtype=np.float64)
        trig = hm.azel_to_trig4(azel).astype(np.float64)
        sincos = np.column_stack(
            [
                np.sin(np.asarray(unix, dtype=np.float64)),
                np.cos(np.asarray(unix, dtype=np.float64)),
                np.asarray(trig[:, 0], dtype=np.float64),
                np.asarray(trig[:, 1], dtype=np.float64),
                np.asarray(trig[:, 2], dtype=np.float64),
                np.asarray(trig[:, 3], dtype=np.float64),
            ]
        ).astype(np.float64)
        return np.asarray(unix, dtype=np.float64), np.asarray(sincos, dtype=np.float64)


def calendar_minute_index_nonleap(unix: np.ndarray) -> np.ndarray:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    sec_local = np.asarray(np.round(u), dtype=np.int64) + int(core.JST_OFFSET_SEC)
    dt_s = sec_local.astype("datetime64[s]")
    dt_d = dt_s.astype("datetime64[D]")
    y0 = dt_s.astype("datetime64[Y]")
    doy = (dt_d - y0).astype(np.int32)
    year_int = y0.astype(np.int64) + 1970
    is_leap = (year_int % 4 == 0) & ((year_int % 100 != 0) | (year_int % 400 == 0))
    doy = doy - ((is_leap) & (doy >= 60)).astype(np.int32)
    doy = np.clip(doy, 0, 364)
    sod = np.mod(sec_local.astype(np.float64), float(exp.SOLAR_DAY_SECONDS))
    minute = np.clip(np.floor(sod / 60.0).astype(np.int32), 0, int(MINUTES_PER_DAY - 1))
    return (doy * int(MINUTES_PER_DAY) + minute).astype(np.int32)


def build_calendar_climatology_trig(years: Sequence[int], suffix: str) -> np.ndarray | None:
    years_use = [int(y) for y in years]
    if not years_use:
        return None
    sums = np.zeros((int(MINUTES_PER_YEAR), 4), dtype=np.float64)
    cnt = np.zeros((int(MINUTES_PER_YEAR),), dtype=np.float64)
    for year in years_use:
        p = ROOT / f"{int(year)}_calc_az_el{suffix}.csv"
        if not p.exists():
            continue
        unix, sincos = load_truth_sincos_columns(p)
        idx = calendar_minute_index_nonleap(unix)
        trig = np.asarray(sincos[:, 2:6], dtype=np.float64)
        for c in range(4):
            np.add.at(sums[:, c], idx, trig[:, c])
        np.add.at(cnt, idx, 1.0)
    valid = cnt > 0.0
    if not np.any(valid):
        return None
    out = np.zeros((int(MINUTES_PER_YEAR), 4), dtype=np.float64)
    out[valid] = sums[valid] / cnt[valid, None]
    mean_vec = np.mean(out[valid], axis=0, dtype=np.float64)
    out[~valid] = mean_vec[None, :]
    out = hm.normalize_trig4_np(out).astype(np.float32)
    return np.asarray(out, dtype=np.float32)


def build_calendar_trend_climatology_trig(years: Sequence[int], suffix: str, target_year: int) -> np.ndarray | None:
    years_use = [int(y) for y in years]
    if not years_use:
        return None
    per_year = []
    for year in years_use:
        p = ROOT / f"{int(year)}_calc_az_el{suffix}.csv"
        if not p.exists():
            continue
        unix, sincos = load_truth_sincos_columns(p)
        idx = calendar_minute_index_nonleap(unix)
        trig = np.asarray(sincos[:, 2:6], dtype=np.float64)
        sums = np.zeros((int(MINUTES_PER_YEAR), 4), dtype=np.float64)
        cnt = np.zeros((int(MINUTES_PER_YEAR),), dtype=np.float64)
        for c in range(4):
            np.add.at(sums[:, c], idx, trig[:, c])
        np.add.at(cnt, idx, 1.0)
        valid = cnt > 0.0
        out = np.zeros((int(MINUTES_PER_YEAR), 4), dtype=np.float64)
        out[valid] = sums[valid] / cnt[valid, None]
        mean_vec = np.mean(out[valid], axis=0, dtype=np.float64) if np.any(valid) else np.zeros((4,), dtype=np.float64)
        out[~valid] = mean_vec[None, :]
        per_year.append((int(year), out))
    if len(per_year) == 0:
        return None
    if len(per_year) == 1:
        return hm.normalize_trig4_np(np.asarray(per_year[0][1], dtype=np.float64)).astype(np.float32)

    ys = np.asarray([float(y) for y, _ in per_year], dtype=np.float64)
    arr = np.stack([np.asarray(v, dtype=np.float64) for _, v in per_year], axis=0)  # [Y, M, 4]
    ym = float(np.mean(ys))
    yv = float(np.sum((ys - ym) ** 2.0))
    if yv <= 1.0e-12:
        pred = np.mean(arr, axis=0, dtype=np.float64)
        return hm.normalize_trig4_np(pred).astype(np.float32)
    w = (ys - ym).reshape(-1, 1, 1)
    b = np.sum(w * arr, axis=0, dtype=np.float64) / yv
    a = np.mean(arr, axis=0, dtype=np.float64) - b * ym
    pred = a + b * float(target_year)
    pred = hm.normalize_trig4_np(np.asarray(pred, dtype=np.float64)).astype(np.float32)
    return np.asarray(pred, dtype=np.float32)


def smooth_circular_1d(values: np.ndarray, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    w = int(max(1, int(window)))
    if n == 0 or w <= 1:
        return np.asarray(v, dtype=np.float64)
    if w % 2 == 0:
        w += 1
    h = w // 2
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    pad = np.concatenate([v[-h:], v, v[:h]], axis=0)
    sm = np.convolve(pad, kernel, mode="same")[h : h + n]
    return np.asarray(sm, dtype=np.float64)


def build_calendar_maneuver_hazard_profile(
    years: Sequence[int],
    suffix: str,
    lag_minutes: float,
    smooth_minutes: int,
) -> np.ndarray | None:
    years_use = [int(y) for y in years]
    if not years_use:
        return None
    sums = np.zeros((int(MINUTES_PER_YEAR),), dtype=np.float64)
    cnt = np.zeros((int(MINUTES_PER_YEAR),), dtype=np.float64)
    for year in years_use:
        p = ROOT / f"{int(year)}_calc_az_el{suffix}.csv"
        if not p.exists():
            continue
        unix, sincos = load_truth_sincos_columns(p)
        u = np.asarray(unix, dtype=np.float64).reshape(-1)
        trig = np.asarray(sincos[:, 2:6], dtype=np.float64)
        n = int(u.shape[0])
        if n <= 4:
            continue
        idx = calendar_minute_index_nonleap(u)
        if n > 1:
            step_min = float(np.median(np.diff(u)) / 60.0)
            if not np.isfinite(step_min) or step_min <= 0.0:
                step_min = 1.0
        else:
            step_min = 1.0
        lag_steps = int(max(1, round(float(lag_minutes) / max(1.0e-9, step_min))))
        if lag_steps >= n:
            continue
        shock = np.zeros((n,), dtype=np.float64)
        d = np.asarray(trig[lag_steps:] - trig[:-lag_steps], dtype=np.float64)
        shock[lag_steps:] = np.sqrt(np.sum(np.square(d), axis=1))
        np.add.at(sums, idx, shock)
        np.add.at(cnt, idx, 1.0)
    valid = cnt > 0.0
    if not np.any(valid):
        return None
    h = np.zeros((int(MINUTES_PER_YEAR),), dtype=np.float64)
    h[valid] = sums[valid] / cnt[valid]
    base = float(np.median(h[valid]))
    h[~valid] = base
    h = smooth_circular_1d(h, window=int(max(3, int(smooth_minutes))))
    p50 = float(np.percentile(h, 50.0))
    p95 = float(np.percentile(h, 95.0))
    den = max(1.0e-9, p95 - p50)
    hz = np.clip((h - p50) / den, 0.0, 1.0).astype(np.float32)
    return np.asarray(hz, dtype=np.float32)


def lookup_calendar_hazard(unix: np.ndarray, hazard_profile: np.ndarray | None) -> np.ndarray:
    if hazard_profile is None:
        return np.zeros((int(np.asarray(unix).shape[0]),), dtype=np.float32)
    idx = calendar_minute_index_nonleap(np.asarray(unix, dtype=np.float64))
    return np.asarray(hazard_profile[idx], dtype=np.float32)


def build_hazard_weighted_prior_trig(
    unix: np.ndarray,
    cycle_trig: np.ndarray,
    clim_aligned_trig: np.ndarray,
    hazard_profile: np.ndarray | None,
    alpha_low: float,
    alpha_high: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cyc = np.asarray(cycle_trig, dtype=np.float64)
    cal = np.asarray(clim_aligned_trig, dtype=np.float64)
    h = np.asarray(lookup_calendar_hazard(unix=np.asarray(unix, dtype=np.float64), hazard_profile=hazard_profile), dtype=np.float64)
    g = float(max(0.1, float(gamma)))
    a0 = float(alpha_low)
    a1 = float(alpha_high)
    if a1 < a0:
        a0, a1 = a1, a0
    alpha = np.clip(a0 + (a1 - a0) * np.power(np.clip(h, 0.0, 1.0), g), 0.0, 1.0)
    mix = (1.0 - alpha[:, None]) * cyc + alpha[:, None] * cal
    out = hm.normalize_trig4_np(mix).astype(np.float32)
    return np.asarray(out, dtype=np.float32), np.asarray(alpha, dtype=np.float32), np.asarray(h, dtype=np.float32)


def build_hazard_residual_prior_trig(
    unix: np.ndarray,
    cycle_trig: np.ndarray,
    residual_template_trig: np.ndarray,
    hazard_profile: np.ndarray | None,
    alpha_low: float,
    alpha_high: float,
    gamma: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cyc = np.asarray(cycle_trig, dtype=np.float64)
    tpl = np.asarray(residual_template_trig, dtype=np.float64)
    h = np.asarray(lookup_calendar_hazard(unix=np.asarray(unix, dtype=np.float64), hazard_profile=hazard_profile), dtype=np.float64)
    g = float(max(0.1, float(gamma)))
    a0 = float(alpha_low)
    a1 = float(alpha_high)
    if a1 < a0:
        a0, a1 = a1, a0
    w = np.clip(a0 + (a1 - a0) * np.power(np.clip(h, 0.0, 1.0), g), 0.0, 1.0)
    pred = hm.normalize_trig4_np(cyc + float(gain) * w[:, None] * tpl).astype(np.float32)
    return np.asarray(pred, dtype=np.float32), np.asarray(w, dtype=np.float32), np.asarray(h, dtype=np.float32)


def build_hazard_hybrid_prior_trig(
    unix: np.ndarray,
    cycle_trig: np.ndarray,
    clim_aligned_trig: np.ndarray,
    residual_template_trig: np.ndarray,
    hazard_profile: np.ndarray | None,
    alpha_low: float,
    alpha_high: float,
    gamma: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cyc = np.asarray(cycle_trig, dtype=np.float64)
    cal = np.asarray(clim_aligned_trig, dtype=np.float64)
    tpl = np.asarray(residual_template_trig, dtype=np.float64)
    h = np.asarray(lookup_calendar_hazard(unix=np.asarray(unix, dtype=np.float64), hazard_profile=hazard_profile), dtype=np.float64)
    g = float(max(0.1, float(gamma)))
    a0 = float(alpha_low)
    a1 = float(alpha_high)
    if a1 < a0:
        a0, a1 = a1, a0
    w = np.clip(a0 + (a1 - a0) * np.power(np.clip(h, 0.0, 1.0), g), 0.0, 1.0)
    mix = (1.0 - w[:, None]) * cyc + w[:, None] * cal + float(gain) * w[:, None] * tpl
    pred = hm.normalize_trig4_np(mix).astype(np.float32)
    return np.asarray(pred, dtype=np.float32), np.asarray(w, dtype=np.float32), np.asarray(h, dtype=np.float32)


def build_cycle_residual_calendar_template_trig(
    sat_name: str,
    years: Sequence[int],
    truth_suffix: str,
    observer_lat: float,
    observer_lon: float,
    days: int,
    train_days: int,
    step_minutes: int,
    fit_days: float,
    cycle_min_period_minutes: float,
    cycle_max_period_minutes: float,
    episode_stride: int,
    max_files: int,
    smooth_minutes: int,
) -> tuple[np.ndarray | None, dict]:
    years_use = [int(y) for y in years]
    truth_cache = load_truth_cache(years_use, suffix=str(truth_suffix))
    if not truth_cache:
        return None, {"episodes": 0, "files": 0, "valid_bins": 0}
    files = collect_year_files(
        sat_name=str(sat_name),
        years=years_use,
        stride=int(max(1, int(episode_stride))),
        max_files=int(max_files),
    )
    sums = np.zeros((int(MINUTES_PER_YEAR), 4), dtype=np.float64)
    cnt = np.zeros((int(MINUTES_PER_YEAR),), dtype=np.float64)
    used = 0
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(sat_name),
            truth_cache=truth_cache,
            observer_lat=float(observer_lat),
            observer_lon=float(observer_lon),
            days=int(days),
            train_days=int(train_days),
            step_minutes=int(step_minutes),
        )
        if ep is None:
            continue
        cycle_trig, _, _, _, _, _ = build_cycle_repeat_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            fit_days=float(fit_days),
            min_period_minutes=float(cycle_min_period_minutes),
            max_period_minutes=float(cycle_max_period_minutes),
        )
        true_trig = np.asarray(hm.azel_to_trig4(np.asarray(ep.true_azel, dtype=np.float64)), dtype=np.float64)
        resid = np.asarray(true_trig - np.asarray(cycle_trig, dtype=np.float64), dtype=np.float64)
        idx = calendar_minute_index_nonleap(np.asarray(ep.unix, dtype=np.float64))
        for c in range(4):
            np.add.at(sums[:, c], idx, resid[:, c])
        np.add.at(cnt, idx, 1.0)
        used += 1
        if i % 25 == 0:
            print(f"[template] processed {i}/{len(files)} files used={used}", flush=True)

    valid = cnt > 0.0
    if not np.any(valid):
        return None, {"episodes": int(used), "files": int(len(files)), "valid_bins": 0}

    out = np.zeros((int(MINUTES_PER_YEAR), 4), dtype=np.float64)
    out[valid] = sums[valid] / cnt[valid, None]
    med = np.median(out[valid], axis=0).astype(np.float64)
    out[~valid] = med[None, :]
    for c in range(4):
        out[:, c] = smooth_circular_1d(out[:, c], window=int(max(3, int(smooth_minutes))))
    for c in range(4):
        lim = float(np.percentile(np.abs(out[:, c]), 99.5))
        if np.isfinite(lim) and lim > 1.0e-8:
            out[:, c] = np.clip(out[:, c], -lim, lim)
    return np.asarray(out, dtype=np.float32), {
        "episodes": int(used),
        "files": int(len(files)),
        "valid_bins": int(np.sum(valid)),
    }


def lookup_calendar_climatology_trig(unix: np.ndarray, clim_trig: np.ndarray) -> np.ndarray:
    if clim_trig is None:
        return np.zeros((int(np.asarray(unix).shape[0]), 4), dtype=np.float32)
    idx = calendar_minute_index_nonleap(unix)
    return np.asarray(clim_trig[idx], dtype=np.float32)


def fit_pair_rotation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    dot = float(np.sum(xx[:, 0] * yy[:, 0] + xx[:, 1] * yy[:, 1]))
    crs = float(np.sum(xx[:, 0] * yy[:, 1] - xx[:, 1] * yy[:, 0]))
    th = float(math.atan2(crs, dot))
    return float(math.cos(th)), float(math.sin(th)), float(np.rad2deg(th))


def rotate_pair(v: np.ndarray, c: float, s: float) -> np.ndarray:
    x = np.asarray(v[:, 0], dtype=np.float64)
    y = np.asarray(v[:, 1], dtype=np.float64)
    return np.column_stack([c * x - s * y, s * x + c * y]).astype(np.float64)


def build_climatology_aligned_prior_trig(
    unix: np.ndarray,
    base_azel: np.ndarray,
    clim_trig: np.ndarray,
    fit_days: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    n = int(u.shape[0])
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool), 0.0, 0.0
    base_trig = hm.azel_to_trig4(np.asarray(base_azel, dtype=np.float64)).astype(np.float64)
    clim = np.asarray(lookup_calendar_climatology_trig(u, np.asarray(clim_trig, dtype=np.float32)), dtype=np.float64)
    fit_mask = compute_teacher_mask(u, teacher_force_days=float(fit_days))
    idx = np.where(np.asarray(fit_mask, dtype=bool))[0]
    if idx.size < 8:
        idx = np.arange(min(n, 8), dtype=np.int64)
    c_az, s_az, az_deg = fit_pair_rotation(clim[idx, 0:2], base_trig[idx, 0:2])
    c_el, s_el, el_deg = fit_pair_rotation(clim[idx, 2:4], base_trig[idx, 2:4])
    out = np.zeros((n, 4), dtype=np.float64)
    out[:, 0:2] = rotate_pair(clim[:, 0:2], c=c_az, s=s_az)
    out[:, 2:4] = rotate_pair(clim[:, 2:4], c=c_el, s=s_el)
    out = hm.normalize_trig4_np(out).astype(np.float32)
    return out, np.asarray(fit_mask, dtype=bool), float(az_deg), float(el_deg)


def pick_truth_window(
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
    start_unix: int,
    end_unix: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    unix_parts = []
    full_parts = []
    for unix, full in cache.values():
        mask = (unix >= float(start_unix)) & (unix <= float(end_unix))
        if np.any(mask):
            unix_parts.append(unix[mask])
            full_parts.append(full[mask])
    if not unix_parts:
        return None, None
    unix = np.concatenate(unix_parts)
    full = np.concatenate(full_parts)
    order = np.argsort(unix)
    return unix[order], full[order]


def collect_year_files(
    sat_name: str,
    years: Sequence[int],
    stride: int,
    max_files: int = 0,
) -> list[Path]:
    bag: list[Path] = []
    for year in years:
        d = ROOT / f"{sat_name}_{int(year)}"
        if not d.exists():
            continue
        for path in sorted(d.glob("*.txt")):
            stamp = parse_tle_stem_flexible(path.stem)
            if stamp.year != int(year):
                continue
            bag.append(path)
    if int(stride) > 1:
        bag = [path for idx, path in enumerate(bag) if idx % int(stride) == 0]
    if int(max_files) > 0:
        bag = bag[: int(max_files)]
    return bag


def collect_tle_files(tle_dir: str, max_files: int, specific: Sequence[str]) -> list[Path]:
    if specific:
        out = []
        for rel in specific:
            p = ROOT / str(rel)
            if p.exists():
                out.append(p)
        return out
    files = sorted((ROOT / str(tle_dir)).glob("*.txt"))
    if int(max_files) > 0:
        files = files[: int(max_files)]
    return list(files)


def build_episode(
    tle_path: Path,
    sat_name: str,
    truth_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    observer_lat: float,
    observer_lon: float,
    days: int,
    train_days: int,
    step_minutes: int,
) -> Episode | None:
    window = exp.build_window(
        tle_path=tle_path,
        days=int(days),
        train_days=int(train_days),
        step_minutes=int(step_minutes),
    )
    truth_unix, truth_full = pick_truth_window(
        cache=truth_cache,
        start_unix=window.start_unix,
        end_unix=window.end_unix,
    )
    if truth_unix is None or truth_full is None:
        return None

    raw_azel, _ = exp.propagate_tle_azel_lla_at_unix(
        tle_path=tle_path,
        sat_name=sat_name,
        observer_lat=float(observer_lat),
        observer_lon=float(observer_lon),
        unix=window.unix,
    )
    pred_full = np.zeros((window.unix.shape[0], 5), dtype=np.float64)
    pred_full[:, 3:5] = raw_azel
    pred_aligned, true_aligned, unix_aligned = core.align_by_unix(window.unix, pred_full, truth_unix, truth_full)
    forecast_mask = np.asarray(unix_aligned >= float(window.train_end_unix), dtype=bool)
    expected_forecast = int(np.sum(np.asarray(window.unix >= float(window.train_end_unix), dtype=bool)))
    if int(np.sum(forecast_mask)) < expected_forecast:
        return None

    return Episode(
        tle_name=tle_path.name,
        unix=np.asarray(unix_aligned, dtype=np.float64),
        baseline_azel=np.asarray(pred_aligned[:, 3:5], dtype=np.float64),
        true_azel=np.asarray(true_aligned[:, 3:5], dtype=np.float64),
        forecast_mask=np.asarray(forecast_mask, dtype=bool),
        static_features=np.asarray(exp.tle_static_features(tle_path=tle_path, sat_name=sat_name), dtype=np.float64),
    )


def build_step_arrays(
    ep: Episode,
    days: int,
    time_yearly_harmonics: int,
    time_feature_mode: str,
    add_baseline_periodic_harmonics: int,
    pseudo_observe_days: float,
    periodic_ls_harmonics: int,
    teacher_force_days: float,
    warmup_weight: float,
    forecast_weight: float,
    use_extra_sincos_features: bool,
    use_cycle_repeat_prior: bool,
    cycle_observe_days: float,
    cycle_min_period_minutes: float,
    cycle_max_period_minutes: float,
    calendar_climatology_trig: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unix = np.asarray(ep.unix, dtype=np.float64)
    base = np.asarray(ep.baseline_azel, dtype=np.float64)
    n = int(base.shape[0])

    if str(time_feature_mode) == "linear":
        t_rel_lin = (unix - float(unix[0])) / float(int(days) * exp.SOLAR_DAY_SECONDS)
        time_feat = np.column_stack(
            [
                np.asarray(t_rel_lin, dtype=np.float32),
                np.asarray(t_rel_lin, dtype=np.float32) ** 2.0,
                np.asarray(t_rel_lin, dtype=np.float32) ** 3.0,
            ]
        ).astype(np.float32)
    else:
        time_feat = hm.build_time_cyclic_features_np(unix=unix, yearly_harmonics=int(time_yearly_harmonics))
    trig_base = hm.azel_to_trig4(base)
    unix_sin_raw = np.sin(np.asarray(unix, dtype=np.float64)).astype(np.float32).reshape(-1, 1)
    unix_cos_raw = np.cos(np.asarray(unix, dtype=np.float64)).astype(np.float32).reshape(-1, 1)
    clim_trig = lookup_calendar_climatology_trig(unix=unix, clim_trig=calendar_climatology_trig)
    prior_trig, fit_mask, fit_rmse_az, fit_rmse_el = build_periodic_drift_prior_trig(
        unix=unix,
        base_azel=base,
        fit_days=float(pseudo_observe_days),
        harmonics=int(periodic_ls_harmonics),
    )
    cycle_trig = np.asarray(trig_base, dtype=np.float32)
    cycle_period_minutes = float(hm.SECONDS_PER_SIDEREAL_DAY / 60.0)
    cycle_score = 0.0
    cycle_rmse_az = 0.0
    cycle_rmse_el = 0.0
    if bool(use_cycle_repeat_prior):
        cycle_trig, _, cycle_period_minutes, cycle_score, cycle_rmse_az, cycle_rmse_el = build_cycle_repeat_prior_trig(
            unix=unix,
            base_azel=base,
            fit_days=float(cycle_observe_days),
            min_period_minutes=float(cycle_min_period_minutes),
            max_period_minutes=float(cycle_max_period_minutes),
        )
    period_sec = float(max(1.0, cycle_period_minutes * 60.0))
    cycle_phase = 2.0 * math.pi * (unix - float(unix[0])) / period_sec
    teacher_mask = compute_teacher_mask(unix=unix, teacher_force_days=float(teacher_force_days))
    d_az = core.angle_diff_deg(base[:, 0], np.roll(base[:, 0], 1)) / 180.0
    d_el = (base[:, 1] - np.roll(base[:, 1], 1)) / 90.0
    if n > 1:
        d_az[0] = d_az[1]
        d_el[0] = d_el[1]
    else:
        d_az[0] = 0.0
        d_el[0] = 0.0
    t_rel = (unix - float(unix[0])) / float(int(days) * exp.SOLAR_DAY_SECONDS)
    static = np.repeat(np.asarray(ep.static_features, dtype=np.float32)[None, :], n, axis=0)
    forecast_flag = np.asarray(ep.forecast_mask, dtype=np.float32).reshape(-1, 1)
    extra_periodic = []
    h = int(max(0, int(add_baseline_periodic_harmonics)))
    if h > 0:
        az_rad = np.deg2rad(np.asarray(base[:, 0], dtype=np.float64))
        el_rad = np.deg2rad(np.asarray(base[:, 1], dtype=np.float64))
        for k in range(2, h + 1):
            extra_periodic.append(np.sin(float(k) * az_rad).astype(np.float32).reshape(-1, 1))
            extra_periodic.append(np.cos(float(k) * az_rad).astype(np.float32).reshape(-1, 1))
            extra_periodic.append(np.sin(float(k) * el_rad).astype(np.float32).reshape(-1, 1))
            extra_periodic.append(np.cos(float(k) * el_rad).astype(np.float32).reshape(-1, 1))

    x_parts = [
        time_feat.astype(np.float32),
        trig_base.astype(np.float32),
    ]
    if bool(use_extra_sincos_features):
        x_parts.extend(
            [
                unix_sin_raw.astype(np.float32),
                unix_cos_raw.astype(np.float32),
                trig_base.astype(np.float32),
            ]
        )
    x_parts.extend(
        [
        clim_trig.astype(np.float32),
        (trig_base - clim_trig).astype(np.float32),
        prior_trig.astype(np.float32),
        (trig_base - prior_trig).astype(np.float32),
        cycle_trig.astype(np.float32),
        (trig_base - cycle_trig).astype(np.float32),
        np.sin(cycle_phase).astype(np.float32).reshape(-1, 1),
        np.cos(cycle_phase).astype(np.float32).reshape(-1, 1),
        np.full((n, 1), float(cycle_period_minutes) / 1440.0, dtype=np.float32),
        np.full((n, 1), float(cycle_score), dtype=np.float32),
        np.full((n, 1), float(cycle_rmse_az), dtype=np.float32),
        np.full((n, 1), float(cycle_rmse_el), dtype=np.float32),
        *extra_periodic,
        d_az.astype(np.float32).reshape(-1, 1),
        d_el.astype(np.float32).reshape(-1, 1),
        t_rel.astype(np.float32).reshape(-1, 1),
        forecast_flag,
        np.asarray(teacher_mask, dtype=np.float32).reshape(-1, 1),
        np.asarray(fit_mask, dtype=np.float32).reshape(-1, 1),
        np.full((n, 1), float(fit_rmse_az), dtype=np.float32),
        np.full((n, 1), float(fit_rmse_el), dtype=np.float32),
        static.astype(np.float32),
        ]
    )
    x = np.concatenate(x_parts, axis=1).astype(np.float32)
    y = hm.azel_to_trig4(np.asarray(ep.true_azel, dtype=np.float64)).astype(np.float32)
    w = np.where(np.asarray(ep.forecast_mask, dtype=bool), float(forecast_weight), float(warmup_weight)).astype(np.float32)
    return x, y, w


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    seq_len: int,
    seq_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    L = int(seq_len)
    S = int(max(1, seq_stride))
    if n < L:
        return (
            np.zeros((0, L, x.shape[1]), dtype=np.float32),
            np.zeros((0, L, y.shape[1]), dtype=np.float32),
            np.zeros((0, L), dtype=np.float32),
        )
    starts = list(range(0, n - L + 1, S))
    tail = n - L
    if starts[-1] != tail:
        starts.append(tail)
    xs = []
    ys = []
    ws = []
    for s in starts:
        e = s + L
        xs.append(np.asarray(x[s:e], dtype=np.float32))
        ys.append(np.asarray(y[s:e], dtype=np.float32))
        ws.append(np.asarray(w[s:e], dtype=np.float32))
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.stack(ys, axis=0).astype(np.float32),
        np.stack(ws, axis=0).astype(np.float32),
    )


def normalize_trig4_seq_tf(y):
    tf = core.tf
    az = y[:, :, 0:2]
    el = y[:, :, 2:4]
    az_norm = tf.sqrt(tf.maximum(1.0e-9, tf.reduce_sum(tf.square(az), axis=-1, keepdims=True)))
    el_norm = tf.sqrt(tf.maximum(1.0e-9, tf.reduce_sum(tf.square(el), axis=-1, keepdims=True)))
    return tf.concat([az / az_norm, el / el_norm], axis=-1)


def build_packed_loss(
    angle_weight: float,
    softmax_weight: float,
    softmax_temp: float,
    worstk_weight: float,
    worstk_frac: float,
    uncertainty_weight: float,
    uncertainty_target_scale: float,
):
    tf = core.tf
    huber = tf.keras.losses.Huber(delta=0.03, reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(y_true_pack, y_pred_pack):
        y_true = tf.cast(y_true_pack[:, :, 0:4], tf.float32)
        step_weight = tf.cast(y_true_pack[:, :, 4], tf.float32)
        y_pred = tf.cast(y_pred_pack[:, :, 0:4], tf.float32)
        uncertainty = tf.clip_by_value(tf.cast(y_pred_pack[:, :, 4], tf.float32), 1.0e-4, 1.0 - 1.0e-4)

        hub = huber(y_true, y_pred)
        step_mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)

        yt = normalize_trig4_seq_tf(y_true)
        yp = normalize_trig4_seq_tf(y_pred)
        az_cos = tf.clip_by_value(tf.reduce_sum(yt[:, :, 0:2] * yp[:, :, 0:2], axis=-1), -1.0, 1.0)
        el_cos = tf.clip_by_value(tf.reduce_sum(yt[:, :, 2:4] * yp[:, :, 2:4], axis=-1), -1.0, 1.0)
        ang = 0.5 * (1.0 - az_cos) + 0.5 * (1.0 - el_cos)

        unc_target = tf.clip_by_value(step_mae / float(max(1.0e-6, float(uncertainty_target_scale))), 0.0, 1.0)
        unc_bce = -(unc_target * tf.math.log(uncertainty) + (1.0 - unc_target) * tf.math.log(1.0 - uncertainty))

        core_step = hub + float(angle_weight) * ang + float(uncertainty_weight) * unc_bce
        w = tf.clip_by_value(step_weight, 1.0e-5, 1.0e6)
        wsum = tf.reduce_sum(w, axis=1) + 1.0e-6
        mean_core = tf.reduce_sum(core_step * w, axis=1) / wsum

        scaled = step_mae * (0.25 + w)
        att = tf.nn.softmax(float(softmax_temp) * scaled, axis=1)
        softmax_max = tf.reduce_sum(att * scaled, axis=1)

        seq_len = tf.cast(tf.shape(scaled)[1], tf.float32)
        k = tf.cast(tf.maximum(1.0, tf.math.round(float(worstk_frac) * seq_len)), tf.int32)
        topk = tf.nn.top_k(scaled, k=k, sorted=False).values
        worstk = tf.reduce_mean(topk, axis=1)

        total = mean_core + float(softmax_weight) * softmax_max + float(worstk_weight) * worstk
        return tf.reduce_mean(total)

    return loss_fn


def packed_mae_metric(y_true_pack, y_pred_pack):
    tf = core.tf
    y_true = tf.cast(y_true_pack[:, :, 0:4], tf.float32)
    step_weight = tf.cast(y_true_pack[:, :, 4], tf.float32)
    y_pred = tf.cast(y_pred_pack[:, :, 0:4], tf.float32)
    step_mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
    w = tf.clip_by_value(step_weight, 1.0e-5, 1.0e6)
    return tf.reduce_sum(step_mae * w) / (tf.reduce_sum(w) + 1.0e-6)


def build_lstm_model(
    seq_len: int,
    feat_dim: int,
    trig_offset: int,
    trig_mean: np.ndarray,
    trig_std: np.ndarray,
    lstm_units_1: int,
    lstm_units_2: int,
    dense_units: int,
    dropout: float,
    delta_scale: float,
    lr: float,
    loss_angle_weight: float,
    loss_softmax_weight: float,
    loss_softmax_temp: float,
    loss_worstk_weight: float,
    loss_worstk_frac: float,
    loss_uncertainty_weight: float,
    uncertainty_target_scale: float,
):
    tf = core.tf
    inp = tf.keras.Input(shape=(int(seq_len), int(feat_dim)), dtype=tf.float32)
    base_trig_norm = inp[:, :, int(trig_offset) : int(trig_offset) + 4]
    trig_mean_tf = tf.constant(np.asarray(trig_mean, dtype=np.float32).reshape(1, 1, 4), dtype=tf.float32)
    trig_std_tf = tf.constant(np.asarray(trig_std, dtype=np.float32).reshape(1, 1, 4), dtype=tf.float32)
    base_trig = base_trig_norm * trig_std_tf + trig_mean_tf
    x = tf.keras.layers.LayerNormalization()(inp)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(int(lstm_units_1), return_sequences=True)
    )(x)
    if float(dropout) > 0.0:
        x = tf.keras.layers.Dropout(float(dropout))(x)
    if int(lstm_units_2) > 0:
        x = tf.keras.layers.LSTM(int(lstm_units_2), return_sequences=True)(x)
        if float(dropout) > 0.0:
            x = tf.keras.layers.Dropout(float(dropout))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(int(dense_units), activation="gelu"))(x)
    raw = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation="tanh"))(x)
    delta = tf.keras.layers.Rescaling(scale=float(delta_scale), name="delta_scaled")(raw)
    pred = tf.keras.layers.Add(name="trig_plus_delta")([base_trig, delta])
    pred = tf.keras.layers.Lambda(normalize_trig4_seq_tf, name="norm_trig4")(pred)
    unc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"), name="uncertainty_head")(x)
    out_pack = tf.keras.layers.Concatenate(axis=-1, name="pred_with_uncertainty")([pred, unc])
    model = tf.keras.Model(inp, out_pack)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss=build_packed_loss(
            angle_weight=float(loss_angle_weight),
            softmax_weight=float(loss_softmax_weight),
            softmax_temp=float(loss_softmax_temp),
            worstk_weight=float(loss_worstk_weight),
            worstk_frac=float(loss_worstk_frac),
            uncertainty_weight=float(loss_uncertainty_weight),
            uncertainty_target_scale=float(uncertainty_target_scale),
        ),
        metrics=[packed_mae_metric],
    )
    return model


def predict_full_pack(
    model,
    x_norm: np.ndarray,
    seq_len: int,
    seq_stride: int,
    predict_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(x_norm.shape[0])
    L = int(seq_len)
    S = int(max(1, seq_stride))
    if n < L:
        pad = np.zeros((L, x_norm.shape[1]), dtype=np.float32)
        pad[:n] = np.asarray(x_norm, dtype=np.float32)
        pred_pack = np.asarray(model.predict(pad[None, ...], verbose=0)[0], dtype=np.float64)
        pred_trig = hm.normalize_trig4_np(pred_pack[:n, 0:4]).astype(np.float32)
        pred_unc = np.clip(np.asarray(pred_pack[:n, 4], dtype=np.float64), 0.0, 1.0).astype(np.float32)
        return pred_trig, pred_unc
    starts = list(range(0, n - L + 1, S))
    tail = n - L
    if starts[-1] != tail:
        starts.append(tail)
    acc = np.zeros((n, 5), dtype=np.float64)
    cnt = np.zeros((n, 1), dtype=np.float64)
    batch = np.stack([np.asarray(x_norm[s : s + L], dtype=np.float32) for s in starts], axis=0).astype(np.float32)
    pred_batch = np.asarray(
        model.predict(batch, verbose=0, batch_size=int(max(1, int(predict_batch_size)))),
        dtype=np.float64,
    )
    for idx, s in enumerate(starts):
        e = s + L
        acc[s:e] += pred_batch[idx]
        cnt[s:e] += 1.0
    out = acc / np.maximum(cnt, 1.0)
    pred_trig = hm.normalize_trig4_np(np.asarray(out[:, 0:4], dtype=np.float64)).astype(np.float32)
    pred_unc = np.clip(np.asarray(out[:, 4], dtype=np.float64), 0.0, 1.0).astype(np.float32)
    return pred_trig, pred_unc


def compute_forecast_max_abs_error(true_azel: np.ndarray, pred_azel: np.ndarray, forecast_mask: np.ndarray) -> float:
    t = np.zeros((int(np.sum(forecast_mask)), 5), dtype=np.float64)
    p = np.zeros((int(np.sum(forecast_mask)), 5), dtype=np.float64)
    t[:, 3:5] = np.asarray(true_azel, dtype=np.float64)[np.asarray(forecast_mask, dtype=bool)]
    p[:, 3:5] = np.asarray(pred_azel, dtype=np.float64)[np.asarray(forecast_mask, dtype=bool)]
    return float(core.compute_metrics_azel(t, p)["overall"]["max_abs_error_max"])


def build_pred_full_from_azel(
    pred_azel: np.ndarray,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    geo_radius_km: float,
    fallback_lla: np.ndarray | None = None,
) -> np.ndarray:
    azel = np.asarray(pred_azel, dtype=np.float64)
    pred_lla = np.asarray(
        core.azel_to_lla_geoshell(
            az_deg=azel[:, 0],
            el_deg=azel[:, 1],
            observer_lat_deg=float(observer_lat),
            observer_lon_deg=float(observer_lon),
            observer_alt_m=float(observer_alt_m),
            geo_radius_km=float(geo_radius_km),
        ),
        dtype=np.float64,
    )
    if fallback_lla is not None:
        fb = np.asarray(fallback_lla, dtype=np.float64)
        if fb.shape == pred_lla.shape:
            bad = ~np.all(np.isfinite(pred_lla), axis=1)
            if np.any(bad):
                pred_lla[bad] = fb[bad]
    out = np.zeros((azel.shape[0], 5), dtype=np.float64)
    out[:, 0:3] = pred_lla
    out[:, 3:5] = azel
    return out


def build_full_error_arrays(y_true_full: np.ndarray, y_pred_full: np.ndarray) -> dict[str, np.ndarray]:
    true_full = np.asarray(y_true_full, dtype=np.float64)
    pred_full = np.asarray(y_pred_full, dtype=np.float64)
    return {
        "AZ": np.asarray(core.angle_diff_deg(pred_full[:, 3], true_full[:, 3]), dtype=np.float64),
        "EL": np.asarray(pred_full[:, 4] - true_full[:, 4], dtype=np.float64),
        "Lat": np.asarray(pred_full[:, 0] - true_full[:, 0], dtype=np.float64),
        "Lon": np.asarray(core.angle_diff_deg(pred_full[:, 1], true_full[:, 1]), dtype=np.float64),
        "Alt": np.asarray(pred_full[:, 2] - true_full[:, 2], dtype=np.float64),
    }


def plot_full_target_compare_and_error(
    unix: np.ndarray,
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
    out_compare_png: Path,
    out_error_png: Path,
    max_points: int,
) -> list[Path]:
    if getattr(core, "plt", None) is None:
        return []
    t_all = (np.asarray(np.round(unix), dtype=np.int64) + int(core.JST_OFFSET_SEC)).astype("datetime64[s]")
    n = int(t_all.shape[0])
    stride = int(max(1, math.ceil(float(n) / float(max(1, int(max_points))))))
    idx = np.arange(0, n, stride, dtype=np.int64)
    if idx.size == 0:
        return []
    t = t_all[idx]
    y_true = np.asarray(y_true_full, dtype=np.float64)[idx]
    y_pred = np.asarray(y_pred_full, dtype=np.float64)[idx]
    err = build_full_error_arrays(y_true, y_pred)

    targets = [
        ("AZ", 3, "deg", True),
        ("EL", 4, "deg", False),
        ("Lat", 0, "deg", False),
        ("Lon", 1, "deg", True),
        ("Alt", 2, "km", False),
    ]

    fig_cmp, axes_cmp = core.plt.subplots(len(targets), 1, figsize=(16, 16), sharex=True)
    for ax, (name, col, unit, is_angle) in zip(axes_cmp, targets):
        if is_angle:
            yt = np.asarray(core.wrap180(y_true[:, col]), dtype=np.float64)
            yp = np.asarray(core.wrap180(y_pred[:, col]), dtype=np.float64)
        else:
            yt = np.asarray(y_true[:, col], dtype=np.float64)
            yp = np.asarray(y_pred[:, col], dtype=np.float64)
        ax.plot(t, yt, linewidth=0.9, label=f"True {name}")
        ax.plot(t, yp, linewidth=0.9, label=f"Pred {name}")
        ax.set_ylabel(f"{name} ({unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes_cmp[-1].set_xlabel("Local time (JST)")
    fig_cmp.suptitle("Prediction vs Truth (AZ/EL/Lat/Lon/Alt)")
    fig_cmp.tight_layout(rect=[0, 0, 1, 0.98])
    out_compare_png.parent.mkdir(parents=True, exist_ok=True)
    fig_cmp.savefig(out_compare_png, dpi=160)
    core.plt.close(fig_cmp)

    fig_err, axes_err = core.plt.subplots(len(targets), 1, figsize=(16, 16), sharex=True)
    for ax, (name, _, unit, _) in zip(axes_err, targets):
        e = np.asarray(err[name], dtype=np.float64)
        ax.plot(t, e, linewidth=0.9, label=f"{name} error")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(f"{name} err ({unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes_err[-1].set_xlabel("Local time (JST)")
    fig_err.suptitle("Prediction Error vs Truth (AZ/EL/Lat/Lon/Alt)")
    fig_err.tight_layout(rect=[0, 0, 1, 0.98])
    out_error_png.parent.mkdir(parents=True, exist_ok=True)
    fig_err.savefig(out_error_png, dpi=160)
    core.plt.close(fig_err)

    return [out_compare_png, out_error_png]


def recompute_azel_from_lla_np(
    sat_lla: np.ndarray,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
) -> np.ndarray:
    lla = np.asarray(sat_lla, dtype=np.float64)
    sx, sy, sz = core.geodetic_to_ecef_km(lla[:, 0], lla[:, 1], lla[:, 2])
    ox, oy, oz = core.geodetic_to_ecef_km(
        np.asarray([float(observer_lat)], dtype=np.float64),
        np.asarray([float(observer_lon)], dtype=np.float64),
        np.asarray([float(observer_alt_m) / 1000.0], dtype=np.float64),
    )
    dx = sx - float(ox[0])
    dy = sy - float(oy[0])
    dz = sz - float(oz[0])

    lat0 = math.radians(float(observer_lat))
    lon0 = math.radians(float(observer_lon))
    sl = math.sin(lat0)
    cl = math.cos(lat0)
    so = math.sin(lon0)
    co = math.cos(lon0)

    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz

    az = core.wrap360(np.rad2deg(np.arctan2(e, n)))
    el = np.clip(np.rad2deg(np.arctan2(u, np.sqrt(np.maximum(1.0e-12, e * e + n * n)))), -90.0, 90.0)
    return np.column_stack([az, el]).astype(np.float64)


def build_lla_residual_features(
    unix: np.ndarray,
    baseline_azel: np.ndarray,
    pred_azel: np.ndarray,
    baseline_lla: np.ndarray,
    pred_lla_geo: np.ndarray,
    static_features: np.ndarray,
    yearly_harmonics: int,
) -> np.ndarray:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    base_az = np.asarray(baseline_azel, dtype=np.float64)
    pred_az = np.asarray(pred_azel, dtype=np.float64)
    base_lla = np.asarray(baseline_lla, dtype=np.float64)
    pred_lla = np.asarray(pred_lla_geo, dtype=np.float64)
    n = int(u.shape[0])
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32)

    time_feat = hm.build_time_cyclic_features_np(unix=u, yearly_harmonics=int(max(0, int(yearly_harmonics))))
    trig_base = hm.azel_to_trig4(base_az).astype(np.float32)
    trig_pred = hm.azel_to_trig4(pred_az).astype(np.float32)

    lon_b = np.deg2rad(base_lla[:, 1])
    lon_p = np.deg2rad(pred_lla[:, 1])
    d_az = (core.angle_diff_deg(pred_az[:, 0], base_az[:, 0]) / 180.0).astype(np.float32).reshape(-1, 1)
    d_el = ((pred_az[:, 1] - base_az[:, 1]) / 90.0).astype(np.float32).reshape(-1, 1)
    d_lat = (pred_lla[:, 0] - base_lla[:, 0]).astype(np.float32).reshape(-1, 1)
    d_lon = core.angle_diff_deg(pred_lla[:, 1], base_lla[:, 1]).astype(np.float32).reshape(-1, 1)
    d_alt = (pred_lla[:, 2] - base_lla[:, 2]).astype(np.float32).reshape(-1, 1)

    static = np.repeat(np.asarray(static_features, dtype=np.float32).reshape(1, -1), n, axis=0)
    x_parts = [
        time_feat.astype(np.float32),
        trig_base.astype(np.float32),
        trig_pred.astype(np.float32),
        (base_lla[:, 0:1] / 90.0).astype(np.float32),
        np.sin(lon_b).astype(np.float32).reshape(-1, 1),
        np.cos(lon_b).astype(np.float32).reshape(-1, 1),
        (base_lla[:, 2:3] / 42164.0).astype(np.float32),
        (pred_lla[:, 0:1] / 90.0).astype(np.float32),
        np.sin(lon_p).astype(np.float32).reshape(-1, 1),
        np.cos(lon_p).astype(np.float32).reshape(-1, 1),
        (pred_lla[:, 2:3] / 42164.0).astype(np.float32),
        d_az,
        d_el,
        (d_lat / 5.0).astype(np.float32),
        (d_lon / 5.0).astype(np.float32),
        (d_alt / 20.0).astype(np.float32),
        static.astype(np.float32),
    ]
    return np.concatenate(x_parts, axis=1).astype(np.float32)


def build_lla_stage2_features(
    x_lla_feat: np.ndarray,
    max_base_features: int,
    pair_features: int,
) -> np.ndarray:
    x = np.asarray(x_lla_feat, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    q = int(max(1, min(int(max_base_features), int(x.shape[1]))))
    base = np.asarray(x[:, :q], dtype=np.float64)
    parts = [base, np.square(base)]
    p = int(max(1, min(int(pair_features), q)))
    if p >= 2:
        pair_cols = []
        for i in range(p):
            for j in range(i + 1, p):
                pair_cols.append((base[:, i] * base[:, j]).reshape(-1, 1))
        if pair_cols:
            parts.append(np.concatenate(pair_cols, axis=1).astype(np.float64))
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_linear_multi_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> dict:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.ndim != 2 or yy.ndim != 2 or xx.shape[0] != yy.shape[0] or xx.shape[0] == 0:
        raise RuntimeError("Invalid training matrix for linear ridge model")
    x_mean = np.mean(xx, axis=0)
    x_std = np.std(xx, axis=0)
    x_std = np.where(x_std < 1.0e-6, 1.0, x_std)
    xn = (xx - x_mean[None, :]) / x_std[None, :]
    phi = np.concatenate([np.ones((xn.shape[0], 1), dtype=np.float64), xn], axis=1)
    d = int(phi.shape[1])
    reg = np.eye(d, dtype=np.float64) * float(max(0.0, float(ridge)))
    reg[0, 0] = 0.0
    a = phi.T @ phi + reg
    b = phi.T @ yy
    w = np.linalg.solve(a, b)
    return {
        "x_mean": np.asarray(x_mean, dtype=np.float64).tolist(),
        "x_std": np.asarray(x_std, dtype=np.float64).tolist(),
        "weights": np.asarray(w, dtype=np.float64).tolist(),
        "feature_dim": int(xx.shape[1]),
        "target_dim": int(yy.shape[1]),
    }


def predict_linear_multi_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float64)
    x_mean = np.asarray(model["x_mean"], dtype=np.float64).reshape(1, -1)
    x_std = np.asarray(model["x_std"], dtype=np.float64).reshape(1, -1)
    w = np.asarray(model["weights"], dtype=np.float64)
    xn = (xx - x_mean) / np.where(np.abs(x_std) < 1.0e-9, 1.0, x_std)
    phi = np.concatenate([np.ones((xn.shape[0], 1), dtype=np.float64), xn], axis=1)
    return np.asarray(phi @ w, dtype=np.float64)


def apply_lla_delta_with_alpha(
    pred_lla: np.ndarray,
    delta_lla: np.ndarray,
    alpha_lat: float,
    alpha_lon: float,
    alpha_alt: float,
) -> np.ndarray:
    p = np.asarray(pred_lla, dtype=np.float64)
    d = np.asarray(delta_lla, dtype=np.float64)
    out = np.asarray(p, dtype=np.float64).copy()
    out[:, 0] = p[:, 0] + float(alpha_lat) * d[:, 0]
    out[:, 1] = core.wrap180(p[:, 1] + float(alpha_lon) * d[:, 1])
    out[:, 2] = p[:, 2] + float(alpha_alt) * d[:, 2]
    return out


def anchor_delta_with_warmup(
    delta_lla: np.ndarray,
    unix: np.ndarray,
    teacher_force_days: float,
    anchor_scale: float,
) -> np.ndarray:
    d = np.asarray(delta_lla, dtype=np.float64)
    if d.shape[0] == 0 or float(anchor_scale) <= 0.0:
        return d
    warm = compute_teacher_mask(unix=np.asarray(unix, dtype=np.float64), teacher_force_days=float(teacher_force_days))
    if not np.any(np.asarray(warm, dtype=bool)):
        return d
    mu = np.mean(np.asarray(d[np.asarray(warm, dtype=bool)], dtype=np.float64), axis=0)
    return np.asarray(d - float(anchor_scale) * mu.reshape(1, -1), dtype=np.float64)


def build_full_from_lla_azel(lla: np.ndarray, azel: np.ndarray) -> np.ndarray:
    out = np.zeros((int(np.asarray(lla).shape[0]), 5), dtype=np.float64)
    out[:, 0:3] = np.asarray(lla, dtype=np.float64)
    out[:, 3:5] = np.asarray(azel, dtype=np.float64)
    return out


def build_cycle_repeat_prior_azel_for_episode(
    ep: Episode,
    observe_days: float,
    min_period_minutes: float,
    max_period_minutes: float,
) -> tuple[np.ndarray, float, float, float, float]:
    trig, _, pmin, score, rmse_az, rmse_el = build_cycle_repeat_prior_trig(
        unix=np.asarray(ep.unix, dtype=np.float64),
        base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
        fit_days=float(observe_days),
        min_period_minutes=float(min_period_minutes),
        max_period_minutes=float(max_period_minutes),
    )
    return (
        np.asarray(hm.trig4_to_azel(trig), dtype=np.float64),
        float(pmin),
        float(score),
        float(rmse_az),
        float(rmse_el),
    )


def blend_azel_unitvec(base_azel: np.ndarray, alt_azel: np.ndarray, alpha: float) -> np.ndarray:
    ua = exp.build_unit_vector_targets_from_azel(np.asarray(base_azel, dtype=np.float64))
    ub = exp.build_unit_vector_targets_from_azel(np.asarray(alt_azel, dtype=np.float64))
    return exp.decode_unit_vector_targets_to_azel((1.0 - float(alpha)) * ua + float(alpha) * ub)


def blend_azel_unitvec_vector(base_azel: np.ndarray, alt_azel: np.ndarray, alpha_vec: np.ndarray) -> np.ndarray:
    ua = np.asarray(exp.build_unit_vector_targets_from_azel(np.asarray(base_azel, dtype=np.float64)), dtype=np.float64)
    ub = np.asarray(exp.build_unit_vector_targets_from_azel(np.asarray(alt_azel, dtype=np.float64)), dtype=np.float64)
    alpha = np.asarray(alpha_vec, dtype=np.float64).reshape(-1, 1)
    uv = (1.0 - alpha) * ua + alpha * ub
    uv_norm = np.linalg.norm(uv, axis=1, keepdims=True)
    uv = uv / np.maximum(uv_norm, 1.0e-9)
    return np.asarray(exp.decode_unit_vector_targets_to_azel(uv), dtype=np.float64)


def wrap360_np(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    out = np.mod(v, 360.0)
    out = np.where(out < 0.0, out + 360.0, out)
    return np.asarray(out, dtype=np.float64)


def angle_diff_deg_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    d = (aa - bb + 180.0) % 360.0 - 180.0
    return np.asarray(d, dtype=np.float64)


def build_reid_feature_matrix(
    unix: np.ndarray,
    sidereal_harmonics: int,
    solar_harmonics: int,
) -> np.ndarray:
    u = np.asarray(unix, dtype=np.float64).reshape(-1)
    n = int(u.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    rel_day = (u - float(u[0])) / float(exp.SOLAR_DAY_SECONDS)
    sid = 2.0 * math.pi * (u - float(u[0])) / float(hm.SECONDS_PER_SIDEREAL_DAY)
    day = 2.0 * math.pi * (u - float(u[0])) / float(exp.SOLAR_DAY_SECONDS)
    cols = [
        np.ones((n, 1), dtype=np.float64),
        rel_day.reshape(-1, 1),
        (rel_day**2.0).reshape(-1, 1),
    ]
    for k in range(1, int(max(1, sidereal_harmonics)) + 1):
        kk = float(k)
        cols.extend([np.sin(kk * sid).reshape(-1, 1), np.cos(kk * sid).reshape(-1, 1)])
    for k in range(1, int(max(0, solar_harmonics)) + 1):
        kk = float(k)
        cols.extend([np.sin(kk * day).reshape(-1, 1), np.cos(kk * day).reshape(-1, 1)])
    return np.concatenate(cols, axis=1).astype(np.float64)


def build_online_reid_prediction(
    ep: Episode,
    observe_days: float,
    mode: str,
    sidereal_harmonics: int,
    solar_harmonics: int,
    ridge_lambda: float,
    forgetting: float,
    ar_coeff: float,
) -> tuple[np.ndarray, dict]:
    m = str(mode).strip().lower()
    if m not in {"obs", "self"}:
        raise ValueError(f"Invalid reid mode: {mode}")
    unix = np.asarray(ep.unix, dtype=np.float64).reshape(-1)
    base = np.asarray(ep.baseline_azel, dtype=np.float64)
    true = np.asarray(ep.true_azel, dtype=np.float64)
    base_trig = np.asarray(hm.azel_to_trig4(base), dtype=np.float64)
    true_trig = np.asarray(hm.azel_to_trig4(true), dtype=np.float64)
    n = int(unix.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), {
            "mode": m,
            "observe_count": 0,
            "updates": 0,
            "fit_rmse_az": float("nan"),
            "fit_rmse_el": float("nan"),
        }

    phi = build_reid_feature_matrix(
        unix=unix,
        sidereal_harmonics=int(sidereal_harmonics),
        solar_harmonics=int(solar_harmonics),
    )
    dim = int(phi.shape[1])
    lam = float(max(1.0e-6, float(ridge_lambda)))
    forget = float(np.clip(float(forgetting), 0.90, 1.0))
    ar = float(np.clip(float(ar_coeff), 0.0, 0.999))
    pmat = np.eye(dim, dtype=np.float64) / lam
    theta = np.zeros((dim, 4), dtype=np.float64)
    obs_mask = np.asarray(compute_teacher_mask(unix=unix, teacher_force_days=float(observe_days)), dtype=bool)
    pred_azel = np.zeros((n, 2), dtype=np.float64)
    res_prev = np.zeros((4,), dtype=np.float64)
    updates = 0

    for i in range(n):
        row = np.asarray(phi[i], dtype=np.float64).reshape(-1)
        base_res = np.asarray(row @ theta, dtype=np.float64)
        res_hat = np.asarray(base_res + ar * res_prev, dtype=np.float64)
        pred_trig_i = np.asarray(hm.normalize_trig4_np(base_trig[i : i + 1] + res_hat[None, :]), dtype=np.float64)[0]
        pred_azel[i] = np.asarray(hm.trig4_to_azel(pred_trig_i[None, :]), dtype=np.float64)[0]

        if bool(obs_mask[i]):
            if m == "obs":
                meas_res = np.asarray(true_trig[i] - base_trig[i], dtype=np.float64)
            else:
                meas_res = np.asarray(pred_trig_i - base_trig[i], dtype=np.float64)

            target = np.asarray(meas_res - ar * res_prev, dtype=np.float64)
            err = np.asarray(target - base_res, dtype=np.float64)
            row_col = row.reshape(-1, 1)
            denom = float(forget + float((row_col.T @ pmat @ row_col)[0, 0]))
            if not np.isfinite(denom) or denom <= 1.0e-12:
                denom = 1.0e-12
            gain = np.asarray((pmat @ row_col).reshape(-1) / denom, dtype=np.float64)
            theta = np.asarray(theta + gain[:, None] * err[None, :], dtype=np.float64)
            pmat = np.asarray((pmat - np.outer(gain, row) @ pmat) / forget, dtype=np.float64)
            pmat = np.asarray(0.5 * (pmat + pmat.T), dtype=np.float64)
            updates += 1
            res_prev = np.asarray(meas_res, dtype=np.float64)
        else:
            res_prev = np.asarray(res_hat, dtype=np.float64)

    if int(np.sum(obs_mask)) > 0:
        fit_daz = angle_diff_deg_np(
            np.asarray(true[:, 0], dtype=np.float64)[obs_mask],
            np.asarray(pred_azel[:, 0], dtype=np.float64)[obs_mask],
        )
        fit_del = (
            np.asarray(true[:, 1], dtype=np.float64)[obs_mask]
            - np.asarray(pred_azel[:, 1], dtype=np.float64)[obs_mask]
        )
        fit_rmse_az = float(np.sqrt(np.mean(np.square(fit_daz))))
        fit_rmse_el = float(np.sqrt(np.mean(np.square(fit_del))))
    else:
        fit_rmse_az = float("nan")
        fit_rmse_el = float("nan")

    return np.asarray(pred_azel, dtype=np.float64), {
        "mode": m,
        "observe_count": int(np.sum(obs_mask)),
        "updates": int(updates),
        "fit_rmse_az": float(fit_rmse_az),
        "fit_rmse_el": float(fit_rmse_el),
    }


def search_best_blend_alpha(items: Sequence[tuple[Episode, np.ndarray]]) -> tuple[float, list[dict]]:
    alpha_grid = [i / 20.0 for i in range(21)]
    scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, pred_azel in items:
            blended = blend_azel_unitvec(
                np.asarray(ep.baseline_azel, dtype=np.float64),
                np.asarray(pred_azel, dtype=np.float64),
                alpha=float(alpha),
            )
            errs.append(
                compute_forecast_max_abs_error(
                    np.asarray(ep.true_azel, dtype=np.float64),
                    np.asarray(blended, dtype=np.float64),
                    np.asarray(ep.forecast_mask, dtype=bool),
                )
            )
        scores.append({"alpha": float(alpha), "summary": summarize(errs)})
    scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    if not scores:
        return 0.0, []
    return float(scores[0]["alpha"]), scores


def apply_dynamic_uncertainty_blend(
    base_azel: np.ndarray,
    pred_azel: np.ndarray,
    uncertainty: np.ndarray,
    teacher_mask: np.ndarray,
    alpha_floor: float,
    alpha_power: float,
    uncertainty_smooth: int,
    use_ruptures_alpha_seg: bool = False,
    ruptures_model: str = "rbf",
    ruptures_penalty: float = 8.0,
    ruptures_min_size: int = 24,
    ruptures_jump: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    unc = np.clip(smooth_1d(unc, window=int(uncertainty_smooth)), 0.0, 1.0)
    if bool(use_ruptures_alpha_seg):
        base_trig = hm.azel_to_trig4(np.asarray(base_azel, dtype=np.float64))
        pred_trig = hm.azel_to_trig4(np.asarray(pred_azel, dtype=np.float64))
        res_mag = np.sqrt(np.sum(np.square(pred_trig - base_trig), axis=1))
        norm = float(np.percentile(res_mag, 90.0))
        if not np.isfinite(norm) or norm < 1.0e-9:
            norm = 1.0
        res01 = np.clip(res_mag / norm, 0.0, 1.0)
        risk = np.clip(0.7 * unc + 0.3 * res01, 0.0, 1.0)
        risk_seg = segment_by_pelt_median(
            values=risk,
            model=str(ruptures_model),
            pen=float(ruptures_penalty),
            min_size=int(ruptures_min_size),
            jump=int(ruptures_jump),
        )
        unc = np.clip(0.5 * unc + 0.5 * np.asarray(risk_seg, dtype=np.float64), 0.0, 1.0)
    conf = np.power(np.clip(1.0 - unc, 0.0, 1.0), float(max(0.1, float(alpha_power))))
    alpha = float(alpha_floor) + (1.0 - float(alpha_floor)) * conf
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.where(np.asarray(teacher_mask, dtype=bool), 0.0, alpha)
    blended = blend_azel_unitvec_vector(base_azel=base_azel, alt_azel=pred_azel, alpha_vec=alpha)
    return blended, alpha.astype(np.float64), unc.astype(np.float64)


def build_episodes(
    paths: list[Path],
    sat_name: str,
    truth_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    observer_lat: float,
    observer_lon: float,
    days: int,
    train_days: int,
    step_minutes: int,
    tag: str,
) -> list[Episode]:
    out = []
    for i, p in enumerate(paths, start=1):
        print(f"[{tag} {i}/{len(paths)}] {p.name}", flush=True)
        ep = build_episode(
            tle_path=p,
            sat_name=sat_name,
            truth_cache=truth_cache,
            observer_lat=float(observer_lat),
            observer_lon=float(observer_lon),
            days=int(days),
            train_days=int(train_days),
            step_minutes=int(step_minutes),
        )
        if ep is not None:
            out.append(ep)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-TLE-only GEO seq2seq LSTM predictor")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train LSTM from historical years")
    t.add_argument("--sat-name", default="23467")
    t.add_argument("--train-years", default="2021,2022")
    t.add_argument("--val-years", default="2023")
    t.add_argument("--truth-suffix", default="")
    t.add_argument("--observer-lat", type=float, default=36.3022)
    t.add_argument("--observer-lon", type=float, default=137.9031)
    t.add_argument("--observer-alt-m", type=float, default=0.0)
    t.add_argument("--geo-radius-km", type=float, default=42164.0)
    t.add_argument("--days", type=int, default=90)
    t.add_argument("--train-days", type=int, default=7)
    t.add_argument("--step-minutes", type=int, default=10)
    t.add_argument("--episode-stride-train", type=int, default=24)
    t.add_argument("--episode-stride-val", type=int, default=48)
    t.add_argument("--max-train-files", type=int, default=0)
    t.add_argument("--max-val-files", type=int, default=0)
    t.add_argument("--seq-len", type=int, default=288)
    t.add_argument("--seq-stride", type=int, default=72)
    t.add_argument("--predict-batch-size", type=int, default=128)
    t.add_argument("--time-feature-mode", choices=["periodic", "linear"], default="periodic")
    t.add_argument("--time-yearly-harmonics", type=int, default=2)
    t.add_argument("--use-extra-sincos-features", type=int, choices=[0, 1], default=1)
    t.add_argument("--use-calendar-climatology", type=int, choices=[0, 1], default=1)
    t.add_argument("--use-cycle-repeat-prior", type=int, choices=[0, 1], default=1)
    t.add_argument("--cycle-observe-days", type=float, default=3.0)
    t.add_argument("--cycle-min-period-minutes", type=float, default=1100.0)
    t.add_argument("--cycle-max-period-minutes", type=float, default=1800.0)
    t.add_argument("--add-baseline-periodic-harmonics", type=int, default=0)
    t.add_argument("--pseudo-observe-days", type=float, default=3.0)
    t.add_argument("--teacher-force-days", type=float, default=3.0)
    t.add_argument("--periodic-ls-harmonics", type=int, default=2)
    t.add_argument("--warmup-weight", type=float, default=0.20)
    t.add_argument("--forecast-weight", type=float, default=1.00)
    t.add_argument("--lstm-units-1", type=int, default=96)
    t.add_argument("--lstm-units-2", type=int, default=48)
    t.add_argument("--dense-units", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.10)
    t.add_argument("--delta-scale", type=float, default=0.35)
    t.add_argument("--loss-angle-weight", type=float, default=0.40)
    t.add_argument("--loss-softmax-weight", type=float, default=0.80)
    t.add_argument("--loss-softmax-temp", type=float, default=8.0)
    t.add_argument("--loss-worstk-weight", type=float, default=1.20)
    t.add_argument("--loss-worstk-frac", type=float, default=0.10)
    t.add_argument("--loss-uncertainty-weight", type=float, default=0.25)
    t.add_argument("--uncertainty-target-scale", type=float, default=0.05)
    t.add_argument("--dynamic-alpha-floor", type=float, default=0.05)
    t.add_argument("--dynamic-alpha-power", type=float, default=1.4)
    t.add_argument("--uncertainty-smooth", type=int, default=11)
    t.add_argument("--use-ruptures-alpha-seg", type=int, choices=[0, 1], default=0)
    t.add_argument("--ruptures-model", choices=["l1", "l2", "rbf"], default="rbf")
    t.add_argument("--ruptures-penalty", type=float, default=8.0)
    t.add_argument("--ruptures-min-size", type=int, default=24)
    t.add_argument("--ruptures-jump", type=int, default=5)
    t.add_argument("--lr", type=float, default=1.0e-3)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--enable-lla-residual-correction", type=int, choices=[0, 1], default=1)
    t.add_argument("--lla-ridge", type=float, default=1.0e-2)
    t.add_argument("--lla-train-sample-stride", type=int, default=10)
    t.add_argument("--lla-max-train-files", type=int, default=0)
    t.add_argument("--lla-max-val-files", type=int, default=0)
    t.add_argument("--lla-alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    t.add_argument("--lla-azel-alpha-grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output-dir", default=".tmp/single_tle_lstm_geo_model")

    e = sub.add_parser("eval-dir", help="Evaluate trained model on a TLE directory")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--sat-name", default="23467")
    e.add_argument("--truth-years", default="2023,2024")
    e.add_argument("--truth-suffix", default="")
    e.add_argument("--observer-lat", type=float, default=36.3022)
    e.add_argument("--observer-lon", type=float, default=137.9031)
    e.add_argument("--days", type=int, default=90)
    e.add_argument("--train-days", type=int, default=7)
    e.add_argument("--step-minutes", type=int, default=10)
    e.add_argument("--predict-batch-size", type=int, default=128)
    e.add_argument("--dynamic-alpha-floor", type=float, default=-1.0)
    e.add_argument("--dynamic-alpha-power", type=float, default=-1.0)
    e.add_argument("--uncertainty-smooth", type=int, default=0)
    e.add_argument("--use-ruptures-alpha-seg", type=int, choices=[-1, 0, 1], default=-1)
    e.add_argument("--ruptures-model", choices=["", "l1", "l2", "rbf"], default="")
    e.add_argument("--ruptures-penalty", type=float, default=-1.0)
    e.add_argument("--ruptures-min-size", type=int, default=0)
    e.add_argument("--ruptures-jump", type=int, default=0)
    e.add_argument("--tle-dir", default="pred_tle")
    e.add_argument("--max-files", type=int, default=0)
    e.add_argument("--specific-files", default="")
    e.add_argument("--output-json", default=".tmp/eval_single_tle_lstm_geo_model.json")

    c = sub.add_parser("eval-cycle-repeat", help="Evaluate pure 3-day cycle-repeat model")
    c.add_argument("--sat-name", default="23467")
    c.add_argument("--truth-years", default="2023,2024")
    c.add_argument("--truth-suffix", default="")
    c.add_argument("--observer-lat", type=float, default=36.3022)
    c.add_argument("--observer-lon", type=float, default=137.9031)
    c.add_argument("--days", type=int, default=90)
    c.add_argument("--train-days", type=int, default=7)
    c.add_argument("--step-minutes", type=int, default=10)
    c.add_argument("--tle-dir", default="pred_tle")
    c.add_argument("--max-files", type=int, default=0)
    c.add_argument("--specific-files", default="")
    c.add_argument("--cycle-observe-days", type=float, default=3.0)
    c.add_argument("--cycle-min-period-minutes", type=float, default=1100.0)
    c.add_argument("--cycle-max-period-minutes", type=float, default=1800.0)
    c.add_argument("--output-json", default=".tmp/eval_cycle_repeat_model.json")

    l = sub.add_parser("eval-periodic-ls", help="Evaluate pure periodic-LS extrapolation model")
    l.add_argument("--sat-name", default="23467")
    l.add_argument("--truth-years", default="2023,2024")
    l.add_argument("--truth-suffix", default="")
    l.add_argument("--observer-lat", type=float, default=36.3022)
    l.add_argument("--observer-lon", type=float, default=137.9031)
    l.add_argument("--days", type=int, default=90)
    l.add_argument("--train-days", type=int, default=7)
    l.add_argument("--step-minutes", type=int, default=10)
    l.add_argument("--tle-dir", default="pred_tle")
    l.add_argument("--max-files", type=int, default=0)
    l.add_argument("--specific-files", default="")
    l.add_argument("--pseudo-observe-days", type=float, default=3.0)
    l.add_argument("--periodic-ls-harmonics", type=int, default=2)
    l.add_argument("--output-json", default=".tmp/eval_periodic_ls_model.json")

    r = sub.add_parser("eval-online-reid", help="Evaluate online re-identification model (obs/self)")
    r.add_argument("--sat-name", default="23467")
    r.add_argument("--truth-years", default="2024")
    r.add_argument("--truth-suffix", default="")
    r.add_argument("--observer-lat", type=float, default=36.3022)
    r.add_argument("--observer-lon", type=float, default=137.9031)
    r.add_argument("--days", type=int, default=90)
    r.add_argument("--train-days", type=int, default=7)
    r.add_argument("--step-minutes", type=int, default=1)
    r.add_argument("--tle-dir", default="pred_tle")
    r.add_argument("--max-files", type=int, default=0)
    r.add_argument("--specific-files", default="")
    r.add_argument("--reid-mode", choices=["obs", "self", "both"], default="both")
    r.add_argument("--reid-observe-days", type=float, default=3.0)
    r.add_argument("--reid-sidereal-harmonics", type=int, default=3)
    r.add_argument("--reid-solar-harmonics", type=int, default=1)
    r.add_argument("--reid-ridge-lambda", type=float, default=1.0e-2)
    r.add_argument("--reid-forgetting", type=float, default=0.9995)
    r.add_argument("--reid-ar-coeff", type=float, default=0.85)
    r.add_argument("--output-json", default=".tmp/eval_online_reid_model.json")

    m = sub.add_parser("eval-maneuver-hazard", help="Evaluate maneuver-timing hazard ensemble")
    m.add_argument("--sat-name", default="23467")
    m.add_argument("--truth-years", default="2024")
    m.add_argument("--truth-suffix", default="")
    m.add_argument("--climatology-years", default="2020,2021,2022")
    m.add_argument("--climatology-suffix", default="_sincos")
    m.add_argument("--target-year", type=int, default=2024)
    m.add_argument("--hazard-years", default="2020,2021,2022,2023")
    m.add_argument("--hazard-suffix", default="_sincos")
    m.add_argument("--hazard-lag-minutes", type=float, default=float(hm.SECONDS_PER_SIDEREAL_DAY / 60.0))
    m.add_argument("--hazard-smooth-minutes", type=int, default=720)
    m.add_argument("--template-years", default="2020,2021,2022,2023")
    m.add_argument("--template-truth-suffix", default="")
    m.add_argument("--template-episode-stride", type=int, default=72)
    m.add_argument("--template-max-files", type=int, default=0)
    m.add_argument("--template-step-minutes", type=int, default=10)
    m.add_argument("--template-smooth-minutes", type=int, default=240)
    m.add_argument("--observer-lat", type=float, default=36.3022)
    m.add_argument("--observer-lon", type=float, default=137.9031)
    m.add_argument("--days", type=int, default=90)
    m.add_argument("--train-days", type=int, default=7)
    m.add_argument("--step-minutes", type=int, default=1)
    m.add_argument("--fit-days", type=float, default=3.0)
    m.add_argument("--cycle-min-period-minutes", type=float, default=1200.0)
    m.add_argument("--cycle-max-period-minutes", type=float, default=1700.0)
    m.add_argument("--alpha-low-grid", default="0.0,0.1,0.2")
    m.add_argument("--alpha-high-grid", default="0.6,0.8,1.0")
    m.add_argument("--gamma-grid", default="0.7,1.0,1.5")
    m.add_argument("--residual-gain-grid", default="0.5,1.0,1.5,2.0")
    m.add_argument("--tune-max-files", type=int, default=30)
    m.add_argument("--tle-dir", default="pred_tle")
    m.add_argument("--max-files", type=int, default=0)
    m.add_argument("--specific-files", default="")
    m.add_argument("--output-json", default=".tmp/eval_maneuver_hazard_model.json")

    g = sub.add_parser("eval-climatology-align", help="Evaluate climatology + 3-day phase alignment model")
    g.add_argument("--sat-name", default="23467")
    g.add_argument("--truth-years", default="2024")
    g.add_argument("--truth-suffix", default="")
    g.add_argument("--climatology-years", default="2020,2021,2022")
    g.add_argument("--climatology-suffix", default="_sincos")
    g.add_argument("--observer-lat", type=float, default=36.3022)
    g.add_argument("--observer-lon", type=float, default=137.9031)
    g.add_argument("--days", type=int, default=90)
    g.add_argument("--train-days", type=int, default=7)
    g.add_argument("--step-minutes", type=int, default=1)
    g.add_argument("--fit-days", type=float, default=3.0)
    g.add_argument("--tle-dir", default="pred_tle")
    g.add_argument("--max-files", type=int, default=0)
    g.add_argument("--specific-files", default="")
    g.add_argument("--output-json", default=".tmp/eval_climatology_align_model.json")

    z = sub.add_parser("eval-cycle-clim-ensemble", help="Evaluate cycle + climatology ensemble")
    z.add_argument("--sat-name", default="23467")
    z.add_argument("--truth-years", default="2024")
    z.add_argument("--truth-suffix", default="")
    z.add_argument("--climatology-years", default="2020,2021,2022")
    z.add_argument("--climatology-suffix", default="_sincos")
    z.add_argument("--use-trend-clim", type=int, choices=[0, 1], default=0)
    z.add_argument("--target-year", type=int, default=2024)
    z.add_argument("--observer-lat", type=float, default=36.3022)
    z.add_argument("--observer-lon", type=float, default=137.9031)
    z.add_argument("--days", type=int, default=90)
    z.add_argument("--train-days", type=int, default=7)
    z.add_argument("--step-minutes", type=int, default=1)
    z.add_argument("--fit-days", type=float, default=3.0)
    z.add_argument("--cycle-min-period-minutes", type=float, default=1200.0)
    z.add_argument("--cycle-max-period-minutes", type=float, default=1700.0)
    z.add_argument("--tle-dir", default="pred_tle")
    z.add_argument("--max-files", type=int, default=0)
    z.add_argument("--specific-files", default="")
    z.add_argument("--output-json", default=".tmp/eval_cycle_clim_ensemble_model.json")

    q = sub.add_parser("eval-climatology-trend-align", help="Evaluate year-trend climatology + 3-day phase alignment")
    q.add_argument("--sat-name", default="23467")
    q.add_argument("--truth-years", default="2024")
    q.add_argument("--truth-suffix", default="")
    q.add_argument("--climatology-years", default="2020,2021,2022")
    q.add_argument("--climatology-suffix", default="_sincos")
    q.add_argument("--target-year", type=int, default=2024)
    q.add_argument("--observer-lat", type=float, default=36.3022)
    q.add_argument("--observer-lon", type=float, default=137.9031)
    q.add_argument("--days", type=int, default=90)
    q.add_argument("--train-days", type=int, default=7)
    q.add_argument("--step-minutes", type=int, default=1)
    q.add_argument("--fit-days", type=float, default=3.0)
    q.add_argument("--tle-dir", default="pred_tle")
    q.add_argument("--max-files", type=int, default=0)
    q.add_argument("--specific-files", default="")
    q.add_argument("--output-json", default=".tmp/eval_climatology_trend_align_model.json")

    s = sub.add_parser("predict", help="Predict one TLE file")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--tle-file", required=True)
    s.add_argument("--sat-name", default="23467")
    s.add_argument("--observer-lat", type=float, default=36.3022)
    s.add_argument("--observer-lon", type=float, default=137.9031)
    s.add_argument("--observer-alt-m", type=float, default=0.0)
    s.add_argument("--geo-radius-km", type=float, default=42164.0)
    s.add_argument("--days", type=int, default=90)
    s.add_argument("--train-days", type=int, default=7)
    s.add_argument("--step-minutes", type=int, default=10)
    s.add_argument("--predict-batch-size", type=int, default=128)
    s.add_argument("--dynamic-alpha-floor", type=float, default=-1.0)
    s.add_argument("--dynamic-alpha-power", type=float, default=-1.0)
    s.add_argument("--uncertainty-smooth", type=int, default=0)
    s.add_argument("--use-ruptures-alpha-seg", type=int, choices=[-1, 0, 1], default=-1)
    s.add_argument("--ruptures-model", choices=["", "l1", "l2", "rbf"], default="")
    s.add_argument("--ruptures-penalty", type=float, default=-1.0)
    s.add_argument("--ruptures-min-size", type=int, default=0)
    s.add_argument("--ruptures-jump", type=int, default=0)
    s.add_argument("--max-plot-points", type=int, default=12000)
    s.add_argument("--output-csv", default=".tmp/single_tle_lstm_prediction.csv")
    s.add_argument("--truth-years", default="")
    s.add_argument("--truth-suffix", default="")
    return p


def run_train(args: argparse.Namespace) -> None:
    core.require_tensorflow()
    tf = core.tf
    set_seed(int(args.seed))

    out_dir = ROOT / str(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_years = parse_int_csv(args.train_years)
    val_years = parse_int_csv(args.val_years)
    truth_years = sorted(set(train_years + val_years))
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    if not truth_cache:
        raise RuntimeError("No truth csv files were found for requested years")
    calendar_clim = None
    if int(args.use_calendar_climatology) == 1:
        calendar_clim = build_calendar_climatology_trig(years=train_years, suffix=str(args.truth_suffix))
        if calendar_clim is None:
            print("warning: calendar climatology was requested but not available; fallback to none", flush=True)

    train_files = collect_year_files(
        sat_name=str(args.sat_name),
        years=train_years,
        stride=int(args.episode_stride_train),
        max_files=int(args.max_train_files),
    )
    val_files = collect_year_files(
        sat_name=str(args.sat_name),
        years=val_years,
        stride=int(args.episode_stride_val),
        max_files=int(args.max_val_files),
    )
    train_eps = build_episodes(
        paths=train_files,
        sat_name=str(args.sat_name),
        truth_cache=truth_cache,
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        days=int(args.days),
        train_days=int(args.train_days),
        step_minutes=int(args.step_minutes),
        tag="train",
    )
    val_eps = build_episodes(
        paths=val_files,
        sat_name=str(args.sat_name),
        truth_cache=truth_cache,
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        days=int(args.days),
        train_days=int(args.train_days),
        step_minutes=int(args.step_minutes),
        tag="val",
    )
    if not train_eps or not val_eps:
        raise RuntimeError("Train/validation episodes are empty")

    train_steps = []
    val_steps = []
    for ep in train_eps:
        train_steps.append(
            build_step_arrays(
                ep=ep,
                days=int(args.days),
                time_yearly_harmonics=int(args.time_yearly_harmonics),
                time_feature_mode=str(args.time_feature_mode),
                add_baseline_periodic_harmonics=int(args.add_baseline_periodic_harmonics),
                pseudo_observe_days=float(args.pseudo_observe_days),
                periodic_ls_harmonics=int(args.periodic_ls_harmonics),
                teacher_force_days=float(args.teacher_force_days),
                warmup_weight=float(args.warmup_weight),
                forecast_weight=float(args.forecast_weight),
                use_extra_sincos_features=bool(int(args.use_extra_sincos_features) == 1),
                use_cycle_repeat_prior=bool(int(args.use_cycle_repeat_prior) == 1),
                cycle_observe_days=float(args.cycle_observe_days),
                cycle_min_period_minutes=float(args.cycle_min_period_minutes),
                cycle_max_period_minutes=float(args.cycle_max_period_minutes),
                calendar_climatology_trig=calendar_clim,
            )
        )
    for ep in val_eps:
        val_steps.append(
            build_step_arrays(
                ep=ep,
                days=int(args.days),
                time_yearly_harmonics=int(args.time_yearly_harmonics),
                time_feature_mode=str(args.time_feature_mode),
                add_baseline_periodic_harmonics=int(args.add_baseline_periodic_harmonics),
                pseudo_observe_days=float(args.pseudo_observe_days),
                periodic_ls_harmonics=int(args.periodic_ls_harmonics),
                teacher_force_days=float(args.teacher_force_days),
                warmup_weight=float(args.warmup_weight),
                forecast_weight=float(args.forecast_weight),
                use_extra_sincos_features=bool(int(args.use_extra_sincos_features) == 1),
                use_cycle_repeat_prior=bool(int(args.use_cycle_repeat_prior) == 1),
                cycle_observe_days=float(args.cycle_observe_days),
                cycle_min_period_minutes=float(args.cycle_min_period_minutes),
                cycle_max_period_minutes=float(args.cycle_max_period_minutes),
                calendar_climatology_trig=calendar_clim,
            )
        )

    x_train_all = np.concatenate([x for x, _, _ in train_steps], axis=0).astype(np.float32)
    x_mean = np.mean(x_train_all, axis=0).astype(np.float32)
    x_std = np.std(x_train_all, axis=0).astype(np.float32)
    x_std = np.where(x_std < 1.0e-6, 1.0, x_std).astype(np.float32)

    xw_train = []
    yw_train = []
    ww_train = []
    xw_val = []
    yw_val = []
    ww_val = []
    for x, y, w in train_steps:
        xn = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
        a, b, c = make_windows(xn, y, w, seq_len=int(args.seq_len), seq_stride=int(args.seq_stride))
        if a.shape[0] > 0:
            xw_train.append(a)
            yw_train.append(b)
            ww_train.append(c)
    for x, y, w in val_steps:
        xn = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
        a, b, c = make_windows(xn, y, w, seq_len=int(args.seq_len), seq_stride=int(args.seq_stride))
        if a.shape[0] > 0:
            xw_val.append(a)
            yw_val.append(b)
            ww_val.append(c)
    if not xw_train or not xw_val:
        raise RuntimeError("Window generation yielded empty train/validation arrays")

    x_train = np.concatenate(xw_train, axis=0).astype(np.float32)
    y_train = np.concatenate(yw_train, axis=0).astype(np.float32)
    w_train = np.concatenate(ww_train, axis=0).astype(np.float32)
    x_val = np.concatenate(xw_val, axis=0).astype(np.float32)
    y_val = np.concatenate(yw_val, axis=0).astype(np.float32)
    w_val = np.concatenate(ww_val, axis=0).astype(np.float32)
    y_train_pack = np.concatenate([y_train, w_train[:, :, None]], axis=2).astype(np.float32)
    y_val_pack = np.concatenate([y_val, w_val[:, :, None]], axis=2).astype(np.float32)

    if str(args.time_feature_mode) == "linear":
        time_feat_dim = 3
    else:
        time_feat_dim = int(
            hm.build_time_cyclic_features_np(
                unix=np.asarray([0.0, 60.0], dtype=np.float64),
                yearly_harmonics=int(args.time_yearly_harmonics),
            ).shape[1]
        )
    trig_offset = int(time_feat_dim)

    model = build_lstm_model(
        seq_len=int(args.seq_len),
        feat_dim=int(x_train.shape[-1]),
        trig_offset=trig_offset,
        trig_mean=np.asarray(x_mean[trig_offset : trig_offset + 4], dtype=np.float32),
        trig_std=np.asarray(x_std[trig_offset : trig_offset + 4], dtype=np.float32),
        lstm_units_1=int(args.lstm_units_1),
        lstm_units_2=int(args.lstm_units_2),
        dense_units=int(args.dense_units),
        dropout=float(args.dropout),
        delta_scale=float(args.delta_scale),
        lr=float(args.lr),
        loss_angle_weight=float(args.loss_angle_weight),
        loss_softmax_weight=float(args.loss_softmax_weight),
        loss_softmax_temp=float(args.loss_softmax_temp),
        loss_worstk_weight=float(args.loss_worstk_weight),
        loss_worstk_frac=float(args.loss_worstk_frac),
        loss_uncertainty_weight=float(args.loss_uncertainty_weight),
        uncertainty_target_scale=float(args.uncertainty_target_scale),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1.0e-5,
        ),
    ]
    hist = model.fit(
        x=x_train,
        y=y_train_pack,
        validation_data=(x_val, y_val_pack),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    model.save(out_dir / "model.keras")
    meta = {
        "sat_name": str(args.sat_name),
        "train_years": list(train_years),
        "val_years": list(val_years),
        "truth_suffix": str(args.truth_suffix),
        "observer_lat": float(args.observer_lat),
        "observer_lon": float(args.observer_lon),
        "days": int(args.days),
        "train_days": int(args.train_days),
        "step_minutes": int(args.step_minutes),
        "seq_len": int(args.seq_len),
        "seq_stride": int(args.seq_stride),
        "predict_batch_size": int(args.predict_batch_size),
        "time_feature_mode": str(args.time_feature_mode),
        "time_yearly_harmonics": int(args.time_yearly_harmonics),
        "use_extra_sincos_features": bool(int(args.use_extra_sincos_features) == 1),
        "use_calendar_climatology": bool(int(args.use_calendar_climatology) == 1),
        "use_cycle_repeat_prior": bool(int(args.use_cycle_repeat_prior) == 1),
        "cycle_observe_days": float(args.cycle_observe_days),
        "cycle_min_period_minutes": float(args.cycle_min_period_minutes),
        "cycle_max_period_minutes": float(args.cycle_max_period_minutes),
        "add_baseline_periodic_harmonics": int(args.add_baseline_periodic_harmonics),
        "pseudo_observe_days": float(args.pseudo_observe_days),
        "teacher_force_days": float(args.teacher_force_days),
        "periodic_ls_harmonics": int(args.periodic_ls_harmonics),
        "time_feat_dim": int(time_feat_dim),
        "trig_offset": int(trig_offset),
        "warmup_weight": float(args.warmup_weight),
        "forecast_weight": float(args.forecast_weight),
        "lstm_units_1": int(args.lstm_units_1),
        "lstm_units_2": int(args.lstm_units_2),
        "dense_units": int(args.dense_units),
        "dropout": float(args.dropout),
        "delta_scale": float(args.delta_scale),
        "loss_angle_weight": float(args.loss_angle_weight),
        "loss_softmax_weight": float(args.loss_softmax_weight),
        "loss_softmax_temp": float(args.loss_softmax_temp),
        "loss_worstk_weight": float(args.loss_worstk_weight),
        "loss_worstk_frac": float(args.loss_worstk_frac),
        "loss_uncertainty_weight": float(args.loss_uncertainty_weight),
        "uncertainty_target_scale": float(args.uncertainty_target_scale),
        "dynamic_alpha_floor": float(args.dynamic_alpha_floor),
        "dynamic_alpha_power": float(args.dynamic_alpha_power),
        "uncertainty_smooth": int(args.uncertainty_smooth),
        "use_ruptures_alpha_seg": bool(int(args.use_ruptures_alpha_seg) == 1),
        "ruptures_model": str(args.ruptures_model),
        "ruptures_penalty": float(args.ruptures_penalty),
        "ruptures_min_size": int(args.ruptures_min_size),
        "ruptures_jump": int(args.ruptures_jump),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "feature_dim": int(x_train.shape[-1]),
        "train_window_count": int(x_train.shape[0]),
        "val_window_count": int(x_val.shape[0]),
        "x_mean": np.asarray(x_mean, dtype=np.float32).tolist(),
        "x_std": np.asarray(x_std, dtype=np.float32).tolist(),
        "history": {k: [float(vv) for vv in v] for k, v in hist.history.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    use_cycle_prior = bool(int(args.use_cycle_repeat_prior) == 1)
    val_rows = []
    val_cache = []
    for ep, (x, _, _) in zip(val_eps, val_steps):
        xn = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
        pred_trig4, pred_unc = predict_full_pack(
            model=model,
            x_norm=xn,
            seq_len=int(args.seq_len),
            seq_stride=int(args.seq_stride),
            predict_batch_size=int(args.predict_batch_size),
        )
        pred_azel_raw = hm.trig4_to_azel(pred_trig4)
        teacher_mask = compute_teacher_mask(ep.unix, teacher_force_days=float(args.teacher_force_days))
        pred_azel_dyn, alpha_dyn, unc_smooth = apply_dynamic_uncertainty_blend(
            base_azel=ep.baseline_azel,
            pred_azel=pred_azel_raw,
            uncertainty=pred_unc,
            teacher_mask=teacher_mask,
            alpha_floor=float(args.dynamic_alpha_floor),
            alpha_power=float(args.dynamic_alpha_power),
            uncertainty_smooth=int(args.uncertainty_smooth),
            use_ruptures_alpha_seg=bool(int(args.use_ruptures_alpha_seg) == 1),
            ruptures_model=str(args.ruptures_model),
            ruptures_penalty=float(args.ruptures_penalty),
            ruptures_min_size=int(args.ruptures_min_size),
            ruptures_jump=int(args.ruptures_jump),
        )
        err_raw = compute_forecast_max_abs_error(ep.true_azel, pred_azel_raw, ep.forecast_mask)
        err_dyn = compute_forecast_max_abs_error(ep.true_azel, pred_azel_dyn, ep.forecast_mask)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        cycle_azel = np.asarray(ep.baseline_azel, dtype=np.float64)
        cycle_period = float(hm.SECONDS_PER_SIDEREAL_DAY / 60.0)
        cycle_score = 0.0
        cycle_rmse_az = 0.0
        cycle_rmse_el = 0.0
        if use_cycle_prior:
            cycle_azel, cycle_period, cycle_score, cycle_rmse_az, cycle_rmse_el = build_cycle_repeat_prior_azel_for_episode(
                ep=ep,
                observe_days=float(args.cycle_observe_days),
                min_period_minutes=float(args.cycle_min_period_minutes),
                max_period_minutes=float(args.cycle_max_period_minutes),
            )
        cycle_err = compute_forecast_max_abs_error(ep.true_azel, cycle_azel, ep.forecast_mask)
        val_cache.append(
            (
                ep,
                pred_azel_dyn,
                cycle_azel,
                float(base_err),
                float(err_raw),
                float(err_dyn),
                float(cycle_err),
                alpha_dyn,
                unc_smooth,
                float(cycle_period),
                float(cycle_score),
                float(cycle_rmse_az),
                float(cycle_rmse_el),
            )
        )
        val_rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "lstm_raw_max_abs_error_max": float(err_raw),
                "lstm_dynamic_max_abs_error_max": float(err_dyn),
                "cycle_repeat_max_abs_error_max": float(cycle_err),
                "dynamic_alpha_mean": float(np.mean(alpha_dyn)),
                "dynamic_alpha_min": float(np.min(alpha_dyn)),
                "dynamic_alpha_max": float(np.max(alpha_dyn)),
                "uncertainty_mean": float(np.mean(unc_smooth)),
                "uncertainty_max": float(np.max(unc_smooth)),
                "cycle_period_minutes": float(cycle_period),
                "cycle_score": float(cycle_score),
                "cycle_fit_rmse_az": float(cycle_rmse_az),
                "cycle_fit_rmse_el": float(cycle_rmse_el),
            }
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    alpha_scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, pred_azel_dyn, _, _, _, _, _, _, _, _, _, _, _ in val_cache:
            blended = blend_azel_unitvec(ep.baseline_azel, pred_azel_dyn, alpha=float(alpha))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask))
        alpha_scores.append(
            {
                "alpha": float(alpha),
                "summary": summarize(errs),
            }
        )
    alpha_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_alpha = float(alpha_scores[0]["alpha"])

    cycle_alpha_grid = [i / 20.0 for i in range(21)] if use_cycle_prior else [0.0]
    cycle_alpha_scores = []
    for cycle_alpha in cycle_alpha_grid:
        errs = []
        for ep, pred_azel_dyn, cycle_azel, _, _, _, _, _, _, _, _, _, _ in val_cache:
            blended = blend_azel_unitvec(ep.baseline_azel, pred_azel_dyn, alpha=best_alpha)
            final = blend_azel_unitvec(blended, cycle_azel, alpha=float(cycle_alpha))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, final, ep.forecast_mask))
        cycle_alpha_scores.append({"cycle_alpha": float(cycle_alpha), "summary": summarize(errs)})
    cycle_alpha_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_cycle_alpha = float(cycle_alpha_scores[0]["cycle_alpha"])

    val_rows = []
    for (
        ep,
        pred_azel_dyn,
        cycle_azel,
        base_err,
        err_raw,
        err_dyn,
        cycle_err,
        alpha_dyn,
        unc_smooth,
        cycle_period,
        cycle_score,
        cycle_rmse_az,
        cycle_rmse_el,
    ) in val_cache:
        blended = blend_azel_unitvec(ep.baseline_azel, pred_azel_dyn, alpha=best_alpha)
        fused = blend_azel_unitvec(blended, cycle_azel, alpha=best_cycle_alpha)
        err_blended = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
        err_fused = compute_forecast_max_abs_error(ep.true_azel, fused, ep.forecast_mask)
        val_rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "lstm_raw_max_abs_error_max": float(err_raw),
                "lstm_dynamic_max_abs_error_max": float(err_dyn),
                "cycle_repeat_max_abs_error_max": float(cycle_err),
                "lstm_blended_max_abs_error_max": float(err_blended),
                "lstm_cycle_blended_max_abs_error_max": float(err_fused),
                "dynamic_alpha_mean": float(np.mean(alpha_dyn)),
                "dynamic_alpha_min": float(np.min(alpha_dyn)),
                "dynamic_alpha_max": float(np.max(alpha_dyn)),
                "uncertainty_mean": float(np.mean(unc_smooth)),
                "uncertainty_max": float(np.max(unc_smooth)),
                "cycle_period_minutes": float(cycle_period),
                "cycle_score": float(cycle_score),
                "cycle_fit_rmse_az": float(cycle_rmse_az),
                "cycle_fit_rmse_el": float(cycle_rmse_el),
            }
        )
    val_rows.sort(key=lambda r: r["lstm_cycle_blended_max_abs_error_max"], reverse=True)
    val_summary = {
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in val_rows]),
        "lstm_raw_summary": summarize([float(r["lstm_raw_max_abs_error_max"]) for r in val_rows]),
        "lstm_dynamic_summary": summarize([float(r["lstm_dynamic_max_abs_error_max"]) for r in val_rows]),
        "cycle_repeat_summary": summarize([float(r["cycle_repeat_max_abs_error_max"]) for r in val_rows]),
        "lstm_blended_summary": summarize([float(r["lstm_blended_max_abs_error_max"]) for r in val_rows]),
        "lstm_cycle_blended_summary": summarize([float(r["lstm_cycle_blended_max_abs_error_max"]) for r in val_rows]),
        "selected_alpha": float(best_alpha),
        "selected_cycle_alpha": float(best_cycle_alpha),
        "alpha_search_top5": alpha_scores[:5],
        "cycle_alpha_search_top5": cycle_alpha_scores[:5],
        "rows": val_rows,
    }
    (out_dir / "validation_rows.json").write_text(json.dumps(val_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["post_blend_alpha"] = float(best_alpha)
    meta["post_cycle_alpha"] = float(best_cycle_alpha)
    if int(args.enable_lla_residual_correction) == 1:
        print("[lla] building residual correction samples...", flush=True)
        lla_alpha_grid = parse_float_csv(args.lla_alpha_grid)
        if not lla_alpha_grid:
            lla_alpha_grid = [i / 10.0 for i in range(11)]
        lla_azel_alpha_grid = parse_float_csv(args.lla_azel_alpha_grid)
        if not lla_azel_alpha_grid:
            lla_azel_alpha_grid = [i / 10.0 for i in range(11)]
        anchor_grid = [0.0, 0.5, 1.0]
        stage2_max_base_features = 24
        stage2_pair_features = 8

        x_train_lla, y_train_lla, _ = build_lla_residual_training_samples(
            paths=train_files,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
            model=model,
            meta=meta,
            x_mean=x_mean,
            x_std=x_std,
            calendar_climatology_trig=calendar_clim,
            sample_stride=int(args.lla_train_sample_stride),
            max_files=int(args.lla_max_train_files),
            tag="lla-train",
        )
        _, _, val_packs = build_lla_residual_training_samples(
            paths=val_files,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
            model=model,
            meta=meta,
            x_mean=x_mean,
            x_std=x_std,
            calendar_climatology_trig=calendar_clim,
            sample_stride=1,
            max_files=int(args.lla_max_val_files),
            tag="lla-val",
        )
        if x_train_lla.shape[0] > 0 and val_packs:
            lla_model = fit_linear_multi_ridge(
                x=np.asarray(x_train_lla, dtype=np.float64),
                y=np.asarray(y_train_lla, dtype=np.float64),
                ridge=float(args.lla_ridge),
            )
            d1_train = predict_linear_multi_ridge(lla_model, np.asarray(x_train_lla, dtype=np.float64))
            y_resid_train = np.asarray(y_train_lla, dtype=np.float64) - np.asarray(d1_train, dtype=np.float64)
            x_train_lla_stage2 = build_lla_stage2_features(
                x_lla_feat=np.asarray(x_train_lla, dtype=np.float64),
                max_base_features=int(stage2_max_base_features),
                pair_features=int(stage2_pair_features),
            )
            lla_model_stage2 = fit_linear_multi_ridge(
                x=np.asarray(x_train_lla_stage2, dtype=np.float64),
                y=np.asarray(y_resid_train, dtype=np.float64),
                ridge=float(max(1.0e-6, float(args.lla_ridge) * 3.0)),
            )

            val_delta: list[tuple[LlaResidualPack, np.ndarray, np.ndarray]] = []
            for pack in val_packs:
                feat = build_lla_residual_features(
                    unix=np.asarray(pack.unix, dtype=np.float64),
                    baseline_azel=np.asarray(pack.baseline_azel, dtype=np.float64),
                    pred_azel=np.asarray(pack.pred_azel, dtype=np.float64),
                    baseline_lla=np.asarray(pack.baseline_lla, dtype=np.float64),
                    pred_lla_geo=np.asarray(pack.pred_lla_geo, dtype=np.float64),
                    static_features=np.asarray(pack.static_features, dtype=np.float64),
                    yearly_harmonics=int(meta.get("time_yearly_harmonics", 2)),
                )
                d1 = predict_linear_multi_ridge(lla_model, feat)
                feat2 = build_lla_stage2_features(
                    x_lla_feat=np.asarray(feat, dtype=np.float64),
                    max_base_features=int(stage2_max_base_features),
                    pair_features=int(stage2_pair_features),
                )
                d2 = predict_linear_multi_ridge(lla_model_stage2, feat2)
                val_delta.append((pack, np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64)))

            lat_errors = []
            lon_errors = []
            alt_errors = []
            for a in lla_alpha_grid:
                lat_pack_errs = []
                lon_pack_errs = []
                alt_pack_errs = []
                for pack, d1, _ in val_delta:
                    m = np.asarray(pack.forecast_mask, dtype=bool)
                    d_tmp = np.zeros((int(d1.shape[0]), 3), dtype=np.float64)
                    d_tmp[:, 0] = float(a) * np.asarray(d1[:, 0], dtype=np.float64)
                    d_tmp[:, 1] = float(a) * np.asarray(d1[:, 1], dtype=np.float64)
                    d_tmp[:, 2] = float(a) * np.asarray(d1[:, 2], dtype=np.float64)
                    d_tmp = anchor_delta_with_warmup(
                        delta_lla=d_tmp,
                        unix=np.asarray(pack.unix, dtype=np.float64),
                        teacher_force_days=float(args.train_days),
                        anchor_scale=1.0,
                    )
                    lat_pred = np.asarray(pack.pred_lla_geo[:, 0] + d_tmp[:, 0], dtype=np.float64)
                    lon_pred = np.asarray(core.wrap180(pack.pred_lla_geo[:, 1] + d_tmp[:, 1]), dtype=np.float64)
                    alt_pred = np.asarray(pack.pred_lla_geo[:, 2] + d_tmp[:, 2], dtype=np.float64)
                    lat_err = np.abs(lat_pred[m] - np.asarray(pack.true_lla[m, 0], dtype=np.float64))
                    lon_err = np.abs(core.angle_diff_deg(lon_pred[m], np.asarray(pack.true_lla[m, 1], dtype=np.float64)))
                    alt_err = np.abs(alt_pred[m] - np.asarray(pack.true_lla[m, 2], dtype=np.float64))
                    lat_pack_errs.append(float(np.max(lat_err)))
                    lon_pack_errs.append(float(np.max(lon_err)))
                    alt_pack_errs.append(float(np.max(alt_err)))
                lat_errors.append((float(a), lat_pack_errs))
                lon_errors.append((float(a), lon_pack_errs))
                alt_errors.append((float(a), alt_pack_errs))

            best_alpha_lat = tune_single_alpha(lat_errors)
            best_alpha_lon = tune_single_alpha(lon_errors)
            best_alpha_alt = tune_single_alpha(alt_errors)

            beta_lat_scores = []
            beta_lon_scores = []
            beta_alt_scores = []
            for b in lla_alpha_grid:
                lat_pack_errs = []
                lon_pack_errs = []
                alt_pack_errs = []
                for pack, d1, d2 in val_delta:
                    m = np.asarray(pack.forecast_mask, dtype=bool)
                    d_tmp = np.zeros((int(d1.shape[0]), 3), dtype=np.float64)
                    d_tmp[:, 0] = float(best_alpha_lat) * np.asarray(d1[:, 0], dtype=np.float64) + float(b) * np.asarray(d2[:, 0], dtype=np.float64)
                    d_tmp[:, 1] = float(best_alpha_lon) * np.asarray(d1[:, 1], dtype=np.float64) + float(b) * np.asarray(d2[:, 1], dtype=np.float64)
                    d_tmp[:, 2] = float(best_alpha_alt) * np.asarray(d1[:, 2], dtype=np.float64) + float(b) * np.asarray(d2[:, 2], dtype=np.float64)
                    d_tmp = anchor_delta_with_warmup(
                        delta_lla=d_tmp,
                        unix=np.asarray(pack.unix, dtype=np.float64),
                        teacher_force_days=float(args.train_days),
                        anchor_scale=1.0,
                    )
                    lat_pred = np.asarray(pack.pred_lla_geo[:, 0] + d_tmp[:, 0], dtype=np.float64)
                    lon_pred = np.asarray(core.wrap180(pack.pred_lla_geo[:, 1] + d_tmp[:, 1]), dtype=np.float64)
                    alt_pred = np.asarray(pack.pred_lla_geo[:, 2] + d_tmp[:, 2], dtype=np.float64)
                    lat_err = np.abs(lat_pred[m] - np.asarray(pack.true_lla[m, 0], dtype=np.float64))
                    lon_err = np.abs(core.angle_diff_deg(lon_pred[m], np.asarray(pack.true_lla[m, 1], dtype=np.float64)))
                    alt_err = np.abs(alt_pred[m] - np.asarray(pack.true_lla[m, 2], dtype=np.float64))
                    lat_pack_errs.append(float(np.max(lat_err)))
                    lon_pack_errs.append(float(np.max(lon_err)))
                    alt_pack_errs.append(float(np.max(alt_err)))
                beta_lat_scores.append((float(b), lat_pack_errs))
                beta_lon_scores.append((float(b), lon_pack_errs))
                beta_alt_scores.append((float(b), alt_pack_errs))
            best_beta_lat = tune_single_alpha(beta_lat_scores)
            best_beta_lon = tune_single_alpha(beta_lon_scores)
            best_beta_alt = tune_single_alpha(beta_alt_scores)

            anchor_scores = []
            for ag in anchor_grid:
                errs = []
                for pack, d1, d2 in val_delta:
                    m = np.asarray(pack.forecast_mask, dtype=bool)
                    d_tmp = np.zeros((int(d1.shape[0]), 3), dtype=np.float64)
                    d_tmp[:, 0] = float(best_alpha_lat) * np.asarray(d1[:, 0], dtype=np.float64) + float(best_beta_lat) * np.asarray(d2[:, 0], dtype=np.float64)
                    d_tmp[:, 1] = float(best_alpha_lon) * np.asarray(d1[:, 1], dtype=np.float64) + float(best_beta_lon) * np.asarray(d2[:, 1], dtype=np.float64)
                    d_tmp[:, 2] = float(best_alpha_alt) * np.asarray(d1[:, 2], dtype=np.float64) + float(best_beta_alt) * np.asarray(d2[:, 2], dtype=np.float64)
                    d_tmp = anchor_delta_with_warmup(
                        delta_lla=d_tmp,
                        unix=np.asarray(pack.unix, dtype=np.float64),
                        teacher_force_days=float(args.train_days),
                        anchor_scale=float(ag),
                    )
                    lla_corr = apply_lla_delta_with_alpha(
                        pred_lla=np.asarray(pack.pred_lla_geo, dtype=np.float64),
                        delta_lla=np.asarray(d_tmp, dtype=np.float64),
                        alpha_lat=1.0,
                        alpha_lon=1.0,
                        alpha_alt=1.0,
                    )
                    true_full = build_full_from_lla_azel(np.asarray(pack.true_lla[m], dtype=np.float64), np.asarray(pack.true_azel[m], dtype=np.float64))
                    corr_full = build_full_from_lla_azel(np.asarray(lla_corr[m], dtype=np.float64), np.asarray(pack.pred_azel[m], dtype=np.float64))
                    errs.append(float(core.compute_metrics_lla(true_full, corr_full)["overall"]["max_abs_error_max"]))
                anchor_scores.append((float(ag), errs))
            best_anchor_scale = tune_single_alpha(anchor_scores)

            azel_scores = []
            for a_azel in lla_azel_alpha_grid:
                errs = []
                for pack, d1, d2 in val_delta:
                    d_tmp = np.zeros((int(d1.shape[0]), 3), dtype=np.float64)
                    d_tmp[:, 0] = float(best_alpha_lat) * np.asarray(d1[:, 0], dtype=np.float64) + float(best_beta_lat) * np.asarray(d2[:, 0], dtype=np.float64)
                    d_tmp[:, 1] = float(best_alpha_lon) * np.asarray(d1[:, 1], dtype=np.float64) + float(best_beta_lon) * np.asarray(d2[:, 1], dtype=np.float64)
                    d_tmp[:, 2] = float(best_alpha_alt) * np.asarray(d1[:, 2], dtype=np.float64) + float(best_beta_alt) * np.asarray(d2[:, 2], dtype=np.float64)
                    d_tmp = anchor_delta_with_warmup(
                        delta_lla=d_tmp,
                        unix=np.asarray(pack.unix, dtype=np.float64),
                        teacher_force_days=float(args.train_days),
                        anchor_scale=float(best_anchor_scale),
                    )
                    lla_corr = apply_lla_delta_with_alpha(
                        pred_lla=np.asarray(pack.pred_lla_geo, dtype=np.float64),
                        delta_lla=np.asarray(d_tmp, dtype=np.float64),
                        alpha_lat=1.0,
                        alpha_lon=1.0,
                        alpha_alt=1.0,
                    )
                    if float(a_azel) > 0.0:
                        az_from_lla = recompute_azel_from_lla_np(
                            sat_lla=lla_corr,
                            observer_lat=float(args.observer_lat),
                            observer_lon=float(args.observer_lon),
                            observer_alt_m=float(args.observer_alt_m),
                        )
                        az_final = blend_azel_unitvec(np.asarray(pack.pred_azel, dtype=np.float64), az_from_lla, alpha=float(a_azel))
                    else:
                        az_final = np.asarray(pack.pred_azel, dtype=np.float64)
                    errs.append(float(compute_forecast_max_abs_error(pack.true_azel, az_final, pack.forecast_mask)))
                azel_scores.append((float(a_azel), errs))
            best_alpha_azel = tune_single_alpha(azel_scores)

            base_lla_errs = []
            corr_lla_errs = []
            corr_azel_errs = []
            for pack, d1, d2 in val_delta:
                m = np.asarray(pack.forecast_mask, dtype=bool)
                true_full = build_full_from_lla_azel(np.asarray(pack.true_lla[m], dtype=np.float64), np.asarray(pack.true_azel[m], dtype=np.float64))
                base_full = build_full_from_lla_azel(np.asarray(pack.pred_lla_geo[m], dtype=np.float64), np.asarray(pack.pred_azel[m], dtype=np.float64))
                base_lla_errs.append(float(core.compute_metrics_lla(true_full, base_full)["overall"]["max_abs_error_max"]))

                d_tmp = np.zeros((int(d1.shape[0]), 3), dtype=np.float64)
                d_tmp[:, 0] = float(best_alpha_lat) * np.asarray(d1[:, 0], dtype=np.float64) + float(best_beta_lat) * np.asarray(d2[:, 0], dtype=np.float64)
                d_tmp[:, 1] = float(best_alpha_lon) * np.asarray(d1[:, 1], dtype=np.float64) + float(best_beta_lon) * np.asarray(d2[:, 1], dtype=np.float64)
                d_tmp[:, 2] = float(best_alpha_alt) * np.asarray(d1[:, 2], dtype=np.float64) + float(best_beta_alt) * np.asarray(d2[:, 2], dtype=np.float64)
                d_tmp = anchor_delta_with_warmup(
                    delta_lla=d_tmp,
                    unix=np.asarray(pack.unix, dtype=np.float64),
                    teacher_force_days=float(args.train_days),
                    anchor_scale=float(best_anchor_scale),
                )
                lla_corr = apply_lla_delta_with_alpha(
                    pred_lla=np.asarray(pack.pred_lla_geo, dtype=np.float64),
                    delta_lla=np.asarray(d_tmp, dtype=np.float64),
                    alpha_lat=1.0,
                    alpha_lon=1.0,
                    alpha_alt=1.0,
                )
                if float(best_alpha_azel) > 0.0:
                    az_from_lla = recompute_azel_from_lla_np(
                        sat_lla=lla_corr,
                        observer_lat=float(args.observer_lat),
                        observer_lon=float(args.observer_lon),
                        observer_alt_m=float(args.observer_alt_m),
                    )
                    az_final = blend_azel_unitvec(np.asarray(pack.pred_azel, dtype=np.float64), az_from_lla, alpha=float(best_alpha_azel))
                else:
                    az_final = np.asarray(pack.pred_azel, dtype=np.float64)
                corr_full = build_full_from_lla_azel(np.asarray(lla_corr[m], dtype=np.float64), np.asarray(az_final[m], dtype=np.float64))
                corr_lla_errs.append(float(core.compute_metrics_lla(true_full, corr_full)["overall"]["max_abs_error_max"]))
                corr_azel_errs.append(float(core.compute_metrics_azel(true_full, corr_full)["overall"]["max_abs_error_max"]))

            lla_payload = {
                "enabled": True,
                "ridge": float(args.lla_ridge),
                "train_rows": int(x_train_lla.shape[0]),
                "val_packs": int(len(val_packs)),
                "alpha_lat": float(best_alpha_lat),
                "alpha_lon": float(best_alpha_lon),
                "alpha_alt": float(best_alpha_alt),
                "beta_lat": float(best_beta_lat),
                "beta_lon": float(best_beta_lon),
                "beta_alt": float(best_beta_alt),
                "alpha_azel": float(best_alpha_azel),
                "anchor_scale": float(best_anchor_scale),
                "stage2_max_base_features": int(stage2_max_base_features),
                "stage2_pair_features": int(stage2_pair_features),
                "baseline_lla_summary": summarize(base_lla_errs),
                "corrected_lla_summary": summarize(corr_lla_errs),
                "corrected_azel_summary": summarize(corr_azel_errs),
                "model": lla_model,
                "model_stage2": lla_model_stage2,
            }
            meta["lla_residual_correction"] = lla_payload
            print(json.dumps({"lla_residual_correction": lla_payload["corrected_lla_summary"]}, ensure_ascii=False, indent=2), flush=True)
        else:
            meta["lla_residual_correction"] = {"enabled": False, "reason": "insufficient_samples"}
            print("[lla] skip residual correction (insufficient samples)", flush=True)

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(val_summary["lstm_cycle_blended_summary"], ensure_ascii=False, indent=2))


def resolve_model_dir(model_dir: Path) -> Path:
    base = Path(model_dir)
    cands: list[Path] = []
    cands.append(base)
    cands.append(ROOT / base)
    cands.append(ROOT / ".tmp" / base)
    cands.append(ROOT / ".tmp" / base.name)

    seen: set[str] = set()
    uniq: list[Path] = []
    for c in cands:
        k = str(c.resolve()) if c.exists() else str(c)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)

    for c in uniq:
        if (c / "model.keras").exists() and (c / "meta.json").exists():
            return c

    detail = ", ".join(str(c) for c in uniq)
    raise FileNotFoundError(
        "Model directory with model.keras/meta.json was not found. "
        f"requested={model_dir}, searched=[{detail}]"
    )


def load_model_and_meta(model_dir: Path):
    core.require_tensorflow()
    tf = core.tf
    resolved_model_dir = resolve_model_dir(model_dir)
    model = tf.keras.models.load_model(
        resolved_model_dir / "model.keras",
        compile=False,
        custom_objects={"normalize_trig4_seq_tf": normalize_trig4_seq_tf},
    )
    meta = json.loads((resolved_model_dir / "meta.json").read_text(encoding="utf-8"))
    x_mean = np.asarray(meta["x_mean"], dtype=np.float32)
    x_std = np.asarray(meta["x_std"], dtype=np.float32)
    return model, meta, x_mean, x_std


def align_features_to_model_input(
    x: np.ndarray,
    x_mean: np.ndarray,
) -> tuple[np.ndarray, dict | None]:
    x_arr = np.asarray(x, dtype=np.float32)
    mean = np.asarray(x_mean, dtype=np.float32).reshape(-1)
    need = int(mean.shape[0])
    have = int(x_arr.shape[1])
    if have == need:
        return x_arr, None
    if have > need:
        return np.asarray(x_arr[:, :need], dtype=np.float32), {
            "mode": "truncate_tail",
            "input_feature_dim": int(have),
            "model_feature_dim": int(need),
        }
    n = int(x_arr.shape[0])
    pad_cols = np.repeat(mean[None, have:need], n, axis=0).astype(np.float32)
    out = np.concatenate([x_arr, pad_cols], axis=1).astype(np.float32)
    return out, {
        "mode": "pad_with_mean",
        "input_feature_dim": int(have),
        "model_feature_dim": int(need),
    }


def predict_episode_with_model(
    model,
    meta: dict,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    ep: Episode,
    warmup_weight: float,
    forecast_weight: float,
    dynamic_alpha_floor: float | None = None,
    dynamic_alpha_power: float | None = None,
    uncertainty_smooth: int | None = None,
    use_ruptures_alpha_seg: int | None = None,
    ruptures_model: str | None = None,
    ruptures_penalty: float | None = None,
    ruptures_min_size: int | None = None,
    ruptures_jump: int | None = None,
    predict_batch_size: int | None = None,
    calendar_climatology_trig: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    x, _, _ = build_step_arrays(
        ep=ep,
        days=int(meta["days"]),
        time_yearly_harmonics=int(meta["time_yearly_harmonics"]),
        time_feature_mode=str(meta.get("time_feature_mode", "periodic")),
        use_extra_sincos_features=bool(meta.get("use_extra_sincos_features", False)),
        use_cycle_repeat_prior=bool(meta.get("use_cycle_repeat_prior", False)),
        cycle_observe_days=float(meta.get("cycle_observe_days", 3.0)),
        cycle_min_period_minutes=float(meta.get("cycle_min_period_minutes", 1100.0)),
        cycle_max_period_minutes=float(meta.get("cycle_max_period_minutes", 1800.0)),
        calendar_climatology_trig=calendar_climatology_trig,
        add_baseline_periodic_harmonics=int(meta.get("add_baseline_periodic_harmonics", 0)),
        pseudo_observe_days=float(meta.get("pseudo_observe_days", 3.0)),
        periodic_ls_harmonics=int(meta.get("periodic_ls_harmonics", 2)),
        teacher_force_days=float(meta.get("teacher_force_days", 3.0)),
        warmup_weight=float(warmup_weight),
        forecast_weight=float(forecast_weight),
    )
    x_aligned, _ = align_features_to_model_input(x=x, x_mean=x_mean)
    xn = ((x_aligned - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    pred_batch = int(meta.get("predict_batch_size", 128))
    if predict_batch_size is not None and int(predict_batch_size) > 0:
        pred_batch = int(predict_batch_size)
    pred_trig4, pred_unc = predict_full_pack(
        model=model,
        x_norm=xn,
        seq_len=int(meta["seq_len"]),
        seq_stride=int(meta["seq_stride"]),
        predict_batch_size=int(pred_batch),
    )
    pred_azel_raw = hm.trig4_to_azel(pred_trig4)
    alpha_floor = float(meta.get("dynamic_alpha_floor", 0.05))
    if dynamic_alpha_floor is not None and float(dynamic_alpha_floor) >= 0.0:
        alpha_floor = float(dynamic_alpha_floor)
    alpha_power = float(meta.get("dynamic_alpha_power", 1.4))
    if dynamic_alpha_power is not None and float(dynamic_alpha_power) >= 0.0:
        alpha_power = float(dynamic_alpha_power)
    unc_smooth = int(meta.get("uncertainty_smooth", 11))
    if uncertainty_smooth is not None and int(uncertainty_smooth) > 0:
        unc_smooth = int(uncertainty_smooth)
    use_rpt = bool(meta.get("use_ruptures_alpha_seg", False))
    if use_ruptures_alpha_seg is not None and int(use_ruptures_alpha_seg) >= 0:
        use_rpt = bool(int(use_ruptures_alpha_seg) == 1)
    rpt_model = str(meta.get("ruptures_model", "rbf"))
    if ruptures_model is not None and str(ruptures_model).strip():
        rpt_model = str(ruptures_model)
    rpt_pen = float(meta.get("ruptures_penalty", 8.0))
    if ruptures_penalty is not None and float(ruptures_penalty) > 0.0:
        rpt_pen = float(ruptures_penalty)
    rpt_min = int(meta.get("ruptures_min_size", 24))
    if ruptures_min_size is not None and int(ruptures_min_size) > 0:
        rpt_min = int(ruptures_min_size)
    rpt_jump = int(meta.get("ruptures_jump", 5))
    if ruptures_jump is not None and int(ruptures_jump) > 0:
        rpt_jump = int(ruptures_jump)
    teacher_mask = compute_teacher_mask(
        unix=ep.unix,
        teacher_force_days=float(meta.get("teacher_force_days", 3.0)),
    )
    pred_azel_dyn, _, _ = apply_dynamic_uncertainty_blend(
        base_azel=ep.baseline_azel,
        pred_azel=pred_azel_raw,
        uncertainty=pred_unc,
        teacher_mask=teacher_mask,
        alpha_floor=alpha_floor,
        alpha_power=alpha_power,
        uncertainty_smooth=unc_smooth,
        use_ruptures_alpha_seg=use_rpt,
        ruptures_model=rpt_model,
        ruptures_penalty=rpt_pen,
        ruptures_min_size=rpt_min,
        ruptures_jump=rpt_jump,
    )
    alpha = float(meta.get("post_blend_alpha", 1.0))
    pred_azel = blend_azel_unitvec(ep.baseline_azel, pred_azel_dyn, alpha=alpha)
    cycle_alpha = float(meta.get("post_cycle_alpha", 0.0))
    if cycle_alpha > 0.0 and bool(meta.get("use_cycle_repeat_prior", False)):
        cycle_azel, _, _, _, _ = build_cycle_repeat_prior_azel_for_episode(
            ep=ep,
            observe_days=float(meta.get("cycle_observe_days", 3.0)),
            min_period_minutes=float(meta.get("cycle_min_period_minutes", 1100.0)),
            max_period_minutes=float(meta.get("cycle_max_period_minutes", 1800.0)),
        )
        pred_azel = blend_azel_unitvec(pred_azel, cycle_azel, alpha=cycle_alpha)
    lstm_err = compute_forecast_max_abs_error(ep.true_azel, pred_azel, ep.forecast_mask)
    base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
    return pred_azel, float(base_err), float(lstm_err)


def build_lla_residual_pack_from_tle(
    tle_path: Path,
    sat_name: str,
    truth_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    geo_radius_km: float,
    days: int,
    train_days: int,
    step_minutes: int,
    model,
    meta: dict,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    calendar_climatology_trig: np.ndarray | None,
) -> LlaResidualPack | None:
    window = exp.build_window(
        tle_path=tle_path,
        days=int(days),
        train_days=int(train_days),
        step_minutes=int(step_minutes),
    )
    truth_unix, truth_full = pick_truth_window(
        cache=truth_cache,
        start_unix=window.start_unix,
        end_unix=window.end_unix,
    )
    if truth_unix is None or truth_full is None:
        return None

    raw_azel, raw_lla = exp.propagate_tle_azel_lla_at_unix(
        tle_path=tle_path,
        sat_name=str(sat_name),
        observer_lat=float(observer_lat),
        observer_lon=float(observer_lon),
        unix=window.unix,
    )
    raw_full = np.column_stack([np.asarray(raw_lla, dtype=np.float64), np.asarray(raw_azel, dtype=np.float64)])
    base_aligned, true_aligned, unix_aligned = core.align_by_unix(window.unix, raw_full, truth_unix, truth_full)
    forecast_mask = np.asarray(unix_aligned >= float(window.train_end_unix), dtype=bool)
    expected_forecast = int(np.sum(np.asarray(window.unix >= float(window.train_end_unix), dtype=bool)))
    if int(np.sum(forecast_mask)) < expected_forecast:
        return None

    ep = Episode(
        tle_name=tle_path.name,
        unix=np.asarray(unix_aligned, dtype=np.float64),
        baseline_azel=np.asarray(base_aligned[:, 3:5], dtype=np.float64),
        true_azel=np.asarray(true_aligned[:, 3:5], dtype=np.float64),
        forecast_mask=np.asarray(forecast_mask, dtype=bool),
        static_features=np.asarray(exp.tle_static_features(tle_path=tle_path, sat_name=str(sat_name)), dtype=np.float64),
    )
    pred_azel, _, _ = predict_episode_with_model(
        model=model,
        meta=meta,
        x_mean=x_mean,
        x_std=x_std,
        ep=ep,
        warmup_weight=float(meta["warmup_weight"]),
        forecast_weight=float(meta["forecast_weight"]),
        dynamic_alpha_floor=float(meta.get("dynamic_alpha_floor", 0.05)),
        dynamic_alpha_power=float(meta.get("dynamic_alpha_power", 1.4)),
        uncertainty_smooth=int(meta.get("uncertainty_smooth", 11)),
        use_ruptures_alpha_seg=1 if bool(meta.get("use_ruptures_alpha_seg", False)) else 0,
        ruptures_model=str(meta.get("ruptures_model", "rbf")),
        ruptures_penalty=float(meta.get("ruptures_penalty", 8.0)),
        ruptures_min_size=int(meta.get("ruptures_min_size", 24)),
        ruptures_jump=int(meta.get("ruptures_jump", 5)),
        predict_batch_size=int(meta.get("predict_batch_size", 128)),
        calendar_climatology_trig=calendar_climatology_trig,
    )
    pred_full = build_pred_full_from_azel(
        pred_azel=pred_azel,
        observer_lat=float(observer_lat),
        observer_lon=float(observer_lon),
        observer_alt_m=float(observer_alt_m),
        geo_radius_km=float(geo_radius_km),
        fallback_lla=np.asarray(base_aligned[:, 0:3], dtype=np.float64),
    )
    return LlaResidualPack(
        tle_name=tle_path.name,
        unix=np.asarray(unix_aligned, dtype=np.float64),
        baseline_azel=np.asarray(base_aligned[:, 3:5], dtype=np.float64),
        true_azel=np.asarray(true_aligned[:, 3:5], dtype=np.float64),
        pred_azel=np.asarray(pred_azel, dtype=np.float64),
        baseline_lla=np.asarray(base_aligned[:, 0:3], dtype=np.float64),
        pred_lla_geo=np.asarray(pred_full[:, 0:3], dtype=np.float64),
        true_lla=np.asarray(true_aligned[:, 0:3], dtype=np.float64),
        forecast_mask=np.asarray(forecast_mask, dtype=bool),
        static_features=np.asarray(ep.static_features, dtype=np.float64),
    )


def build_lla_residual_training_samples(
    paths: Sequence[Path],
    sat_name: str,
    truth_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    geo_radius_km: float,
    days: int,
    train_days: int,
    step_minutes: int,
    model,
    meta: dict,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    calendar_climatology_trig: np.ndarray | None,
    sample_stride: int,
    max_files: int,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, list[LlaResidualPack]]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    packs: list[LlaResidualPack] = []
    use_paths = list(paths[: int(max_files)]) if int(max_files) > 0 else list(paths)
    stride = int(max(1, int(sample_stride)))
    for i, p in enumerate(use_paths, start=1):
        pack = build_lla_residual_pack_from_tle(
            tle_path=p,
            sat_name=str(sat_name),
            truth_cache=truth_cache,
            observer_lat=float(observer_lat),
            observer_lon=float(observer_lon),
            observer_alt_m=float(observer_alt_m),
            geo_radius_km=float(geo_radius_km),
            days=int(days),
            train_days=int(train_days),
            step_minutes=int(step_minutes),
            model=model,
            meta=meta,
            x_mean=x_mean,
            x_std=x_std,
            calendar_climatology_trig=calendar_climatology_trig,
        )
        if pack is None:
            print(f"[{tag} {i}/{len(use_paths)}] skip {p.name}", flush=True)
            continue
        idx = np.where(np.asarray(pack.forecast_mask, dtype=bool))[0]
        if idx.size == 0:
            continue
        idx = idx[::stride]
        feat = build_lla_residual_features(
            unix=np.asarray(pack.unix, dtype=np.float64),
            baseline_azel=np.asarray(pack.baseline_azel, dtype=np.float64),
            pred_azel=np.asarray(pack.pred_azel, dtype=np.float64),
            baseline_lla=np.asarray(pack.baseline_lla, dtype=np.float64),
            pred_lla_geo=np.asarray(pack.pred_lla_geo, dtype=np.float64),
            static_features=np.asarray(pack.static_features, dtype=np.float64),
            yearly_harmonics=int(meta.get("time_yearly_harmonics", 2)),
        )
        y_delta = np.column_stack(
            [
                np.asarray(pack.true_lla[:, 0] - pack.pred_lla_geo[:, 0], dtype=np.float64),
                np.asarray(core.angle_diff_deg(pack.true_lla[:, 1], pack.pred_lla_geo[:, 1]), dtype=np.float64),
                np.asarray(pack.true_lla[:, 2] - pack.pred_lla_geo[:, 2], dtype=np.float64),
            ]
        ).astype(np.float64)
        x_list.append(np.asarray(feat[idx], dtype=np.float32))
        y_list.append(np.asarray(y_delta[idx], dtype=np.float32))
        packs.append(pack)
        print(f"[{tag} {i}/{len(use_paths)}] {p.name} rows={int(idx.size)}", flush=True)
    if not x_list:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), packs
    x_out = np.concatenate(x_list, axis=0).astype(np.float32)
    y_out = np.concatenate(y_list, axis=0).astype(np.float32)
    return x_out, y_out, packs


def tune_single_alpha(errors_by_alpha: list[tuple[float, list[float]]]) -> float:
    if not errors_by_alpha:
        return 0.0
    ranked = sorted(
        errors_by_alpha,
        key=lambda t: (
            float(np.max(np.asarray(t[1], dtype=np.float64))) if t[1] else float("inf"),
            float(np.mean(np.asarray(t[1], dtype=np.float64))) if t[1] else float("inf"),
            float(t[0]),
        ),
    )
    return float(ranked[0][0])


def apply_lla_residual_correction_for_pack(
    pack: LlaResidualPack,
    lla_model: dict,
    alpha_lat: float,
    alpha_lon: float,
    alpha_alt: float,
    alpha_azel: float,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    yearly_harmonics: int,
) -> tuple[np.ndarray, np.ndarray]:
    feat = build_lla_residual_features(
        unix=np.asarray(pack.unix, dtype=np.float64),
        baseline_azel=np.asarray(pack.baseline_azel, dtype=np.float64),
        pred_azel=np.asarray(pack.pred_azel, dtype=np.float64),
        baseline_lla=np.asarray(pack.baseline_lla, dtype=np.float64),
        pred_lla_geo=np.asarray(pack.pred_lla_geo, dtype=np.float64),
        static_features=np.asarray(pack.static_features, dtype=np.float64),
        yearly_harmonics=int(yearly_harmonics),
    )
    d_hat = predict_linear_multi_ridge(lla_model, feat)
    lla_corr = apply_lla_delta_with_alpha(
        pred_lla=np.asarray(pack.pred_lla_geo, dtype=np.float64),
        delta_lla=np.asarray(d_hat, dtype=np.float64),
        alpha_lat=float(alpha_lat),
        alpha_lon=float(alpha_lon),
        alpha_alt=float(alpha_alt),
    )
    if float(alpha_azel) > 0.0:
        azel_from_lla = recompute_azel_from_lla_np(
            sat_lla=lla_corr,
            observer_lat=float(observer_lat),
            observer_lon=float(observer_lon),
            observer_alt_m=float(observer_alt_m),
        )
        azel_final = blend_azel_unitvec(np.asarray(pack.pred_azel, dtype=np.float64), azel_from_lla, alpha=float(alpha_azel))
    else:
        azel_final = np.asarray(pack.pred_azel, dtype=np.float64)
    return np.asarray(lla_corr, dtype=np.float64), np.asarray(azel_final, dtype=np.float64)


def run_eval_dir(args: argparse.Namespace) -> None:
    model_dir = ROOT / str(args.model_dir)
    model, meta, x_mean, x_std = load_model_and_meta(model_dir)
    calendar_clim = None
    if bool(meta.get("use_calendar_climatology", False)):
        calendar_clim = build_calendar_climatology_trig(
            years=[int(v) for v in meta.get("train_years", [])],
            suffix=str(meta.get("truth_suffix", "")),
        )

    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    rows = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue
        _, base_err, lstm_err = predict_episode_with_model(
            model=model,
            meta=meta,
            x_mean=x_mean,
            x_std=x_std,
            ep=ep,
            warmup_weight=float(meta["warmup_weight"]),
            forecast_weight=float(meta["forecast_weight"]),
            dynamic_alpha_floor=float(args.dynamic_alpha_floor),
            dynamic_alpha_power=float(args.dynamic_alpha_power),
            uncertainty_smooth=int(args.uncertainty_smooth),
            use_ruptures_alpha_seg=int(args.use_ruptures_alpha_seg),
            ruptures_model=str(args.ruptures_model),
            ruptures_penalty=float(args.ruptures_penalty),
            ruptures_min_size=int(args.ruptures_min_size),
            ruptures_jump=int(args.ruptures_jump),
            predict_batch_size=int(args.predict_batch_size),
            calendar_climatology_trig=calendar_clim,
        )
        rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "lstm_max_abs_error_max": float(lstm_err),
            }
        )
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} lstm={lstm_err:.12f}",
            flush=True,
        )
    rows.sort(key=lambda r: r["lstm_max_abs_error_max"], reverse=True)
    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in rows]),
        "lstm_summary": summarize([float(r["lstm_max_abs_error_max"]) for r in rows]),
        "rows": rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["lstm_summary"], ensure_ascii=False, indent=2))


def run_eval_cycle_repeat(args: argparse.Namespace) -> None:
    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    rows = []
    cache = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue
        cycle_azel, cycle_period, cycle_score, cycle_rmse_az, cycle_rmse_el = build_cycle_repeat_prior_azel_for_episode(
            ep=ep,
            observe_days=float(args.cycle_observe_days),
            min_period_minutes=float(args.cycle_min_period_minutes),
            max_period_minutes=float(args.cycle_max_period_minutes),
        )
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        cycle_err = compute_forecast_max_abs_error(ep.true_azel, cycle_azel, ep.forecast_mask)
        cache.append((ep, cycle_azel, float(base_err), float(cycle_err)))
        rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "cycle_repeat_max_abs_error_max": float(cycle_err),
                "cycle_period_minutes": float(cycle_period),
                "cycle_score": float(cycle_score),
                "cycle_fit_rmse_az": float(cycle_rmse_az),
                "cycle_fit_rmse_el": float(cycle_rmse_el),
            }
        )
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} cycle={cycle_err:.12f} period_min={cycle_period:.3f}",
            flush=True,
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    alpha_scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, cycle_azel, _, _ in cache:
            blended = blend_azel_unitvec(ep.baseline_azel, cycle_azel, alpha=float(alpha))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask))
        alpha_scores.append({"alpha": float(alpha), "summary": summarize(errs)})
    alpha_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_alpha = float(alpha_scores[0]["alpha"]) if alpha_scores else 0.0

    out_rows = []
    for row, (ep, cycle_azel, _, _) in zip(rows, cache):
        blended = blend_azel_unitvec(ep.baseline_azel, cycle_azel, alpha=best_alpha)
        blend_err = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
        row2 = dict(row)
        row2["cycle_blended_max_abs_error_max"] = float(blend_err)
        out_rows.append(row2)
    out_rows.sort(key=lambda r: r["cycle_blended_max_abs_error_max"], reverse=True)

    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(out_rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in out_rows]),
        "cycle_repeat_summary": summarize([float(r["cycle_repeat_max_abs_error_max"]) for r in out_rows]),
        "cycle_blended_summary": summarize([float(r["cycle_blended_max_abs_error_max"]) for r in out_rows]),
        "selected_alpha": float(best_alpha),
        "alpha_search_top5": alpha_scores[:5],
        "rows": out_rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["cycle_blended_summary"], ensure_ascii=False, indent=2))


def run_eval_online_reid(args: argparse.Namespace) -> None:
    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    mode = str(args.reid_mode).strip().lower()
    run_obs = mode in {"obs", "both"}
    run_self = mode in {"self", "both"}

    rows = []
    obs_items: list[tuple[int, Episode, np.ndarray]] = []
    self_items: list[tuple[int, Episode, np.ndarray]] = []

    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue

        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        row: dict[str, float | int | str] = {
            "tle_name": ep.tle_name,
            "baseline_max_abs_error_max": float(base_err),
        }

        if run_obs:
            pred_obs, diag_obs = build_online_reid_prediction(
                ep=ep,
                observe_days=float(args.reid_observe_days),
                mode="obs",
                sidereal_harmonics=int(args.reid_sidereal_harmonics),
                solar_harmonics=int(args.reid_solar_harmonics),
                ridge_lambda=float(args.reid_ridge_lambda),
                forgetting=float(args.reid_forgetting),
                ar_coeff=float(args.reid_ar_coeff),
            )
            err_obs = compute_forecast_max_abs_error(ep.true_azel, pred_obs, ep.forecast_mask)
            row["online_reid_obs_max_abs_error_max"] = float(err_obs)
            row["online_reid_obs_fit_rmse_az"] = float(diag_obs["fit_rmse_az"])
            row["online_reid_obs_fit_rmse_el"] = float(diag_obs["fit_rmse_el"])
            row["online_reid_obs_updates"] = int(diag_obs["updates"])
            row["online_reid_obs_observe_count"] = int(diag_obs["observe_count"])
            obs_items.append((int(len(rows)), ep, np.asarray(pred_obs, dtype=np.float64)))

        if run_self:
            pred_self, diag_self = build_online_reid_prediction(
                ep=ep,
                observe_days=float(args.reid_observe_days),
                mode="self",
                sidereal_harmonics=int(args.reid_sidereal_harmonics),
                solar_harmonics=int(args.reid_solar_harmonics),
                ridge_lambda=float(args.reid_ridge_lambda),
                forgetting=float(args.reid_forgetting),
                ar_coeff=float(args.reid_ar_coeff),
            )
            err_self = compute_forecast_max_abs_error(ep.true_azel, pred_self, ep.forecast_mask)
            row["online_reid_self_max_abs_error_max"] = float(err_self)
            row["online_reid_self_fit_rmse_az"] = float(diag_self["fit_rmse_az"])
            row["online_reid_self_fit_rmse_el"] = float(diag_self["fit_rmse_el"])
            row["online_reid_self_updates"] = int(diag_self["updates"])
            row["online_reid_self_observe_count"] = int(diag_self["observe_count"])
            self_items.append((int(len(rows)), ep, np.asarray(pred_self, dtype=np.float64)))

        rows.append(row)
        msg = f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f}"
        if run_obs:
            msg += f" obs={float(row['online_reid_obs_max_abs_error_max']):.12f}"
        if run_self:
            msg += f" self={float(row['online_reid_self_max_abs_error_max']):.12f}"
        print(msg, flush=True)

    payload = {
        "reid_mode": str(mode),
        "count_total": int(len(files)),
        "count_evaluated": int(len(rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in rows]),
        "params": {
            "reid_observe_days": float(args.reid_observe_days),
            "reid_sidereal_harmonics": int(args.reid_sidereal_harmonics),
            "reid_solar_harmonics": int(args.reid_solar_harmonics),
            "reid_ridge_lambda": float(args.reid_ridge_lambda),
            "reid_forgetting": float(args.reid_forgetting),
            "reid_ar_coeff": float(args.reid_ar_coeff),
        },
    }

    if obs_items:
        best_alpha_obs, obs_alpha_scores = search_best_blend_alpha([(ep, pred) for _, ep, pred in obs_items])
        obs_raw = []
        obs_blended = []
        for row_idx, ep, pred in obs_items:
            blended = blend_azel_unitvec(ep.baseline_azel, pred, alpha=float(best_alpha_obs))
            blend_err = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
            rows[row_idx]["online_reid_obs_blended_max_abs_error_max"] = float(blend_err)
            obs_raw.append(float(rows[row_idx]["online_reid_obs_max_abs_error_max"]))
            obs_blended.append(float(blend_err))
        payload["online_reid_obs_summary"] = summarize(obs_raw)
        payload["online_reid_obs_blended_summary"] = summarize(obs_blended)
        payload["selected_alpha_obs"] = float(best_alpha_obs)
        payload["alpha_search_obs_top5"] = obs_alpha_scores[:5]

    if self_items:
        best_alpha_self, self_alpha_scores = search_best_blend_alpha([(ep, pred) for _, ep, pred in self_items])
        self_raw = []
        self_blended = []
        for row_idx, ep, pred in self_items:
            blended = blend_azel_unitvec(ep.baseline_azel, pred, alpha=float(best_alpha_self))
            blend_err = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
            rows[row_idx]["online_reid_self_blended_max_abs_error_max"] = float(blend_err)
            self_raw.append(float(rows[row_idx]["online_reid_self_max_abs_error_max"]))
            self_blended.append(float(blend_err))
        payload["online_reid_self_summary"] = summarize(self_raw)
        payload["online_reid_self_blended_summary"] = summarize(self_blended)
        payload["selected_alpha_self"] = float(best_alpha_self)
        payload["alpha_search_self_top5"] = self_alpha_scores[:5]

    if obs_items:
        sort_key = "online_reid_obs_blended_max_abs_error_max"
    elif self_items:
        sort_key = "online_reid_self_blended_max_abs_error_max"
    else:
        sort_key = "baseline_max_abs_error_max"
    rows.sort(key=lambda r: float(r.get(sort_key, r["baseline_max_abs_error_max"])), reverse=True)
    payload["rows"] = rows

    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if "online_reid_obs_blended_summary" in payload:
        print(json.dumps(payload["online_reid_obs_blended_summary"], ensure_ascii=False, indent=2))
    elif "online_reid_self_blended_summary" in payload:
        print(json.dumps(payload["online_reid_self_blended_summary"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload["baseline_summary"], ensure_ascii=False, indent=2))


def run_eval_maneuver_hazard(args: argparse.Namespace) -> None:
    clim_years = parse_int_csv(args.climatology_years)
    trend_clim = build_calendar_trend_climatology_trig(
        years=clim_years,
        suffix=str(args.climatology_suffix),
        target_year=int(args.target_year),
    )
    if trend_clim is None:
        raise RuntimeError("Failed to build trend climatology")
    hazard_years = parse_int_csv(args.hazard_years)
    hazard_profile = build_calendar_maneuver_hazard_profile(
        years=hazard_years,
        suffix=str(args.hazard_suffix),
        lag_minutes=float(args.hazard_lag_minutes),
        smooth_minutes=int(args.hazard_smooth_minutes),
    )
    if hazard_profile is None:
        raise RuntimeError("Failed to build maneuver hazard profile")
    template_years = parse_int_csv(args.template_years)
    residual_template_trig, template_stats = build_cycle_residual_calendar_template_trig(
        sat_name=str(args.sat_name),
        years=template_years,
        truth_suffix=str(args.template_truth_suffix),
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        days=int(args.days),
        train_days=int(args.train_days),
        step_minutes=int(args.template_step_minutes),
        fit_days=float(args.fit_days),
        cycle_min_period_minutes=float(args.cycle_min_period_minutes),
        cycle_max_period_minutes=float(args.cycle_max_period_minutes),
        episode_stride=int(args.template_episode_stride),
        max_files=int(args.template_max_files),
        smooth_minutes=int(args.template_smooth_minutes),
    )
    if residual_template_trig is None:
        raise RuntimeError("Failed to build cycle residual template")

    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )

    rows = []
    cache = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue

        cycle_trig, _, cycle_period, cycle_score, cycle_rmse_az, cycle_rmse_el = build_cycle_repeat_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            fit_days=float(args.fit_days),
            min_period_minutes=float(args.cycle_min_period_minutes),
            max_period_minutes=float(args.cycle_max_period_minutes),
        )
        clim_trig, _, az_off, el_off = build_climatology_aligned_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            clim_trig=np.asarray(trend_clim, dtype=np.float32),
            fit_days=float(args.fit_days),
        )
        cycle_azel = np.asarray(hm.trig4_to_azel(cycle_trig), dtype=np.float64)
        clim_azel = np.asarray(hm.trig4_to_azel(clim_trig), dtype=np.float64)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        cycle_err = compute_forecast_max_abs_error(ep.true_azel, cycle_azel, ep.forecast_mask)
        clim_err = compute_forecast_max_abs_error(ep.true_azel, clim_azel, ep.forecast_mask)

        row = {
            "tle_name": ep.tle_name,
            "baseline_max_abs_error_max": float(base_err),
            "cycle_repeat_max_abs_error_max": float(cycle_err),
            "clim_trend_align_max_abs_error_max": float(clim_err),
            "cycle_period_minutes": float(cycle_period),
            "cycle_score": float(cycle_score),
            "cycle_fit_rmse_az": float(cycle_rmse_az),
            "cycle_fit_rmse_el": float(cycle_rmse_el),
            "az_offset_deg": float(az_off),
            "el_offset_deg": float(el_off),
        }
        rows.append(row)
        cache.append(
            {
                "row_idx": int(len(rows) - 1),
                "ep": ep,
                "cycle_trig": np.asarray(cycle_trig, dtype=np.float32),
                "clim_trig": np.asarray(clim_trig, dtype=np.float32),
                "template_trig": np.asarray(
                    lookup_calendar_climatology_trig(
                        unix=np.asarray(ep.unix, dtype=np.float64),
                        clim_trig=np.asarray(residual_template_trig, dtype=np.float32),
                    ),
                    dtype=np.float32,
                ),
            }
        )
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} cycle={cycle_err:.12f} clim={clim_err:.12f}",
            flush=True,
        )

    if not cache:
        payload = {
            "count_total": int(len(files)),
            "count_evaluated": 0,
            "rows": [],
        }
        out = ROOT / str(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    alpha_low_grid = parse_float_csv(args.alpha_low_grid)
    if not alpha_low_grid:
        alpha_low_grid = [0.0]
    alpha_high_grid = parse_float_csv(args.alpha_high_grid)
    if not alpha_high_grid:
        alpha_high_grid = [1.0]
    gamma_grid = parse_float_csv(args.gamma_grid)
    if not gamma_grid:
        gamma_grid = [1.0]
    gain_grid = parse_float_csv(args.residual_gain_grid)
    if not gain_grid:
        gain_grid = [1.0]

    tune_n = int(len(cache)) if int(args.tune_max_files) <= 0 else int(min(len(cache), int(args.tune_max_files)))
    tune_items = cache[:tune_n]
    search_rows = []
    for a0 in alpha_low_grid:
        for a1 in alpha_high_grid:
            if float(a1) + 1.0e-12 < float(a0):
                continue
            for gm in gamma_grid:
                for gain in gain_grid:
                    errs = []
                    for item in tune_items:
                        ep = item["ep"]
                        pred_trig, _, _ = build_hazard_hybrid_prior_trig(
                            unix=np.asarray(ep.unix, dtype=np.float64),
                            cycle_trig=np.asarray(item["cycle_trig"], dtype=np.float32),
                            clim_aligned_trig=np.asarray(item["clim_trig"], dtype=np.float32),
                            residual_template_trig=np.asarray(item["template_trig"], dtype=np.float32),
                            hazard_profile=np.asarray(hazard_profile, dtype=np.float32),
                            alpha_low=float(a0),
                            alpha_high=float(a1),
                            gamma=float(gm),
                            gain=float(gain),
                        )
                        pred_azel = np.asarray(hm.trig4_to_azel(pred_trig), dtype=np.float64)
                        err = compute_forecast_max_abs_error(ep.true_azel, pred_azel, ep.forecast_mask)
                        errs.append(float(err))
                    search_rows.append(
                        {
                            "alpha_low": float(a0),
                            "alpha_high": float(a1),
                            "gamma": float(gm),
                            "gain": float(gain),
                            "summary": summarize(errs),
                        }
                    )
    search_rows.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best = search_rows[0] if search_rows else {"alpha_low": 0.0, "alpha_high": 1.0, "gamma": 1.0}

    tune_preds = []
    for item in tune_items:
        ep = item["ep"]
        pred_trig, _, _ = build_hazard_hybrid_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            cycle_trig=np.asarray(item["cycle_trig"], dtype=np.float32),
            clim_aligned_trig=np.asarray(item["clim_trig"], dtype=np.float32),
            residual_template_trig=np.asarray(item["template_trig"], dtype=np.float32),
            hazard_profile=np.asarray(hazard_profile, dtype=np.float32),
            alpha_low=float(best["alpha_low"]),
            alpha_high=float(best["alpha_high"]),
            gamma=float(best["gamma"]),
            gain=float(best.get("gain", 1.0)),
        )
        pred_azel = np.asarray(hm.trig4_to_azel(pred_trig), dtype=np.float64)
        tune_preds.append((ep, pred_azel))

    beta_grid = [i / 20.0 for i in range(21)]
    beta_scores = []
    for beta in beta_grid:
        errs = []
        for ep, pred_azel in tune_preds:
            b = float(beta)
            if b <= 1.0e-12:
                fused = np.asarray(ep.baseline_azel, dtype=np.float64)
            elif b >= 1.0 - 1.0e-12:
                fused = np.asarray(pred_azel, dtype=np.float64)
            else:
                fused = blend_azel_unitvec(ep.baseline_azel, pred_azel, alpha=b)
            err = compute_forecast_max_abs_error(ep.true_azel, fused, ep.forecast_mask)
            errs.append(float(err))
        beta_scores.append({"beta": float(beta), "summary": summarize(errs)})
    beta_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_beta = float(beta_scores[0]["beta"]) if beta_scores else 1.0

    for item in cache:
        ep = item["ep"]
        pred_trig, alpha_vec, hz_vec = build_hazard_hybrid_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            cycle_trig=np.asarray(item["cycle_trig"], dtype=np.float32),
            clim_aligned_trig=np.asarray(item["clim_trig"], dtype=np.float32),
            residual_template_trig=np.asarray(item["template_trig"], dtype=np.float32),
            hazard_profile=np.asarray(hazard_profile, dtype=np.float32),
            alpha_low=float(best["alpha_low"]),
            alpha_high=float(best["alpha_high"]),
            gamma=float(best["gamma"]),
            gain=float(best.get("gain", 1.0)),
        )
        pred_azel = np.asarray(hm.trig4_to_azel(pred_trig), dtype=np.float64)
        raw_err = compute_forecast_max_abs_error(ep.true_azel, pred_azel, ep.forecast_mask)
        if float(best_beta) <= 1.0e-12:
            fused = np.asarray(ep.baseline_azel, dtype=np.float64)
        elif float(best_beta) >= 1.0 - 1.0e-12:
            fused = np.asarray(pred_azel, dtype=np.float64)
        else:
            fused = blend_azel_unitvec(ep.baseline_azel, pred_azel, alpha=float(best_beta))
        fused_err = compute_forecast_max_abs_error(ep.true_azel, fused, ep.forecast_mask)

        row = rows[int(item["row_idx"])]
        row["maneuver_hazard_max_abs_error_max"] = float(raw_err)
        row["maneuver_hazard_blended_max_abs_error_max"] = float(fused_err)
        row["maneuver_hazard_alpha_mean"] = float(np.mean(alpha_vec))
        row["maneuver_hazard_alpha_max"] = float(np.max(alpha_vec))
        row["maneuver_hazard_hazard_mean"] = float(np.mean(hz_vec))
        row["maneuver_hazard_hazard_max"] = float(np.max(hz_vec))

    rows.sort(key=lambda r: r["maneuver_hazard_blended_max_abs_error_max"], reverse=True)
    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in rows]),
        "cycle_repeat_summary": summarize([float(r["cycle_repeat_max_abs_error_max"]) for r in rows]),
        "clim_trend_align_summary": summarize([float(r["clim_trend_align_max_abs_error_max"]) for r in rows]),
        "maneuver_hazard_summary": summarize([float(r["maneuver_hazard_max_abs_error_max"]) for r in rows]),
        "maneuver_hazard_blended_summary": summarize(
            [float(r["maneuver_hazard_blended_max_abs_error_max"]) for r in rows]
        ),
        "selected_hazard_params": {
            "alpha_low": float(best["alpha_low"]),
            "alpha_high": float(best["alpha_high"]),
            "gamma": float(best["gamma"]),
            "gain": float(best.get("gain", 1.0)),
        },
        "selected_baseline_blend_beta": float(best_beta),
        "hazard_profile_summary": summarize(np.asarray(hazard_profile, dtype=np.float64).tolist()),
        "template_stats": template_stats,
        "grid_search_top10": search_rows[:10],
        "beta_search_top10": beta_scores[:10],
        "rows": rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["maneuver_hazard_blended_summary"], ensure_ascii=False, indent=2))


def run_eval_periodic_ls(args: argparse.Namespace) -> None:
    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    rows = []
    cache = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue
        prior_trig, _, fit_rmse_az, fit_rmse_el = build_periodic_drift_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            fit_days=float(args.pseudo_observe_days),
            harmonics=int(args.periodic_ls_harmonics),
        )
        prior_azel = np.asarray(hm.trig4_to_azel(prior_trig), dtype=np.float64)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        prior_err = compute_forecast_max_abs_error(ep.true_azel, prior_azel, ep.forecast_mask)
        cache.append((ep, prior_azel, float(base_err), float(prior_err)))
        rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "periodic_ls_max_abs_error_max": float(prior_err),
                "fit_rmse_az": float(fit_rmse_az),
                "fit_rmse_el": float(fit_rmse_el),
            }
        )
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} periodic_ls={prior_err:.12f}",
            flush=True,
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    alpha_scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, prior_azel, _, _ in cache:
            blended = blend_azel_unitvec(ep.baseline_azel, prior_azel, alpha=float(alpha))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask))
        alpha_scores.append({"alpha": float(alpha), "summary": summarize(errs)})
    alpha_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_alpha = float(alpha_scores[0]["alpha"]) if alpha_scores else 0.0

    out_rows = []
    for row, (ep, prior_azel, _, _) in zip(rows, cache):
        blended = blend_azel_unitvec(ep.baseline_azel, prior_azel, alpha=best_alpha)
        blend_err = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
        row2 = dict(row)
        row2["periodic_ls_blended_max_abs_error_max"] = float(blend_err)
        out_rows.append(row2)
    out_rows.sort(key=lambda r: r["periodic_ls_blended_max_abs_error_max"], reverse=True)

    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(out_rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in out_rows]),
        "periodic_ls_summary": summarize([float(r["periodic_ls_max_abs_error_max"]) for r in out_rows]),
        "periodic_ls_blended_summary": summarize([float(r["periodic_ls_blended_max_abs_error_max"]) for r in out_rows]),
        "selected_alpha": float(best_alpha),
        "alpha_search_top5": alpha_scores[:5],
        "rows": out_rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["periodic_ls_blended_summary"], ensure_ascii=False, indent=2))


def run_eval_climatology_align(args: argparse.Namespace) -> None:
    clim_years = parse_int_csv(args.climatology_years)
    clim = build_calendar_climatology_trig(years=clim_years, suffix=str(args.climatology_suffix))
    if clim is None:
        raise RuntimeError("Failed to build climatology")
    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    rows = []
    cache = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue
        prior_trig, _, az_off, el_off = build_climatology_aligned_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            clim_trig=np.asarray(clim, dtype=np.float32),
            fit_days=float(args.fit_days),
        )
        prior_azel = np.asarray(hm.trig4_to_azel(prior_trig), dtype=np.float64)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        clim_err = compute_forecast_max_abs_error(ep.true_azel, prior_azel, ep.forecast_mask)
        cache.append((ep, prior_azel, float(base_err), float(clim_err)))
        rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "clim_align_max_abs_error_max": float(clim_err),
                "az_offset_deg": float(az_off),
                "el_offset_deg": float(el_off),
            }
        )
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} clim={clim_err:.12f} az_off={az_off:.4f} el_off={el_off:.4f}",
            flush=True,
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    alpha_scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, prior_azel, _, _ in cache:
            blended = blend_azel_unitvec(ep.baseline_azel, prior_azel, alpha=float(alpha))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask))
        alpha_scores.append({"alpha": float(alpha), "summary": summarize(errs)})
    alpha_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_alpha = float(alpha_scores[0]["alpha"]) if alpha_scores else 0.0

    out_rows = []
    for row, (ep, prior_azel, _, _) in zip(rows, cache):
        blended = blend_azel_unitvec(ep.baseline_azel, prior_azel, alpha=best_alpha)
        blend_err = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
        row2 = dict(row)
        row2["clim_align_blended_max_abs_error_max"] = float(blend_err)
        out_rows.append(row2)
    out_rows.sort(key=lambda r: r["clim_align_blended_max_abs_error_max"], reverse=True)
    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(out_rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in out_rows]),
        "clim_align_summary": summarize([float(r["clim_align_max_abs_error_max"]) for r in out_rows]),
        "clim_align_blended_summary": summarize([float(r["clim_align_blended_max_abs_error_max"]) for r in out_rows]),
        "selected_alpha": float(best_alpha),
        "alpha_search_top5": alpha_scores[:5],
        "rows": out_rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["clim_align_blended_summary"], ensure_ascii=False, indent=2))


def run_eval_cycle_clim_ensemble(args: argparse.Namespace) -> None:
    clim_years = parse_int_csv(args.climatology_years)
    if int(args.use_trend_clim) == 1:
        clim = build_calendar_trend_climatology_trig(
            years=clim_years,
            suffix=str(args.climatology_suffix),
            target_year=int(args.target_year),
        )
    else:
        clim = build_calendar_climatology_trig(years=clim_years, suffix=str(args.climatology_suffix))
    if clim is None:
        raise RuntimeError("Failed to build climatology")
    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    cache = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue
        cyc_trig, _, _, _, _, _ = build_cycle_repeat_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            fit_days=float(args.fit_days),
            min_period_minutes=float(args.cycle_min_period_minutes),
            max_period_minutes=float(args.cycle_max_period_minutes),
        )
        clim_trig, _, _, _ = build_climatology_aligned_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            clim_trig=np.asarray(clim, dtype=np.float32),
            fit_days=float(args.fit_days),
        )
        cyc_azel = np.asarray(hm.trig4_to_azel(cyc_trig), dtype=np.float64)
        cal_azel = np.asarray(hm.trig4_to_azel(clim_trig), dtype=np.float64)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        cyc_err = compute_forecast_max_abs_error(ep.true_azel, cyc_azel, ep.forecast_mask)
        cal_err = compute_forecast_max_abs_error(ep.true_azel, cal_azel, ep.forecast_mask)
        cache.append((ep, cyc_azel, cal_azel, float(base_err), float(cyc_err), float(cal_err)))
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} cycle={cyc_err:.12f} clim={cal_err:.12f}",
            flush=True,
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    scores = []
    for ac in alpha_grid:
        errs = []
        for ep, cyc_azel, cal_azel, _, _, _ in cache:
            ens = blend_azel_unitvec(cyc_azel, cal_azel, alpha=float(ac))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, ens, ep.forecast_mask))
        scores.append({"alpha_cycle_to_clim": float(ac), "summary": summarize(errs)})
    scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_alpha = float(scores[0]["alpha_cycle_to_clim"]) if scores else 0.0

    rows = []
    for ep, cyc_azel, cal_azel, base_err, cyc_err, cal_err in cache:
        ens = blend_azel_unitvec(cyc_azel, cal_azel, alpha=best_alpha)
        ens_err = compute_forecast_max_abs_error(ep.true_azel, ens, ep.forecast_mask)
        rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "cycle_repeat_max_abs_error_max": float(cyc_err),
                "clim_align_max_abs_error_max": float(cal_err),
                "ensemble_max_abs_error_max": float(ens_err),
            }
        )
    rows.sort(key=lambda r: r["ensemble_max_abs_error_max"], reverse=True)
    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in rows]),
        "cycle_summary": summarize([float(r["cycle_repeat_max_abs_error_max"]) for r in rows]),
        "climatology_summary": summarize([float(r["clim_align_max_abs_error_max"]) for r in rows]),
        "ensemble_summary": summarize([float(r["ensemble_max_abs_error_max"]) for r in rows]),
        "selected_alpha_cycle_to_clim": float(best_alpha),
        "alpha_search_top5": scores[:5],
        "rows": rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["ensemble_summary"], ensure_ascii=False, indent=2))


def run_eval_climatology_trend_align(args: argparse.Namespace) -> None:
    clim_years = parse_int_csv(args.climatology_years)
    trend_clim = build_calendar_trend_climatology_trig(
        years=clim_years,
        suffix=str(args.climatology_suffix),
        target_year=int(args.target_year),
    )
    if trend_clim is None:
        raise RuntimeError("Failed to build trend climatology")
    truth_years = parse_int_csv(args.truth_years)
    truth_cache = load_truth_cache(truth_years, suffix=str(args.truth_suffix))
    files = collect_tle_files(
        tle_dir=str(args.tle_dir),
        max_files=int(args.max_files),
        specific=parse_str_csv(args.specific_files),
    )
    rows = []
    cache = []
    for i, p in enumerate(files, start=1):
        ep = build_episode(
            tle_path=p,
            sat_name=str(args.sat_name),
            truth_cache=truth_cache,
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            days=int(args.days),
            train_days=int(args.train_days),
            step_minutes=int(args.step_minutes),
        )
        if ep is None:
            print(f"[{i}/{len(files)}] skip {p.name}", flush=True)
            continue
        prior_trig, _, az_off, el_off = build_climatology_aligned_prior_trig(
            unix=np.asarray(ep.unix, dtype=np.float64),
            base_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
            clim_trig=np.asarray(trend_clim, dtype=np.float32),
            fit_days=float(args.fit_days),
        )
        prior_azel = np.asarray(hm.trig4_to_azel(prior_trig), dtype=np.float64)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        err = compute_forecast_max_abs_error(ep.true_azel, prior_azel, ep.forecast_mask)
        cache.append((ep, prior_azel, float(base_err), float(err)))
        rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "clim_trend_align_max_abs_error_max": float(err),
                "az_offset_deg": float(az_off),
                "el_offset_deg": float(el_off),
            }
        )
        print(
            f"[{i}/{len(files)}] {ep.tle_name} baseline={base_err:.12f} clim_trend={err:.12f} az_off={az_off:.4f} el_off={el_off:.4f}",
            flush=True,
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    alpha_scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, prior_azel, _, _ in cache:
            blended = blend_azel_unitvec(ep.baseline_azel, prior_azel, alpha=float(alpha))
            errs.append(compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask))
        alpha_scores.append({"alpha": float(alpha), "summary": summarize(errs)})
    alpha_scores.sort(
        key=lambda r: (
            float(r["summary"]["max_max_abs_error"]),
            float(r["summary"]["mean_max_abs_error"]),
        )
    )
    best_alpha = float(alpha_scores[0]["alpha"]) if alpha_scores else 0.0
    out_rows = []
    for row, (ep, prior_azel, _, _) in zip(rows, cache):
        blended = blend_azel_unitvec(ep.baseline_azel, prior_azel, alpha=best_alpha)
        blend_err = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
        row2 = dict(row)
        row2["clim_trend_align_blended_max_abs_error_max"] = float(blend_err)
        out_rows.append(row2)
    out_rows.sort(key=lambda r: r["clim_trend_align_blended_max_abs_error_max"], reverse=True)
    payload = {
        "count_total": int(len(files)),
        "count_evaluated": int(len(out_rows)),
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in out_rows]),
        "clim_trend_align_summary": summarize([float(r["clim_trend_align_max_abs_error_max"]) for r in out_rows]),
        "clim_trend_align_blended_summary": summarize([float(r["clim_trend_align_blended_max_abs_error_max"]) for r in out_rows]),
        "selected_alpha": float(best_alpha),
        "alpha_search_top5": alpha_scores[:5],
        "rows": out_rows,
    }
    out = ROOT / str(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["clim_trend_align_blended_summary"], ensure_ascii=False, indent=2))


def run_predict(args: argparse.Namespace) -> None:
    model_dir = ROOT / str(args.model_dir)
    model, meta, x_mean, x_std = load_model_and_meta(model_dir)
    calendar_clim = None
    if bool(meta.get("use_calendar_climatology", False)):
        calendar_clim = build_calendar_climatology_trig(
            years=[int(v) for v in meta.get("train_years", [])],
            suffix=str(meta.get("truth_suffix", "")),
        )
    tle_path = ROOT / str(args.tle_file)
    if not tle_path.exists():
        raise FileNotFoundError(f"TLE file not found: {tle_path}")

    truth_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if str(args.truth_years).strip():
        truth_cache = load_truth_cache(parse_int_csv(args.truth_years), suffix=str(args.truth_suffix))

    window = exp.build_window(
        tle_path=tle_path,
        days=int(args.days),
        train_days=int(args.train_days),
        step_minutes=int(args.step_minutes),
    )
    raw_azel, raw_lla = exp.propagate_tle_azel_lla_at_unix(
        tle_path=tle_path,
        sat_name=str(args.sat_name),
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        unix=window.unix,
    )
    static = np.asarray(exp.tle_static_features(tle_path=tle_path, sat_name=str(args.sat_name)), dtype=np.float64)
    ep = Episode(
        tle_name=tle_path.name,
        unix=np.asarray(window.unix, dtype=np.float64),
        baseline_azel=np.asarray(raw_azel, dtype=np.float64),
        true_azel=np.asarray(raw_azel, dtype=np.float64),
        forecast_mask=np.asarray(window.unix >= float(window.train_end_unix), dtype=bool),
        static_features=static,
    )
    x, _, _ = build_step_arrays(
        ep=ep,
        days=int(meta["days"]),
        time_yearly_harmonics=int(meta["time_yearly_harmonics"]),
        time_feature_mode=str(meta.get("time_feature_mode", "periodic")),
        use_extra_sincos_features=bool(meta.get("use_extra_sincos_features", False)),
        use_cycle_repeat_prior=bool(meta.get("use_cycle_repeat_prior", False)),
        cycle_observe_days=float(meta.get("cycle_observe_days", 3.0)),
        cycle_min_period_minutes=float(meta.get("cycle_min_period_minutes", 1100.0)),
        cycle_max_period_minutes=float(meta.get("cycle_max_period_minutes", 1800.0)),
        calendar_climatology_trig=calendar_clim,
        add_baseline_periodic_harmonics=int(meta.get("add_baseline_periodic_harmonics", 0)),
        pseudo_observe_days=float(meta.get("pseudo_observe_days", 3.0)),
        periodic_ls_harmonics=int(meta.get("periodic_ls_harmonics", 2)),
        teacher_force_days=float(meta.get("teacher_force_days", 3.0)),
        warmup_weight=float(meta["warmup_weight"]),
        forecast_weight=float(meta["forecast_weight"]),
    )
    x_aligned, align_info = align_features_to_model_input(x=x, x_mean=x_mean)
    xn = ((x_aligned - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    pred_trig4, pred_unc = predict_full_pack(
        model=model,
        x_norm=xn,
        seq_len=int(meta["seq_len"]),
        seq_stride=int(meta["seq_stride"]),
        predict_batch_size=int(args.predict_batch_size),
    )
    pred_azel_raw = hm.trig4_to_azel(pred_trig4)
    alpha_floor = float(meta.get("dynamic_alpha_floor", 0.05))
    if float(args.dynamic_alpha_floor) >= 0.0:
        alpha_floor = float(args.dynamic_alpha_floor)
    alpha_power = float(meta.get("dynamic_alpha_power", 1.4))
    if float(args.dynamic_alpha_power) >= 0.0:
        alpha_power = float(args.dynamic_alpha_power)
    unc_smooth = int(meta.get("uncertainty_smooth", 11))
    if int(args.uncertainty_smooth) > 0:
        unc_smooth = int(args.uncertainty_smooth)
    use_rpt = bool(meta.get("use_ruptures_alpha_seg", False))
    if int(args.use_ruptures_alpha_seg) >= 0:
        use_rpt = bool(int(args.use_ruptures_alpha_seg) == 1)
    rpt_model = str(meta.get("ruptures_model", "rbf"))
    if str(args.ruptures_model).strip():
        rpt_model = str(args.ruptures_model)
    rpt_pen = float(meta.get("ruptures_penalty", 8.0))
    if float(args.ruptures_penalty) > 0.0:
        rpt_pen = float(args.ruptures_penalty)
    rpt_min = int(meta.get("ruptures_min_size", 24))
    if int(args.ruptures_min_size) > 0:
        rpt_min = int(args.ruptures_min_size)
    rpt_jump = int(meta.get("ruptures_jump", 5))
    if int(args.ruptures_jump) > 0:
        rpt_jump = int(args.ruptures_jump)
    teacher_mask = compute_teacher_mask(unix=ep.unix, teacher_force_days=float(meta.get("teacher_force_days", 3.0)))
    pred_azel_dyn, alpha_dyn, unc_dyn = apply_dynamic_uncertainty_blend(
        base_azel=ep.baseline_azel,
        pred_azel=pred_azel_raw,
        uncertainty=pred_unc,
        teacher_mask=teacher_mask,
        alpha_floor=alpha_floor,
        alpha_power=alpha_power,
        uncertainty_smooth=unc_smooth,
        use_ruptures_alpha_seg=use_rpt,
        ruptures_model=rpt_model,
        ruptures_penalty=rpt_pen,
        ruptures_min_size=rpt_min,
        ruptures_jump=rpt_jump,
    )
    post_alpha = float(meta.get("post_blend_alpha", 1.0))
    pred_azel = blend_azel_unitvec(ep.baseline_azel, pred_azel_dyn, alpha=post_alpha)
    post_cycle_alpha = float(meta.get("post_cycle_alpha", 0.0))
    if post_cycle_alpha > 0.0 and bool(meta.get("use_cycle_repeat_prior", False)):
        cycle_azel, _, _, _, _ = build_cycle_repeat_prior_azel_for_episode(
            ep=ep,
            observe_days=float(meta.get("cycle_observe_days", 3.0)),
            min_period_minutes=float(meta.get("cycle_min_period_minutes", 1100.0)),
            max_period_minutes=float(meta.get("cycle_max_period_minutes", 1800.0)),
        )
        pred_azel = blend_azel_unitvec(pred_azel, cycle_azel, alpha=post_cycle_alpha)

    pred_full = build_pred_full_from_azel(
        pred_azel=pred_azel,
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        observer_alt_m=float(args.observer_alt_m),
        geo_radius_km=float(args.geo_radius_km),
        fallback_lla=np.asarray(raw_lla, dtype=np.float64),
    )
    lla_corr_info = None
    lla_corr_cfg = meta.get("lla_residual_correction")
    if isinstance(lla_corr_cfg, dict) and bool(lla_corr_cfg.get("enabled", False)) and isinstance(lla_corr_cfg.get("model"), dict):
        try:
            feat_lla = build_lla_residual_features(
                unix=np.asarray(ep.unix, dtype=np.float64),
                baseline_azel=np.asarray(ep.baseline_azel, dtype=np.float64),
                pred_azel=np.asarray(pred_azel, dtype=np.float64),
                baseline_lla=np.asarray(raw_lla, dtype=np.float64),
                pred_lla_geo=np.asarray(pred_full[:, 0:3], dtype=np.float64),
                static_features=np.asarray(ep.static_features, dtype=np.float64),
                yearly_harmonics=int(meta.get("time_yearly_harmonics", 2)),
            )
            d1 = predict_linear_multi_ridge(lla_corr_cfg["model"], feat_lla)
            d_total = np.zeros_like(np.asarray(d1, dtype=np.float64), dtype=np.float64)
            d_total[:, 0] = float(lla_corr_cfg.get("alpha_lat", 0.0)) * np.asarray(d1[:, 0], dtype=np.float64)
            d_total[:, 1] = float(lla_corr_cfg.get("alpha_lon", 0.0)) * np.asarray(d1[:, 1], dtype=np.float64)
            d_total[:, 2] = float(lla_corr_cfg.get("alpha_alt", 0.0)) * np.asarray(d1[:, 2], dtype=np.float64)

            if isinstance(lla_corr_cfg.get("model_stage2"), dict):
                feat2 = build_lla_stage2_features(
                    x_lla_feat=np.asarray(feat_lla, dtype=np.float64),
                    max_base_features=int(lla_corr_cfg.get("stage2_max_base_features", 24)),
                    pair_features=int(lla_corr_cfg.get("stage2_pair_features", 8)),
                )
                d2 = predict_linear_multi_ridge(lla_corr_cfg["model_stage2"], feat2)
                d_total[:, 0] += float(lla_corr_cfg.get("beta_lat", 0.0)) * np.asarray(d2[:, 0], dtype=np.float64)
                d_total[:, 1] += float(lla_corr_cfg.get("beta_lon", 0.0)) * np.asarray(d2[:, 1], dtype=np.float64)
                d_total[:, 2] += float(lla_corr_cfg.get("beta_alt", 0.0)) * np.asarray(d2[:, 2], dtype=np.float64)

            d_total = anchor_delta_with_warmup(
                delta_lla=np.asarray(d_total, dtype=np.float64),
                unix=np.asarray(ep.unix, dtype=np.float64),
                teacher_force_days=float(args.train_days),
                anchor_scale=float(lla_corr_cfg.get("anchor_scale", 0.0)),
            )
            lla_corr = apply_lla_delta_with_alpha(
                pred_lla=np.asarray(pred_full[:, 0:3], dtype=np.float64),
                delta_lla=np.asarray(d_total, dtype=np.float64),
                alpha_lat=1.0,
                alpha_lon=1.0,
                alpha_alt=1.0,
            )
            azel_final = np.asarray(pred_azel, dtype=np.float64)
            alpha_azel = float(lla_corr_cfg.get("alpha_azel", 0.0))
            if alpha_azel > 0.0:
                az_from_lla = recompute_azel_from_lla_np(
                    sat_lla=lla_corr,
                    observer_lat=float(args.observer_lat),
                    observer_lon=float(args.observer_lon),
                    observer_alt_m=float(args.observer_alt_m),
                )
                azel_final = blend_azel_unitvec(np.asarray(pred_azel, dtype=np.float64), az_from_lla, alpha=alpha_azel)
            pred_full[:, 0:3] = np.asarray(lla_corr, dtype=np.float64)
            pred_full[:, 3:5] = np.asarray(azel_final, dtype=np.float64)
            pred_azel = np.asarray(azel_final, dtype=np.float64)
            lla_corr_info = {
                "enabled": True,
                "alpha_lat": float(lla_corr_cfg.get("alpha_lat", 0.0)),
                "alpha_lon": float(lla_corr_cfg.get("alpha_lon", 0.0)),
                "alpha_alt": float(lla_corr_cfg.get("alpha_alt", 0.0)),
                "beta_lat": float(lla_corr_cfg.get("beta_lat", 0.0)),
                "beta_lon": float(lla_corr_cfg.get("beta_lon", 0.0)),
                "beta_alt": float(lla_corr_cfg.get("beta_alt", 0.0)),
                "alpha_azel": float(alpha_azel),
                "anchor_scale": float(lla_corr_cfg.get("anchor_scale", 0.0)),
            }
        except Exception as ex:
            lla_corr_info = {"enabled": False, "error": str(ex)}

    out = ROOT / str(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["unix,lat_deg,lon_deg,alt_km,az_deg,el_deg,alpha_dynamic,uncertainty_smooth"]
    for u, v, al, uc in zip(window.unix.tolist(), pred_full.tolist(), alpha_dyn.tolist(), unc_dyn.tolist()):
        lines.append(
            f"{int(round(float(u)))},{float(v[0]):.10f},{float(v[1]):.10f},{float(v[2]):.10f},"
            f"{float(v[3]):.10f},{float(v[4]):.10f},{float(al):.6f},{float(uc):.6f}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if truth_cache:
        truth_unix, truth_full = pick_truth_window(
            cache=truth_cache,
            start_unix=window.start_unix,
            end_unix=window.end_unix,
        )
        if truth_unix is not None and truth_full is not None:
            pred_aligned, true_aligned, unix_aligned = core.align_by_unix(window.unix, pred_full, truth_unix, truth_full)
            forecast_mask = np.asarray(unix_aligned >= float(window.train_end_unix), dtype=bool)
            pred_forecast = np.asarray(pred_aligned[forecast_mask], dtype=np.float64)
            true_forecast = np.asarray(true_aligned[forecast_mask], dtype=np.float64)
            unix_forecast = np.asarray(unix_aligned[forecast_mask], dtype=np.float64)
            azel_metrics = core.compute_metrics_azel(true_forecast, pred_forecast)
            lla_metrics = core.compute_metrics_lla(true_forecast, pred_forecast)
            compare_png = out.with_name(f"{out.stem}_forecast_compare.png")
            error_png = out.with_name(f"{out.stem}_forecast_error.png")
            plot_paths = plot_full_target_compare_and_error(
                unix=unix_forecast,
                y_true_full=true_forecast,
                y_pred_full=pred_forecast,
                out_compare_png=compare_png,
                out_error_png=error_png,
                max_points=int(args.max_plot_points),
            )
            payload = {
                "tle_file": tle_path.name,
                "output_csv": str(out),
                "forecast_metrics_azel": azel_metrics,
                "forecast_metrics_lla": lla_metrics,
                "forecast_max_abs_error_max": float(azel_metrics["overall"]["max_abs_error_max"]),
                "forecast_lla_max_abs_error_max": float(lla_metrics["overall"]["max_abs_error_max"]),
                "forecast_error_plots": [str(p) for p in plot_paths],
            }
            if align_info is not None:
                payload["feature_alignment"] = align_info
            if lla_corr_info is not None:
                payload["lla_residual_correction"] = lla_corr_info
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
    payload = {"tle_file": tle_path.name, "output_csv": str(out)}
    if align_info is not None:
        payload["feature_alignment"] = align_info
    if lla_corr_info is not None:
        payload["lla_residual_correction"] = lla_corr_info
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "eval-dir":
        run_eval_dir(args)
    elif args.cmd == "eval-cycle-repeat":
        run_eval_cycle_repeat(args)
    elif args.cmd == "eval-online-reid":
        run_eval_online_reid(args)
    elif args.cmd == "eval-maneuver-hazard":
        run_eval_maneuver_hazard(args)
    elif args.cmd == "eval-periodic-ls":
        run_eval_periodic_ls(args)
    elif args.cmd == "eval-climatology-align":
        run_eval_climatology_align(args)
    elif args.cmd == "eval-cycle-clim-ensemble":
        run_eval_cycle_clim_ensemble(args)
    elif args.cmd == "eval-climatology-trend-align":
        run_eval_climatology_trend_align(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
