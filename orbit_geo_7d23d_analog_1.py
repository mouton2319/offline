#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEO analog-forecast model: use first 7 days (+ optional previous 1 day)
to predict residuals for the following 23 days.

Design:
1) Training sample (2023):
   - baseline orbit from each TLE (pred-dir/*.csv, 30 days, 1-min).
   - truth orbit from truth-csv.
   - residual := baseline - truth for AZ/EL.
   - signature := trajectory shape of [pre_days, +fit_days] around anchor
     (default: 1 day before + first 7 days after anchor), plus TLE features.
   - target := low-rank coefficients of residual over next 23 days.
2) Inference:
   - given one TLE file, compute signature from:
     pre_days (backward propagation from same TLE) + first 7 days baseline.
   - find nearest analogs in 2023 bank and predict residual coefficients.
   - apply correction only for days 8-30.

This script intentionally changes strategy from direct No-index correction:
it performs analog pattern matching conditioned on early trajectory dynamics.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from skyfield.api import EarthSatellite, load, wgs84
except Exception:
    EarthSatellite = None
    load = None
    wgs84 = None

from orbit_error_ai_no_tle_tf import (
    angle_diff_deg,
    compute_metrics,
    ensure_dir,
    load_orbit_csv,
    log,
    now_jst_str,
    parse_tle_file,
    plot_baseline_vs_corrected,
    plot_metric_summary,
    plot_truth_baseline_corrected,
    tle_feature_vector,
    trim_horizon_and_stride,
    wrap180,
    wrap360,
)


MINUTES_PER_DAY = 1440


@dataclasses.dataclass
class TrainRecord:
    stem: str
    ts: dt.datetime
    truth_key: str
    tle_path: Path
    pred_csv_path: Path


@dataclasses.dataclass
class TrainSample:
    stem: str
    truth_key: str
    feat: np.ndarray
    coef_az: np.ndarray
    coef_el: np.ndarray
    pre_res_az: np.ndarray
    pre_res_el: np.ndarray
    pre_valid: np.ndarray
    res_az_future: np.ndarray
    res_el_future: np.ndarray
    valid_future: np.ndarray
    base_az_rmse_future: float
    base_el_rmse_future: float
    base_azel_rmse_future: float


@dataclasses.dataclass
class TrainDatasetSpec:
    name: str
    tle_dir: Path
    pred_dir: Path
    truth_csv: Path


def parse_stem_datetime(stem: str) -> Optional[dt.datetime]:
    s = str(stem).strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})[-_](\d{2})-(\d{2})-(\d{2})", s)
    if m is None:
        return None
    ds = m.group(1)
    hh = int(m.group(2))
    mm = int(m.group(3))
    ss = int(m.group(4))
    try:
        d0 = dt.datetime.strptime(ds, "%Y-%m-%d")
    except Exception:
        return None
    return d0.replace(hour=hh, minute=mm, second=ss)


def round_to_nearest_minute(d: dt.datetime) -> dt.datetime:
    f = d.replace(second=0, microsecond=0)
    if d - f >= dt.timedelta(seconds=30):
        return f + dt.timedelta(minutes=1)
    return f


def parse_float_csv(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def parse_int_csv(text: str) -> List[int]:
    vals: List[int] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(float(t)))
    return vals


def deterministic_split(stem: str, val_split: float, seed: int) -> str:
    import hashlib

    h = hashlib.md5((stem + str(seed)).encode("utf-8")).hexdigest()
    r = int(h[:8], 16) / float(16**8)
    return "val" if r < val_split else "train"


def split_stems(
    stems: List[str],
    val_split: float,
    split_mode: str,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if len(stems) == 0:
        return [], []
    v = float(np.clip(val_split, 0.0, 0.9))
    if v <= 0.0:
        return list(stems), []
    if split_mode == "hash":
        tr: List[str] = []
        va: List[str] = []
        for s in stems:
            if deterministic_split(s, v, seed) == "val":
                va.append(s)
            else:
                tr.append(s)
        if len(va) == 0 and len(tr) > 1:
            va = [tr.pop(-1)]
        return tr, va
    ss = sorted(stems)
    n_val = max(1, int(round(len(ss) * v)))
    n_val = min(n_val, max(1, len(ss) - 1))
    return ss[:-n_val], ss[-n_val:]


def read_tle_pair_lines(path: Path) -> Tuple[str, str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            return lines[i], lines[i + 1]
    if len(lines) >= 2 and lines[-2].startswith("1 ") and lines[-1].startswith("2 "):
        return lines[-2], lines[-1]
    raise ValueError(f"TLE pair not found: {path}")


def jst_naive_to_utc_components(start_jst: dt.datetime, rows: int, step_minutes: int) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[float]]:
    start_utc = start_jst - dt.timedelta(hours=9)
    y: List[int] = []
    m: List[int] = []
    d: List[int] = []
    hh: List[int] = []
    mm: List[int] = []
    ss: List[float] = []
    cur = start_utc
    step = dt.timedelta(minutes=step_minutes)
    for _ in range(rows):
        y.append(cur.year)
        m.append(cur.month)
        d.append(cur.day)
        hh.append(cur.hour)
        mm.append(cur.minute)
        ss.append(float(cur.second))
        cur += step
    return y, m, d, hh, mm, ss


def propagate_tle_relative(
    tle_path: Path,
    sat_name: str,
    observer_lat: float,
    observer_lon: float,
    start_offset_minutes: int,
    rows: int,
    step_minutes: int,
) -> pd.DataFrame:
    if EarthSatellite is None or load is None or wgs84 is None:
        raise RuntimeError("skyfield is required but not installed")
    anchor_raw = parse_stem_datetime(tle_path.stem)
    if anchor_raw is None:
        raise ValueError(f"cannot parse datetime from tle filename: {tle_path.name}")
    anchor = round_to_nearest_minute(anchor_raw)
    start_jst = anchor + dt.timedelta(minutes=int(start_offset_minutes))

    line1, line2 = read_tle_pair_lines(tle_path)
    ts = load.timescale()
    sat = EarthSatellite(line1, line2, sat_name, ts)
    obs = wgs84.latlon(observer_lat, observer_lon)

    y, mo, d, hh, mm, ss = jst_naive_to_utc_components(start_jst, rows, step_minutes)
    t = ts.utc(y, mo, d, hh, mm, ss)

    geoc = sat.at(t)
    lat, lon = wgs84.latlon_of(geoc)
    alt_km = wgs84.height_of(geoc).km

    topo = (sat - obs).at(t)
    alt, az, _ = topo.altaz()

    unix0 = int((start_jst - dt.timedelta(hours=9)).timestamp())
    step_sec = int(step_minutes * 60)
    unix = unix0 + np.arange(rows, dtype=np.int64) * step_sec
    date = [(start_jst + dt.timedelta(minutes=step_minutes * i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(rows)]

    df = pd.DataFrame(
        {
            "No": np.arange(1, rows + 1, dtype=np.int64),
            "date": date,
            "UNIX": unix,
            "Lat": lat.degrees.astype(np.float64),
            "Lon": lon.degrees.astype(np.float64),
            "Alt": np.asarray(alt_km, dtype=np.float64),
            "AZ": az.degrees.astype(np.float64),
            "EL": alt.degrees.astype(np.float64),
        }
    )
    return df


def load_trimmed_csv(path: Path, days: int, step_minutes: int, sample_every: int) -> pd.DataFrame:
    df = load_orbit_csv(path, require_targets=True)
    df = trim_horizon_and_stride(
        df,
        days_horizon=int(days),
        step_minutes=int(step_minutes),
        sample_every=int(sample_every),
    )
    if "No" not in df.columns:
        df.insert(0, "No", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


def collect_train_records(
    tle_dir: Path,
    pred_dir: Path,
    truth_key: str,
    stem_prefix: Optional[str] = None,
) -> List[TrainRecord]:
    tmap = {p.stem: p for p in sorted(tle_dir.glob("*.txt")) if p.is_file()}
    cmap = {p.stem: p for p in sorted(pred_dir.glob("*.csv")) if p.is_file()}
    stems = sorted(set(tmap.keys()) & set(cmap.keys()))
    out: List[TrainRecord] = []
    for s in stems:
        ts = parse_stem_datetime(s)
        if ts is None:
            continue
        stem_id = f"{stem_prefix}::{s}" if stem_prefix else s
        out.append(
            TrainRecord(
                stem=stem_id,
                ts=ts,
                truth_key=str(truth_key),
                tle_path=tmap[s],
                pred_csv_path=cmap[s],
            )
        )
    out.sort(key=lambda r: r.ts)
    return out


def _sanitize_dataset_name(name: str) -> str:
    x = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("._-")
    return x if x else "dataset"


def parse_train_dataset_specs(args: argparse.Namespace) -> List[TrainDatasetSpec]:
    specs: List[TrainDatasetSpec] = []
    raw = getattr(args, "train_dataset", None)
    if raw is None or len(raw) == 0:
        specs.append(
            TrainDatasetSpec(
                name="default",
                tle_dir=Path(args.tle_dir),
                pred_dir=Path(args.pred_dir),
                truth_csv=Path(args.truth_csv),
            )
        )
        return specs

    for i, text in enumerate(raw, start=1):
        tx = str(text)
        delim = "|" if "|" in tx else ","
        parts = [p.strip() for p in tx.split(delim)]
        if len(parts) == 3:
            tle_dir_s, pred_dir_s, truth_csv_s = parts
            name = f"ds{i}_{Path(tle_dir_s).name}"
        elif len(parts) == 4:
            name, tle_dir_s, pred_dir_s, truth_csv_s = parts
        else:
            raise ValueError(
                "--train-dataset must be "
                "'tle_dir|pred_dir|truth_csv' / 'name|tle_dir|pred_dir|truth_csv' "
                "or comma-separated equivalent"
            )
        specs.append(
            TrainDatasetSpec(
                name=_sanitize_dataset_name(name),
                tle_dir=Path(tle_dir_s),
                pred_dir=Path(pred_dir_s),
                truth_csv=Path(truth_csv_s),
            )
        )
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate dataset names in --train-dataset: {names}")
    return specs


def fit_linear_fill(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(vals, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return np.zeros_like(x)
    if len(idx) == 1:
        return np.full_like(x, x[idx[0]])
    out = x.copy()
    out[:] = np.interp(np.arange(len(x), dtype=np.float64), idx.astype(np.float64), x[idx])
    return out


def fit_angle_fill(vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(vals, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return np.zeros_like(x)
    if len(idx) == 1:
        return wrap180(np.full_like(x, x[idx[0]]))
    rad = np.unwrap(np.deg2rad(x[idx]))
    out = np.rad2deg(np.interp(np.arange(len(x), dtype=np.float64), idx.astype(np.float64), rad))
    return wrap180(out)


def make_future_basis(future_rows: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
    no = np.arange(1, future_rows + 1, dtype=np.int64)
    t_min = (no - 1).astype(np.float64)
    t_day = t_min / float(MINUTES_PER_DAY)
    t_norm = t_day / max(1e-9, t_day[-1] if len(t_day) else 1.0)

    feats: List[np.ndarray] = []
    names: List[str] = []

    feats.append(np.ones_like(t_norm))
    names.append("bias")
    feats.append(t_norm)
    names.append("t_norm")
    feats.append(t_norm * t_norm)
    names.append("t_norm2")

    for k in (1, 2):
        ph = 2.0 * math.pi * k * t_day
        feats.append(np.sin(ph))
        feats.append(np.cos(ph))
        names.append(f"day_sin_{k}")
        names.append(f"day_cos_{k}")

    for p in (7.0, 14.0, 23.0):
        ph = 2.0 * math.pi * (t_day / p)
        feats.append(np.sin(ph))
        feats.append(np.cos(ph))
        names.append(f"p{int(p)}d_sin")
        names.append(f"p{int(p)}d_cos")

    B = np.stack(feats, axis=1).astype(np.float64)
    return B, names, t_day


def fit_basis_coeff(
    series: np.ndarray,
    valid_mask: np.ndarray,
    basis: np.ndarray,
    t_day: np.ndarray,
    is_angle: bool,
    time_weight_power: float,
) -> np.ndarray:
    y = np.asarray(series, dtype=np.float64).reshape(-1)
    m = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if len(y) != basis.shape[0]:
        raise ValueError("series length mismatch")
    if np.sum(m) < 100:
        return np.zeros((basis.shape[1],), dtype=np.float64)

    if is_angle:
        y_fill = fit_angle_fill(y, m)
        y_use = np.rad2deg(np.unwrap(np.deg2rad(y_fill)))
    else:
        y_use = fit_linear_fill(y, m)

    w = np.zeros_like(y_use, dtype=np.float64)
    w[m] = 1.0 / np.power(1.0 + t_day[m], float(max(0.0, time_weight_power)))
    sw = np.sqrt(np.maximum(w, 0.0))
    A = basis * sw.reshape(-1, 1)
    b = y_use * sw
    ata = A.T @ A + 1e-6 * np.eye(A.shape[1], dtype=np.float64)
    atb = A.T @ b
    return np.linalg.solve(ata, atb)


def reconstruct_from_coeff(coeff: np.ndarray, basis: np.ndarray, is_angle: bool) -> np.ndarray:
    y = basis @ np.asarray(coeff, dtype=np.float64).reshape(-1)
    if is_angle:
        return wrap180(y)
    return y


def build_signature_feature(
    pre_az: np.ndarray,
    pre_el: np.ndarray,
    fit_az: np.ndarray,
    fit_el: np.ndarray,
    step_minutes: int,
    downsample_minutes: int,
    tle_feat: Optional[np.ndarray],
) -> np.ndarray:
    pre_az = np.asarray(pre_az, dtype=np.float64).reshape(-1)
    pre_el = np.asarray(pre_el, dtype=np.float64).reshape(-1)
    fit_az = np.asarray(fit_az, dtype=np.float64).reshape(-1)
    fit_el = np.asarray(fit_el, dtype=np.float64).reshape(-1)

    az = np.concatenate([pre_az, fit_az], axis=0)
    el = np.concatenate([pre_el, fit_el], axis=0)
    if len(az) == 0:
        return np.zeros((1,), dtype=np.float64)

    az_u = np.rad2deg(np.unwrap(np.deg2rad(az)))
    az_rel = az_u - az_u[0]
    el_rel = el - el[0]

    stride = max(1, int(round(float(downsample_minutes) / max(1, int(step_minutes)))))
    idx = np.arange(0, len(az_rel), stride, dtype=np.int64)
    az_s = az_rel[idx]
    el_s = el_rel[idx]

    if len(az_rel) >= 2:
        az_r = np.diff(az_rel) / float(step_minutes)
        el_r = np.diff(el_rel) / float(step_minutes)
    else:
        az_r = np.zeros((1,), dtype=np.float64)
        el_r = np.zeros((1,), dtype=np.float64)
    idxr = np.arange(0, len(az_r), stride, dtype=np.int64)
    az_rs = az_r[idxr]
    el_rs = el_r[idxr]

    # Global stats of signature window
    stats = np.array(
        [
            float(np.mean(az_rel)),
            float(np.std(az_rel)),
            float(np.max(az_rel) - np.min(az_rel)),
            float(np.mean(el_rel)),
            float(np.std(el_rel)),
            float(np.max(el_rel) - np.min(el_rel)),
            float(np.mean(az_r)),
            float(np.std(az_r)),
            float(np.mean(el_r)),
            float(np.std(el_r)),
        ],
        dtype=np.float64,
    )

    if tle_feat is None:
        tf = np.zeros((19,), dtype=np.float64)
    else:
        tf = np.asarray(tle_feat, dtype=np.float64).reshape(-1)
        if len(tf) != 19:
            tf = np.zeros((19,), dtype=np.float64)

    x = np.concatenate([az_s, el_s, az_rs, el_rs, stats, tf], axis=0).astype(np.float64)
    return x


def circular_mean_weighted_deg(vals_deg: np.ndarray, w: np.ndarray) -> np.ndarray:
    # vals_deg shape: (k, T), w shape: (k,)
    rr = np.deg2rad(np.asarray(vals_deg, dtype=np.float64))
    ww = np.asarray(w, dtype=np.float64).reshape(-1, 1)
    s = np.sum(ww * np.sin(rr), axis=0)
    c = np.sum(ww * np.cos(rr), axis=0)
    return wrap180(np.rad2deg(np.arctan2(s, c)))


def circular_blend_deg(a_deg: np.ndarray, b_deg: np.ndarray, w_b: float) -> np.ndarray:
    wb = float(np.clip(w_b, 0.0, 1.0))
    wa = 1.0 - wb
    ar = np.deg2rad(np.asarray(a_deg, dtype=np.float64))
    br = np.deg2rad(np.asarray(b_deg, dtype=np.float64))
    s = wa * np.sin(ar) + wb * np.sin(br)
    c = wa * np.cos(ar) + wb * np.cos(br)
    return wrap180(np.rad2deg(np.arctan2(s, c)))


def ensure_same_feature_length(feats: List[np.ndarray]) -> List[np.ndarray]:
    if len(feats) == 0:
        return feats
    n = len(feats[0])
    out: List[np.ndarray] = []
    for f in feats:
        a = np.asarray(f, dtype=np.float64).reshape(-1)
        if len(a) == n:
            out.append(a)
            continue
        if len(a) > n:
            out.append(a[:n].copy())
        else:
            b = np.zeros((n,), dtype=np.float64)
            b[: len(a)] = a
            out.append(b)
    return out


def pairwise_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    d2 = np.maximum(0.0, a2 + b2 - 2.0 * (A @ B.T))
    return np.sqrt(d2 + 1e-12)


def calibrate_alpha(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    valid_list: List[np.ndarray],
    alpha_grid: Sequence[float],
    is_angle: bool,
    min_improved_ratio: float,
) -> Tuple[float, float, float, float]:
    best_alpha = 0.0
    best_rmse = math.inf
    best_ratio = 0.0
    base_rmse = 0.0

    for a in alpha_grid:
        a = float(a)
        rmses: List[float] = []
        bases: List[float] = []
        improved = 0
        for yt, yp, vm in zip(y_true_list, y_pred_list, valid_list):
            m = np.asarray(vm, dtype=bool)
            if np.sum(m) < 10:
                continue
            t = np.asarray(yt, dtype=np.float64)[m]
            p = np.asarray(yp, dtype=np.float64)[m]
            if is_angle:
                e = wrap180(t - a * p)
            else:
                e = t - a * p
            rb = float(np.sqrt(np.mean(t * t)))
            rc = float(np.sqrt(np.mean(e * e)))
            bases.append(rb)
            rmses.append(rc)
            if rc < rb:
                improved += 1
        if len(rmses) == 0:
            continue
        ratio = float(improved / len(rmses))
        rm = float(np.mean(rmses))
        if ratio >= float(min_improved_ratio):
            if (rm < best_rmse - 1e-12) or (abs(rm - best_rmse) <= 1e-12 and a < best_alpha):
                best_alpha = a
                best_rmse = rm
                best_ratio = ratio
                base_rmse = float(np.mean(bases))

    if not np.isfinite(best_rmse):
        # conservative fallback
        base_only: List[float] = []
        for yt, vm in zip(y_true_list, valid_list):
            m = np.asarray(vm, dtype=bool)
            if np.sum(m) < 10:
                continue
            t = np.asarray(yt, dtype=np.float64)[m]
            base_only.append(float(np.sqrt(np.mean(t * t))))
        base_rmse = float(np.mean(base_only)) if len(base_only) else 0.0
        return 0.0, base_rmse, base_rmse, 0.0

    return float(best_alpha), float(base_rmse), float(best_rmse), float(best_ratio)


def build_train_samples(
    records: List[TrainRecord],
    truth_idx_map: Dict[str, pd.DataFrame],
    days_horizon: int,
    fit_days: int,
    pre_days: int,
    step_minutes: int,
    sample_every: int,
    use_pre1d: bool,
    observer_lat: float,
    observer_lon: float,
    sat_name: str,
    downsample_minutes: int,
    time_weight_power: float,
    min_valid_future_rows: int,
) -> Tuple[List[TrainSample], Dict]:
    rows = int(days_horizon * 24 * 60 // max(1, int(step_minutes)) // max(1, int(sample_every)))
    fit_rows = int(fit_days * 24 * 60 // max(1, int(step_minutes)) // max(1, int(sample_every)))
    pre_rows = int(pre_days * 24 * 60 // max(1, int(step_minutes)) // max(1, int(sample_every)))
    if rows <= fit_rows:
        raise ValueError("days_horizon must be greater than fit_days")
    future_rows = rows - fit_rows
    basis_fut, basis_names, t_day_fut = make_future_basis(future_rows)

    samples: List[TrainSample] = []
    feats_buf: List[np.ndarray] = []

    pre_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    pred_cache: Dict[str, pd.DataFrame] = {}
    tle_feat_cache: Dict[str, np.ndarray] = {}

    for rec in records:
        truth_idx = truth_idx_map.get(str(rec.truth_key))
        if truth_idx is None:
            log(f"[WARN] skip {rec.stem}: truth key not found ({rec.truth_key})")
            continue
        try:
            if rec.stem in pred_cache:
                bdf = pred_cache[rec.stem]
            else:
                bdf = load_trimmed_csv(rec.pred_csv_path, days=days_horizon, step_minutes=step_minutes, sample_every=sample_every)
                pred_cache[rec.stem] = bdf
        except Exception as e:
            log(f"[WARN] skip {rec.stem}: baseline load failed ({e})")
            continue
        if len(bdf) < rows:
            log(f"[WARN] skip {rec.stem}: baseline rows {len(bdf)} < {rows}")
            continue
        if len(bdf) > rows:
            bdf = bdf.iloc[:rows].copy()

        unix = pd.to_numeric(bdf["UNIX"], errors="coerce").to_numpy(dtype=np.int64)
        az_base = pd.to_numeric(bdf["AZ"], errors="coerce").to_numpy(dtype=np.float64)
        el_base = pd.to_numeric(bdf["EL"], errors="coerce").to_numpy(dtype=np.float64)

        tv = truth_idx.reindex(unix)
        az_true = pd.to_numeric(tv["AZ"], errors="coerce").to_numpy(dtype=np.float64)
        el_true = pd.to_numeric(tv["EL"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(az_true) & np.isfinite(el_true)
        if np.sum(valid) < max(1000, int(0.5 * rows)):
            continue

        res_az = angle_diff_deg(az_base, az_true)
        res_el = az_base * 0.0  # shape only
        res_el[:] = el_base - el_true

        valid_fut = valid[fit_rows:rows]
        if int(np.sum(valid_fut)) < int(min_valid_future_rows):
            continue
        res_az_fut = np.asarray(res_az[fit_rows:rows], dtype=np.float64)
        res_el_fut = np.asarray(res_el[fit_rows:rows], dtype=np.float64)
        m_fut = np.asarray(valid_fut, dtype=bool)
        rmse_az_fut = float(np.sqrt(np.mean((res_az_fut[m_fut]) ** 2))) if np.any(m_fut) else 0.0
        rmse_el_fut = float(np.sqrt(np.mean((res_el_fut[m_fut]) ** 2))) if np.any(m_fut) else 0.0
        rmse_azel_fut = 0.5 * (rmse_az_fut + rmse_el_fut)

        coef_az = fit_basis_coeff(
            series=res_az_fut,
            valid_mask=valid_fut,
            basis=basis_fut,
            t_day=t_day_fut,
            is_angle=True,
            time_weight_power=time_weight_power,
        )
        coef_el = fit_basis_coeff(
            series=res_el_fut,
            valid_mask=valid_fut,
            basis=basis_fut,
            t_day=t_day_fut,
            is_angle=False,
            time_weight_power=time_weight_power,
        )

        az_fit = az_base[:fit_rows]
        el_fit = el_base[:fit_rows]

        if use_pre1d and pre_rows > 0:
            if rec.stem in pre_cache:
                az_pre, el_pre = pre_cache[rec.stem]
                pre_unix = None
            else:
                try:
                    pdf = propagate_tle_relative(
                        tle_path=rec.tle_path,
                        sat_name=sat_name,
                        observer_lat=observer_lat,
                        observer_lon=observer_lon,
                        start_offset_minutes=-pre_rows * step_minutes,
                        rows=pre_rows,
                        step_minutes=step_minutes,
                    )
                    az_pre = pd.to_numeric(pdf["AZ"], errors="coerce").to_numpy(dtype=np.float64)
                    el_pre = pd.to_numeric(pdf["EL"], errors="coerce").to_numpy(dtype=np.float64)
                    pre_unix = pd.to_numeric(pdf["UNIX"], errors="coerce").to_numpy(dtype=np.int64)
                except Exception:
                    az_pre = np.zeros((pre_rows,), dtype=np.float64)
                    el_pre = np.zeros((pre_rows,), dtype=np.float64)
                    pre_unix = None
                pre_cache[rec.stem] = (az_pre, el_pre)
            if pre_unix is None:
                # reconstruct unix by stepping backward from first prediction time
                step_sec = int(step_minutes * 60)
                u0 = int(unix[0]) - int(pre_rows) * step_sec
                pre_unix = u0 + np.arange(pre_rows, dtype=np.int64) * step_sec
        else:
            az_pre = np.zeros((0,), dtype=np.float64)
            el_pre = np.zeros((0,), dtype=np.float64)
            pre_unix = np.zeros((0,), dtype=np.int64)

        if len(az_pre) > 0:
            tvp = truth_idx.reindex(pre_unix)
            az_true_pre = pd.to_numeric(tvp["AZ"], errors="coerce").to_numpy(dtype=np.float64)
            el_true_pre = pd.to_numeric(tvp["EL"], errors="coerce").to_numpy(dtype=np.float64)
            pre_valid = np.isfinite(az_true_pre) & np.isfinite(el_true_pre)
            pre_res_az = np.zeros_like(az_pre, dtype=np.float64)
            pre_res_el = np.zeros_like(el_pre, dtype=np.float64)
            pre_res_az[pre_valid] = angle_diff_deg(az_pre[pre_valid], az_true_pre[pre_valid])
            pre_res_el[pre_valid] = el_pre[pre_valid] - el_true_pre[pre_valid]
        else:
            pre_valid = np.zeros((0,), dtype=bool)
            pre_res_az = np.zeros((0,), dtype=np.float64)
            pre_res_el = np.zeros((0,), dtype=np.float64)

        if rec.stem not in tle_feat_cache:
            try:
                t = parse_tle_file(rec.tle_path)
                tf, _ = tle_feature_vector(t)
                tle_feat_cache[rec.stem] = np.asarray(tf, dtype=np.float64)
            except Exception:
                tle_feat_cache[rec.stem] = np.zeros((19,), dtype=np.float64)
        tf = tle_feat_cache[rec.stem]

        feat = build_signature_feature(
            pre_az=az_pre,
            pre_el=el_pre,
            fit_az=az_fit,
            fit_el=el_fit,
            step_minutes=step_minutes,
            downsample_minutes=downsample_minutes,
            tle_feat=tf,
        )
        feats_buf.append(feat)
        samples.append(
            TrainSample(
                stem=rec.stem,
                truth_key=str(rec.truth_key),
                feat=feat,
                coef_az=np.asarray(coef_az, dtype=np.float64),
                coef_el=np.asarray(coef_el, dtype=np.float64),
                pre_res_az=np.asarray(pre_res_az, dtype=np.float64),
                pre_res_el=np.asarray(pre_res_el, dtype=np.float64),
                pre_valid=np.asarray(pre_valid, dtype=bool),
                res_az_future=np.asarray(res_az_fut, dtype=np.float64),
                res_el_future=np.asarray(res_el_fut, dtype=np.float64),
                valid_future=np.asarray(valid_fut, dtype=bool),
                base_az_rmse_future=float(rmse_az_fut),
                base_el_rmse_future=float(rmse_el_fut),
                base_azel_rmse_future=float(rmse_azel_fut),
            )
        )

    if len(samples) == 0:
        return [], {}

    feats_fixed = ensure_same_feature_length(feats_buf)
    for s, f in zip(samples, feats_fixed):
        s.feat = f

    info = {
        "rows": rows,
        "fit_rows": fit_rows,
        "future_rows": future_rows,
        "basis_future": basis_fut,
        "basis_future_names": basis_names,
    }
    return samples, info


def knn_weighted_predict_coeff(
    x_query_z: np.ndarray,
    Xbank_z: np.ndarray,
    coef_az_bank: np.ndarray,
    coef_el_bank: np.ndarray,
    k: int,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.sqrt(np.maximum(0.0, np.sum((Xbank_z - x_query_z.reshape(1, -1)) ** 2, axis=1)))
    idx = np.argsort(d)[: max(1, int(k))]
    dd = d[idx]
    tt = max(1e-6, float(tau))
    w = np.exp(-dd / tt)
    sw = float(np.sum(w))
    if sw <= 1e-12:
        w = np.ones_like(w) / max(1, len(w))
    else:
        w = w / sw
    paz = np.sum(coef_az_bank[idx] * w.reshape(-1, 1), axis=0)
    pel = np.sum(coef_el_bank[idx] * w.reshape(-1, 1), axis=0)
    return paz, pel, idx, w


def train_cmd(args: argparse.Namespace) -> int:
    out_model = Path(args.out_model)

    dataset_specs = parse_train_dataset_specs(args)
    if len(dataset_specs) == 0:
        raise RuntimeError("no train datasets resolved")
    multi_ds = len(dataset_specs) > 1

    truth_idx_map: Dict[str, pd.DataFrame] = {}
    records: List[TrainRecord] = []
    ds_info: List[Dict] = []

    for dsi, ds in enumerate(dataset_specs, start=1):
        if not ds.tle_dir.exists():
            raise FileNotFoundError(f"tle-dir not found: {ds.tle_dir} (dataset={ds.name})")
        if not ds.pred_dir.exists():
            raise FileNotFoundError(f"pred-dir not found: {ds.pred_dir} (dataset={ds.name})")
        if not ds.truth_csv.exists():
            raise FileNotFoundError(f"truth-csv not found: {ds.truth_csv} (dataset={ds.name})")

        truth_df = load_orbit_csv(ds.truth_csv, require_targets=True)
        truth_idx = truth_df.drop_duplicates(subset=["UNIX"]).set_index("UNIX")[["AZ", "EL"]]
        truth_idx_map[str(ds.name)] = truth_idx

        stem_prefix = str(ds.name) if multi_ds else None
        rec_ds = collect_train_records(
            tle_dir=ds.tle_dir,
            pred_dir=ds.pred_dir,
            truth_key=str(ds.name),
            stem_prefix=stem_prefix,
        )
        records.extend(rec_ds)
        ds_info.append(
            {
                "name": str(ds.name),
                "tle_dir": str(ds.tle_dir),
                "pred_dir": str(ds.pred_dir),
                "truth_csv": str(ds.truth_csv),
                "n_records": int(len(rec_ds)),
            }
        )
        log(f"[INFO] dataset[{dsi}] {ds.name}: records={len(rec_ds)}")

    records.sort(key=lambda r: r.ts)
    if len(records) == 0:
        raise RuntimeError("no matching tle/csv records")
    if args.max_files is not None:
        records = records[: int(args.max_files)]
    log(f"[INFO] train records: {len(records)} (datasets={len(dataset_specs)})")

    samples, info = build_train_samples(
        records=records,
        truth_idx_map=truth_idx_map,
        days_horizon=int(args.days_horizon),
        fit_days=int(args.fit_days),
        pre_days=int(args.pre_days),
        step_minutes=int(args.step_minutes),
        sample_every=int(args.sample_every),
        use_pre1d=bool(args.use_pre1d),
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        sat_name=str(args.sat_name),
        downsample_minutes=int(args.signature_downsample_minutes),
        time_weight_power=float(args.time_weight_power),
        min_valid_future_rows=int(args.min_valid_future_rows),
    )
    if len(samples) < 20:
        raise RuntimeError(f"not enough usable samples: {len(samples)}")
    log(f"[INFO] usable samples: {len(samples)}")
    if len(dataset_specs) > 1:
        cnt: Dict[str, int] = {}
        for s in samples:
            k = str(s.truth_key)
            cnt[k] = cnt.get(k, 0) + 1
        for nm in sorted(cnt.keys()):
            log(f"[INFO] usable samples dataset={nm}: {cnt[nm]}")

    stems = [s.stem for s in samples]
    tr_stems, va_stems = split_stems(
        stems=stems,
        val_split=float(args.val_split),
        split_mode=str(args.split_mode),
        seed=int(args.seed),
    )
    tr_set = set(tr_stems)
    va_set = set(va_stems)
    tr = [s for s in samples if s.stem in tr_set]
    va = [s for s in samples if s.stem in va_set]
    if len(va) == 0:
        raise RuntimeError("validation split is empty")
    log(f"[INFO] split train/val = {len(tr)}/{len(va)}")

    Xtr = np.stack([s.feat for s in tr], axis=0).astype(np.float64)
    Xva = np.stack([s.feat for s in va], axis=0).astype(np.float64)
    Ctr_az = np.stack([s.coef_az for s in tr], axis=0).astype(np.float64)
    Ctr_el = np.stack([s.coef_el for s in tr], axis=0).astype(np.float64)

    x_mean = np.mean(Xtr, axis=0)
    x_std = np.std(Xtr, axis=0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)
    Xtr_z = (Xtr - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    Xva_z = (Xva - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

    basis_f = np.asarray(info["basis_future"], dtype=np.float64)
    future_rows = int(info["future_rows"])

    D = pairwise_l2(Xva_z, Xtr_z)
    ord_idx = np.argsort(D, axis=1)
    ord_d = np.take_along_axis(D, ord_idx, axis=1)

    alpha_grid = parse_float_csv(args.alpha_grid)
    if len(alpha_grid) == 0:
        alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    k_grid = parse_int_csv(args.k_grid)
    if len(k_grid) == 0:
        k_grid = [5, 10, 20]
    tau_grid = parse_float_csv(args.tau_grid)
    if len(tau_grid) == 0:
        tau_grid = [0.5, 1.0, 2.0]

    Yva_az = [s.res_az_future for s in va]
    Yva_el = [s.res_el_future for s in va]
    Mva = [s.valid_future for s in va]

    best = None
    best_obj = math.inf

    for k in k_grid:
        kk = max(1, min(int(k), Xtr_z.shape[0]))
        idx_k = ord_idx[:, :kk]
        d_k = ord_d[:, :kk]

        for tau in tau_grid:
            tt = max(1e-6, float(tau))
            w = np.exp(-d_k / tt)
            sw = np.sum(w, axis=1, keepdims=True)
            sw = np.where(sw <= 1e-12, 1.0, sw)
            w = w / sw

            pred_coef_az = np.sum(Ctr_az[idx_k] * w[:, :, None], axis=1)
            pred_coef_el = np.sum(Ctr_el[idx_k] * w[:, :, None], axis=1)

            P_az = wrap180(pred_coef_az @ basis_f.T)
            P_el = pred_coef_el @ basis_f.T

            a_az, b_az, c_az, r_az = calibrate_alpha(
                y_true_list=Yva_az,
                y_pred_list=[P_az[i] for i in range(len(va))],
                valid_list=Mva,
                alpha_grid=alpha_grid,
                is_angle=True,
                min_improved_ratio=float(args.min_improved_file_ratio),
            )
            a_el, b_el, c_el, r_el = calibrate_alpha(
                y_true_list=Yva_el,
                y_pred_list=[P_el[i] for i in range(len(va))],
                valid_list=Mva,
                alpha_grid=alpha_grid,
                is_angle=False,
                min_improved_ratio=float(args.min_improved_file_ratio),
            )

            obj = 0.5 * (c_az + c_el)
            log(
                f"[SEARCH] k={kk} tau={tt:g} | "
                f"AZ {b_az:.6g}->{c_az:.6g} (alpha={a_az:.3f}, ratio={r_az:.3f}) | "
                f"EL {b_el:.6g}->{c_el:.6g} (alpha={a_el:.3f}, ratio={r_el:.3f}) | "
                f"AZEL={obj:.6g}"
            )
            if obj < best_obj:
                best_obj = obj
                best = {
                    "k": int(kk),
                    "tau": float(tt),
                    "alpha_az": float(a_az),
                    "alpha_el": float(a_el),
                    "val_rmse_az_base": float(b_az),
                    "val_rmse_az_corr": float(c_az),
                    "val_rmse_el_base": float(b_el),
                    "val_rmse_el_corr": float(c_el),
                }

    if best is None:
        raise RuntimeError("grid search failed")

    alpha_infer_shrink = float(np.clip(args.alpha_infer_shrink, 0.0, 1.0))
    alpha_az_infer = float(best["alpha_az"] * alpha_infer_shrink)
    alpha_el_infer = float(best["alpha_el"] * alpha_infer_shrink)
    hybrid_weight = float(np.clip(args.hybrid_weight, 0.0, 1.0))
    pre_rows_meta = int(
        int(args.pre_days) * 24 * 60 // max(1, int(args.step_minutes)) // max(1, int(args.sample_every))
    )

    # Build full bank from all usable samples (for inference analog retrieval)
    Xall = np.stack([s.feat for s in samples], axis=0).astype(np.float64)
    Xall_z = (Xall - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    Call_az = np.stack([s.coef_az for s in samples], axis=0).astype(np.float64)
    Call_el = np.stack([s.coef_el for s in samples], axis=0).astype(np.float64)
    Pall_az = np.stack([s.pre_res_az for s in samples], axis=0).astype(np.float32)
    Pall_el = np.stack([s.pre_res_el for s in samples], axis=0).astype(np.float32)
    Pall_valid = np.stack([s.pre_valid.astype(np.uint8) for s in samples], axis=0).astype(np.uint8)
    Rall_az = np.stack([s.res_az_future for s in samples], axis=0).astype(np.float32)
    Rall_el = np.stack([s.res_el_future for s in samples], axis=0).astype(np.float32)
    base_az_rmse_bank = np.asarray([s.base_az_rmse_future for s in samples], dtype=np.float64)
    base_el_rmse_bank = np.asarray([s.base_el_rmse_future for s in samples], dtype=np.float64)
    base_azel_rmse_bank = np.asarray([s.base_azel_rmse_future for s in samples], dtype=np.float64)
    sample_count_by_truth_key: Dict[str, int] = {}
    for s in samples:
        k = str(s.truth_key)
        sample_count_by_truth_key[k] = sample_count_by_truth_key.get(k, 0) + 1

    # Distance reference for OOD confidence (leave-one-out NN distance among bank)
    Db = pairwise_l2(Xall_z, Xall_z)
    np.fill_diagonal(Db, np.inf)
    dnn = np.min(Db, axis=1)
    dist_ref = float(np.percentile(dnn[np.isfinite(dnn)], 70)) if np.any(np.isfinite(dnn)) else 1.0
    if not np.isfinite(dist_ref) or dist_ref <= 1e-9:
        dist_ref = 1.0

    meta = {
        "created_jst": now_jst_str(),
        "model_type": "geo_7d23d_analog_v1",
        "days_horizon": int(args.days_horizon),
        "fit_days": int(args.fit_days),
        "pre_days": int(args.pre_days),
        "step_minutes": int(args.step_minutes),
        "sample_every": int(args.sample_every),
        "rows": int(info["rows"]),
        "fit_rows": int(info["fit_rows"]),
        "future_rows": int(info["future_rows"]),
        "use_pre1d": bool(args.use_pre1d),
        "signature_downsample_minutes": int(args.signature_downsample_minutes),
        "selected_k": int(best["k"]),
        "selected_tau": float(best["tau"]),
        "alpha_az": float(best["alpha_az"]),
        "alpha_el": float(best["alpha_el"]),
        "alpha_infer_shrink": float(alpha_infer_shrink),
        "alpha_az_infer": float(alpha_az_infer),
        "alpha_el_infer": float(alpha_el_infer),
        "hybrid_weight": float(hybrid_weight),
        "cap_deg": float(args.cap_deg),
        "ood_conf_floor": float(args.ood_conf_floor),
        "min_ood_conf_apply": float(args.min_ood_conf_apply),
        "transition_hours": float(args.transition_hours),
        "auto_lowerr_th": (None if args.auto_lowerr_th is None else float(max(0.0, args.auto_lowerr_th))),
        "auto_max_neighbor_dist": (None if args.auto_max_neighbor_dist is None else float(max(0.0, args.auto_max_neighbor_dist))),
        "auto_max_unc_az": (None if args.auto_max_unc_az is None else float(max(0.0, args.auto_max_unc_az))),
        "auto_max_unc_el": (None if args.auto_max_unc_el is None else float(max(0.0, args.auto_max_unc_el))),
        "auto_max_corr_ratio": (None if args.auto_max_corr_ratio is None else float(max(0.0, args.auto_max_corr_ratio))),
        "dist_ref": float(dist_ref),
        "val_rmse_az_base": float(best["val_rmse_az_base"]),
        "val_rmse_az_corr": float(best["val_rmse_az_corr"]),
        "val_rmse_el_base": float(best["val_rmse_el_base"]),
        "val_rmse_el_corr": float(best["val_rmse_el_corr"]),
        "n_samples": int(len(samples)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_datasets": int(len(dataset_specs)),
        "train_datasets": ds_info,
        "sample_count_by_dataset": sample_count_by_truth_key,
        "pre_rows": int(pre_rows_meta),
        "has_pre_residual_bank": bool(Pall_az.size > 0),
        "has_direct_future_residual_bank": True,
        "basis_future_names": info["basis_future_names"],
    }

    ensure_dir(out_model)
    meta_path = out_model / "meta.json"
    bank_path = out_model / "bank.npz"
    summary_path = out_model / "train_summary.json"
    if (meta_path.exists() or bank_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(f"model exists: {out_model} (use --overwrite)")

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        bank_path,
        x_mean=x_mean.astype(np.float64),
        x_std=x_std.astype(np.float64),
        Xbank_z=Xall_z.astype(np.float64),
        coef_az_bank=Call_az.astype(np.float64),
        coef_el_bank=Call_el.astype(np.float64),
        pre_res_az_bank=Pall_az.astype(np.float32),
        pre_res_el_bank=Pall_el.astype(np.float32),
        pre_valid_bank=Pall_valid.astype(np.uint8),
        res_az_future_bank=Rall_az.astype(np.float32),
        res_el_future_bank=Rall_el.astype(np.float32),
        base_az_rmse_bank=base_az_rmse_bank.astype(np.float64),
        base_el_rmse_bank=base_el_rmse_bank.astype(np.float64),
        base_azel_rmse_bank=base_azel_rmse_bank.astype(np.float64),
        basis_future=np.asarray(basis_f, dtype=np.float64),
        stems=np.asarray([s.stem for s in samples]),
    )
    summary = {
        "selected_k": int(best["k"]),
        "selected_tau": float(best["tau"]),
        "alpha_az": float(best["alpha_az"]),
        "alpha_el": float(best["alpha_el"]),
        "alpha_az_infer": float(alpha_az_infer),
        "alpha_el_infer": float(alpha_el_infer),
        "val_rmse_az_base": float(best["val_rmse_az_base"]),
        "val_rmse_az_corr": float(best["val_rmse_az_corr"]),
        "val_rmse_el_base": float(best["val_rmse_el_base"]),
        "val_rmse_el_corr": float(best["val_rmse_el_corr"]),
        "val_rmse_azel_base_mean": float(0.5 * (best["val_rmse_az_base"] + best["val_rmse_el_base"])),
        "val_rmse_azel_corr_mean": float(0.5 * (best["val_rmse_az_corr"] + best["val_rmse_el_corr"])),
        "dist_ref": float(dist_ref),
        "n_datasets": int(len(dataset_specs)),
        "sample_count_by_dataset": sample_count_by_truth_key,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"[SAVED] model: {out_model}")
    log(
        f"[VAL] AZ {best['val_rmse_az_base']:.6g}->{best['val_rmse_az_corr']:.6g} | "
        f"EL {best['val_rmse_el_base']:.6g}->{best['val_rmse_el_corr']:.6g}"
    )
    return 0


def load_model(model_dir: Path) -> Tuple[Dict, Dict[str, np.ndarray]]:
    meta_path = model_dir / "meta.json"
    bank_path = model_dir / "bank.npz"
    if not meta_path.exists() or not bank_path.exists():
        search_root = model_dir.parent if model_dir.parent.exists() else Path(".")
        cands: List[str] = []
        try:
            for p in sorted(search_root.iterdir()):
                if not p.is_dir():
                    continue
                if (p / "meta.json").exists() and (p / "bank.npz").exists():
                    cands.append(p.name)
        except Exception:
            cands = []
        hint = ""
        if len(cands) > 0:
            hint = f" | available model dirs: {', '.join(cands[:12])}"
        raise FileNotFoundError(
            f"model files not found under: {model_dir} "
            f"(required: meta.json and bank.npz){hint}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(bank_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def compute_segment_metrics(
    truth_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    start_no: int,
) -> pd.DataFrame:
    b = baseline_df[pd.to_numeric(baseline_df["No"], errors="coerce") >= int(start_no)].copy()
    c = corrected_df[pd.to_numeric(corrected_df["No"], errors="coerce") >= int(start_no)].copy()
    mdf, _ = compute_metrics(truth_df=truth_df, baseline_df=b, corrected_df=c)
    return mdf


def predict_cmd(args: argparse.Namespace) -> int:
    model_dir = Path(args.model)
    meta, arrays = load_model(model_dir)

    tle_file = Path(args.tle_file)
    if not tle_file.exists():
        raise FileNotFoundError(f"tle-file not found: {tle_file}")

    days_h = int(args.days_horizon if args.days_horizon is not None else meta["days_horizon"])
    fit_days = int(args.fit_days if args.fit_days is not None else meta["fit_days"])
    pre_days = int(args.pre_days if args.pre_days is not None else meta["pre_days"])
    step_minutes = int(args.step_minutes if args.step_minutes is not None else meta["step_minutes"])
    sample_every = int(args.sample_every if args.sample_every is not None else meta.get("sample_every", 1))
    rows = int(days_h * 24 * 60 // max(1, step_minutes) // max(1, sample_every))
    fit_rows = int(fit_days * 24 * 60 // max(1, step_minutes) // max(1, sample_every))
    pre_rows = int(pre_days * 24 * 60 // max(1, step_minutes) // max(1, sample_every))
    if rows <= fit_rows:
        raise ValueError("days_horizon must be greater than fit_days")
    future_rows = rows - fit_rows

    # Baseline 30-day trajectory from csv (if provided) else propagate from TLE.
    if args.baseline_csv:
        base_df = load_trimmed_csv(Path(args.baseline_csv), days=days_h, step_minutes=step_minutes, sample_every=sample_every)
        if len(base_df) < rows:
            raise RuntimeError(f"baseline rows {len(base_df)} < required {rows}")
        if len(base_df) > rows:
            base_df = base_df.iloc[:rows].copy()
    else:
        base_df = propagate_tle_relative(
            tle_path=tle_file,
            sat_name=str(args.sat_name),
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            start_offset_minutes=0,
            rows=rows,
            step_minutes=step_minutes,
        )
        if sample_every > 1:
            base_df = base_df.iloc[::sample_every].reset_index(drop=True)
            base_df["No"] = np.arange(1, len(base_df) + 1, dtype=np.int64)
            rows = len(base_df)
            fit_rows = int(fit_days * 24 * 60 // max(1, step_minutes) // max(1, sample_every))
            future_rows = rows - fit_rows
            if future_rows <= 0:
                raise RuntimeError("future_rows <= 0 after sample_every")

    az_base = pd.to_numeric(base_df["AZ"], errors="coerce").to_numpy(dtype=np.float64)
    el_base = pd.to_numeric(base_df["EL"], errors="coerce").to_numpy(dtype=np.float64)

    # Previous 1-day trajectory from same TLE (allowed by your requirement).
    use_pre1d = bool(args.use_pre1d if args.use_pre1d is not None else meta.get("use_pre1d", True))
    if use_pre1d and pre_rows > 0:
        pre_df = propagate_tle_relative(
            tle_path=tle_file,
            sat_name=str(args.sat_name),
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            start_offset_minutes=-pre_rows * step_minutes,
            rows=pre_rows,
            step_minutes=step_minutes,
        )
        az_pre = pd.to_numeric(pre_df["AZ"], errors="coerce").to_numpy(dtype=np.float64)
        el_pre = pd.to_numeric(pre_df["EL"], errors="coerce").to_numpy(dtype=np.float64)
        pre_unix = pd.to_numeric(pre_df["UNIX"], errors="coerce").to_numpy(dtype=np.int64)
    else:
        az_pre = np.zeros((0,), dtype=np.float64)
        el_pre = np.zeros((0,), dtype=np.float64)
        pre_unix = np.zeros((0,), dtype=np.int64)

    fit_az = az_base[:fit_rows]
    fit_el = el_base[:fit_rows]
    if len(fit_az) < fit_rows:
        raise RuntimeError("baseline fit segment is shorter than expected")

    try:
        t = parse_tle_file(tle_file)
        tle_feat, _ = tle_feature_vector(t)
    except Exception:
        tle_feat = np.zeros((19,), dtype=np.float64)

    x = build_signature_feature(
        pre_az=az_pre,
        pre_el=el_pre,
        fit_az=fit_az,
        fit_el=fit_el,
        step_minutes=step_minutes,
        downsample_minutes=int(meta.get("signature_downsample_minutes", 30)),
        tle_feat=tle_feat,
    )

    x_mean = arrays["x_mean"].astype(np.float64).reshape(-1)
    x_std = arrays["x_std"].astype(np.float64).reshape(-1)
    if len(x) != len(x_mean):
        if len(x) > len(x_mean):
            x = x[: len(x_mean)]
        else:
            y = np.zeros_like(x_mean)
            y[: len(x)] = x
            x = y
    xz = (x - x_mean) / x_std

    Xbank_z = arrays["Xbank_z"].astype(np.float64)
    Caz_bank = arrays["coef_az_bank"].astype(np.float64)
    Cel_bank = arrays["coef_el_bank"].astype(np.float64)
    Paz_bank = arrays["pre_res_az_bank"].astype(np.float64) if "pre_res_az_bank" in arrays else None
    Pel_bank = arrays["pre_res_el_bank"].astype(np.float64) if "pre_res_el_bank" in arrays else None
    Pvalid_bank = arrays["pre_valid_bank"].astype(np.uint8) if "pre_valid_bank" in arrays else None
    Raz_bank = arrays["res_az_future_bank"].astype(np.float64) if "res_az_future_bank" in arrays else None
    Rel_bank = arrays["res_el_future_bank"].astype(np.float64) if "res_el_future_bank" in arrays else None
    base_az_bank = arrays["base_az_rmse_bank"].astype(np.float64) if "base_az_rmse_bank" in arrays else None
    base_el_bank = arrays["base_el_rmse_bank"].astype(np.float64) if "base_el_rmse_bank" in arrays else None
    base_azel_bank = arrays["base_azel_rmse_bank"].astype(np.float64) if "base_azel_rmse_bank" in arrays else None
    basis_f = arrays["basis_future"].astype(np.float64)
    stems = arrays["stems"]
    stem_list = [str(s) for s in stems.tolist()]

    k = int(args.k if args.k is not None else meta["selected_k"])
    tau = float(args.tau if args.tau is not None else meta["selected_tau"])
    alpha_az = float(args.alpha_az if args.alpha_az is not None else meta.get("alpha_az_infer", meta["alpha_az"]))
    alpha_el = float(args.alpha_el if args.alpha_el is not None else meta.get("alpha_el_infer", meta["alpha_el"]))
    predictor_mode = str(args.predictor_mode if args.predictor_mode is not None else "basis").strip().lower()
    if predictor_mode not in ("basis", "direct", "hybrid"):
        predictor_mode = "basis"
    hybrid_weight = float(
        np.clip(
            args.hybrid_weight if args.hybrid_weight is not None else meta.get("hybrid_weight", 0.5),
            0.0,
            1.0,
        )
    )
    cap_deg = float(args.cap_deg if args.cap_deg is not None else meta.get("cap_deg", 0.02))
    ood_floor = float(args.ood_conf_floor if args.ood_conf_floor is not None else meta.get("ood_conf_floor", 0.0))
    min_ood_apply = float(args.min_ood_conf_apply if args.min_ood_conf_apply is not None else meta.get("min_ood_conf_apply", 0.15))
    transition_h = float(args.transition_hours if args.transition_hours is not None else meta.get("transition_hours", 6.0))
    auto_lowerr_th = (
        None
        if args.auto_lowerr_th is None
        else float(max(0.0, args.auto_lowerr_th))
    )
    if auto_lowerr_th is None and meta.get("auto_lowerr_th") is not None:
        auto_lowerr_th = float(max(0.0, meta.get("auto_lowerr_th")))
    auto_max_neighbor_dist = (
        None
        if args.auto_max_neighbor_dist is None
        else float(max(0.0, args.auto_max_neighbor_dist))
    )
    if auto_max_neighbor_dist is None and meta.get("auto_max_neighbor_dist") is not None:
        auto_max_neighbor_dist = float(max(0.0, meta.get("auto_max_neighbor_dist")))
    auto_max_unc_az = (
        None
        if args.auto_max_unc_az is None
        else float(max(0.0, args.auto_max_unc_az))
    )
    if auto_max_unc_az is None and meta.get("auto_max_unc_az") is not None:
        auto_max_unc_az = float(max(0.0, meta.get("auto_max_unc_az")))
    auto_max_unc_el = (
        None
        if args.auto_max_unc_el is None
        else float(max(0.0, args.auto_max_unc_el))
    )
    if auto_max_unc_el is None and meta.get("auto_max_unc_el") is not None:
        auto_max_unc_el = float(max(0.0, meta.get("auto_max_unc_el")))
    auto_max_corr_ratio = (
        None
        if args.auto_max_corr_ratio is None
        else float(max(0.0, args.auto_max_corr_ratio))
    )
    if auto_max_corr_ratio is None and meta.get("auto_max_corr_ratio") is not None:
        auto_max_corr_ratio = float(max(0.0, meta.get("auto_max_corr_ratio")))
    obs_truth_csv_path = Path(args.obs_truth_csv) if args.obs_truth_csv else None
    obs_hours = float(max(0.0, args.obs_hours))
    obs_min_valid = int(max(5, args.obs_min_valid))
    obs_assim_strength = float(max(0.0, args.obs_assim_strength))
    obs_pool_size = int(max(k, args.obs_pool_size))
    obs_sign_flip = bool(args.obs_sign_flip)
    obs_sign_margin = float(max(0.0, args.obs_sign_flip_margin))
    obs_conf_floor = float(max(0.0, args.obs_conf_floor))

    # Base kNN from signature distance.
    d = np.sqrt(np.maximum(0.0, np.sum((Xbank_z - xz.reshape(1, -1)) ** 2, axis=1)))
    ord_idx = np.argsort(d)
    kk = max(1, min(int(k), len(ord_idx)))
    idx = ord_idx[:kk]
    dd = d[idx]
    tt = max(1e-6, float(tau))
    w = np.exp(-dd / tt)
    sw = float(np.sum(w))
    if sw <= 1e-12:
        w = np.ones_like(w) / max(1, len(w))
    else:
        w = w / sw

    # Optional online assimilation using recent measured AZ/EL (before TLE epoch).
    obs_used = False
    obs_valid_rows = 0
    obs_weight_gain = 1.0
    obs_sign_az = 1.0
    obs_sign_el = 1.0
    if (
        obs_truth_csv_path is not None
        and obs_hours > 0.0
        and Paz_bank is not None
        and Pel_bank is not None
        and Pvalid_bank is not None
        and Paz_bank.ndim == 2
        and Pel_bank.ndim == 2
        and Pvalid_bank.ndim == 2
        and Paz_bank.shape[0] == Xbank_z.shape[0]
        and Pel_bank.shape[0] == Xbank_z.shape[0]
        and Pvalid_bank.shape[0] == Xbank_z.shape[0]
    ):
        obs_rows = int(round(obs_hours * 60.0 / max(1, step_minutes)))
        obs_rows = max(1, obs_rows)
        pre_bank_rows = int(Paz_bank.shape[1])
        use_rows = min(obs_rows, pre_bank_rows)
        if use_rows >= obs_min_valid and len(pre_unix) >= use_rows:
            obs_unix = pre_unix[-use_rows:]
            obs_base_az = az_pre[-use_rows:]
            obs_base_el = el_pre[-use_rows:]
            try:
                obs_truth_df = load_orbit_csv(obs_truth_csv_path, require_targets=True)
                obs_truth_idx = obs_truth_df.drop_duplicates(subset=["UNIX"]).set_index("UNIX")[["AZ", "EL"]]
                tvo = obs_truth_idx.reindex(obs_unix)
                obs_true_az = pd.to_numeric(tvo["AZ"], errors="coerce").to_numpy(dtype=np.float64)
                obs_true_el = pd.to_numeric(tvo["EL"], errors="coerce").to_numpy(dtype=np.float64)
                obs_valid = np.isfinite(obs_true_az) & np.isfinite(obs_true_el)
                obs_res_az = np.zeros((use_rows,), dtype=np.float64)
                obs_res_el = np.zeros((use_rows,), dtype=np.float64)
                obs_res_az[obs_valid] = angle_diff_deg(obs_base_az[obs_valid], obs_true_az[obs_valid])
                obs_res_el[obs_valid] = obs_base_el[obs_valid] - obs_true_el[obs_valid]
                obs_valid_rows = int(np.sum(obs_valid))
                if obs_valid_rows >= obs_min_valid:
                    pool = ord_idx[: min(len(ord_idx), int(obs_pool_size))]
                    e = np.full((len(pool),), 1e6, dtype=np.float64)
                    s_az = float(np.std(obs_res_az[obs_valid])) if np.any(obs_valid) else 1.0
                    s_el = float(np.std(obs_res_el[obs_valid])) if np.any(obs_valid) else 1.0
                    s_az = max(1e-4, s_az)
                    s_el = max(1e-4, s_el)
                    for ii, j in enumerate(pool.tolist()):
                        pj_az = Paz_bank[j, -use_rows:].astype(np.float64)
                        pj_el = Pel_bank[j, -use_rows:].astype(np.float64)
                        pj_v = Pvalid_bank[j, -use_rows:] > 0
                        m = obs_valid & pj_v
                        nv = int(np.sum(m))
                        if nv < obs_min_valid:
                            continue
                        er_az = float(np.sqrt(np.mean(angle_diff_deg(obs_res_az[m], pj_az[m]) ** 2)))
                        er_el = float(np.sqrt(np.mean((obs_res_el[m] - pj_el[m]) ** 2)))
                        e[ii] = 0.5 * (er_az / s_az + er_el / s_el)
                    if np.any(np.isfinite(e)):
                        d_pool = d[pool]
                        e_use = np.where(np.isfinite(e), e, 1e6)
                        logw = -d_pool / tt - float(obs_assim_strength) * e_use
                        logw = logw - float(np.max(logw))
                        w_comb = np.exp(logw)
                        if np.sum(w_comb) > 0.0 and np.all(np.isfinite(w_comb)):
                            top = np.argsort(w_comb)[::-1][:kk]
                            idx = pool[top]
                            w = w_comb[top]
                            w = w / np.sum(w)
                            obs_used = True
                            obs_weight_gain = float(np.exp(-float(np.min(e_use)) * float(obs_assim_strength)))

                            # Online sign correction from pre-window fit quality.
                            if obs_sign_flip:
                                p_pre_az = circular_mean_weighted_deg(Paz_bank[idx, -use_rows:], w)
                                p_pre_el = np.sum(w.reshape(-1, 1) * Pel_bank[idx, -use_rows:], axis=0)
                                m = obs_valid
                                if int(np.sum(m)) >= obs_min_valid:
                                    pos_az = float(np.sqrt(np.mean(angle_diff_deg(obs_res_az[m], p_pre_az[m]) ** 2)))
                                    neg_az = float(np.sqrt(np.mean(angle_diff_deg(obs_res_az[m], -p_pre_az[m]) ** 2)))
                                    pos_el = float(np.sqrt(np.mean((obs_res_el[m] - p_pre_el[m]) ** 2)))
                                    neg_el = float(np.sqrt(np.mean((obs_res_el[m] + p_pre_el[m]) ** 2)))
                                    if neg_az + obs_sign_margin < pos_az:
                                        obs_sign_az = -1.0
                                    if neg_el + obs_sign_margin < pos_el:
                                        obs_sign_el = -1.0
            except Exception as e_obs:
                log(f"[WARN] obs assimilation skipped: {e_obs}")

    pcoef_az = np.sum(Caz_bank[idx] * w.reshape(-1, 1), axis=0)
    pcoef_el = np.sum(Cel_bank[idx] * w.reshape(-1, 1), axis=0)
    pred_basis_az_future = reconstruct_from_coeff(pcoef_az, basis_f, is_angle=True)
    pred_basis_el_future = reconstruct_from_coeff(pcoef_el, basis_f, is_angle=False)
    if len(pred_basis_az_future) != future_rows:
        pred_basis_az_future = pred_basis_az_future[:future_rows]
        pred_basis_el_future = pred_basis_el_future[:future_rows]

    direct_available = (
        Raz_bank is not None
        and Rel_bank is not None
        and Raz_bank.ndim == 2
        and Rel_bank.ndim == 2
        and Raz_bank.shape[0] == Xbank_z.shape[0]
        and Rel_bank.shape[0] == Xbank_z.shape[0]
    )
    direct_shape_ok = (
        direct_available
        and Raz_bank.shape[1] == future_rows
        and Rel_bank.shape[1] == future_rows
    )
    has_direct = bool(direct_shape_ok)
    if predictor_mode in ("direct", "hybrid") and not has_direct:
        if direct_available:
            log(
                f"[WARN] predictor-mode direct/hybrid requested, but direct bank length mismatch "
                f"(bank={Raz_bank.shape[1]}, required={future_rows}). fallback to basis."
            )
        else:
            log("[WARN] predictor-mode direct/hybrid requested, but direct residual bank not found. fallback to basis.")
        predictor_mode = "basis"

    if has_direct:
        neigh_dir_az = Raz_bank[idx]
        neigh_dir_el = Rel_bank[idx]
        pred_dir_az_future = circular_mean_weighted_deg(neigh_dir_az, w)
        pred_dir_el_future = np.sum(w.reshape(-1, 1) * neigh_dir_el, axis=0)
        if len(pred_dir_az_future) != future_rows:
            pred_dir_az_future = pred_dir_az_future[:future_rows]
            pred_dir_el_future = pred_dir_el_future[:future_rows]
    else:
        neigh_dir_az = None
        neigh_dir_el = None
        pred_dir_az_future = None
        pred_dir_el_future = None

    if predictor_mode == "direct" and pred_dir_az_future is not None and pred_dir_el_future is not None:
        pred_res_az_future = pred_dir_az_future
        pred_res_el_future = pred_dir_el_future
    elif predictor_mode == "hybrid" and pred_dir_az_future is not None and pred_dir_el_future is not None:
        pred_res_az_future = circular_blend_deg(pred_basis_az_future, pred_dir_az_future, hybrid_weight)
        pred_res_el_future = (1.0 - hybrid_weight) * pred_basis_el_future + hybrid_weight * pred_dir_el_future
    else:
        predictor_mode = "basis"
        pred_res_az_future = pred_basis_az_future
        pred_res_el_future = pred_basis_el_future

    if obs_used:
        pred_res_az_future = wrap180(float(obs_sign_az) * pred_res_az_future)
        pred_res_el_future = float(obs_sign_el) * pred_res_el_future

    # OOD confidence from nearest analog distance
    d = np.sqrt(np.maximum(0.0, np.sum((Xbank_z - xz.reshape(1, -1)) ** 2, axis=1)))
    d0 = float(np.min(d))
    dist_ref = float(meta.get("dist_ref", 1.0))
    if not np.isfinite(dist_ref) or dist_ref <= 1e-9:
        dist_ref = 1.0
    conf_raw = float(np.exp(-((d0 / dist_ref) ** 2)))
    conf = float(max(ood_floor, conf_raw))
    if obs_used and obs_conf_floor > 0.0:
        conf = float(max(conf, obs_conf_floor))
    corr_scale = 1.0 if conf >= min_ood_apply else 0.0

    # Neighbor-based uncertainty and baseline-error estimates for auto no-harm gate.
    neigh_basis_az = Caz_bank[idx] @ basis_f.T
    neigh_basis_el = Cel_bank[idx] @ basis_f.T
    if predictor_mode == "direct" and neigh_dir_az is not None and neigh_dir_el is not None:
        neigh_az = neigh_dir_az
        neigh_el = neigh_dir_el
    elif predictor_mode == "hybrid" and neigh_dir_az is not None and neigh_dir_el is not None:
        neigh_az = np.stack(
            [circular_blend_deg(neigh_basis_az[i], neigh_dir_az[i], hybrid_weight) for i in range(len(idx))],
            axis=0,
        )
        neigh_el = (1.0 - hybrid_weight) * neigh_basis_el + hybrid_weight * neigh_dir_el
    else:
        neigh_az = neigh_basis_az
        neigh_el = neigh_basis_el
    wv = w.reshape(-1, 1)
    mean_az = np.sum(wv * neigh_az, axis=0)
    mean_el = np.sum(wv * neigh_el, axis=0)
    var_az = np.sum(wv * (neigh_az - mean_az.reshape(1, -1)) ** 2, axis=0)
    var_el = np.sum(wv * (neigh_el - mean_el.reshape(1, -1)) ** 2, axis=0)
    unc_az_rms = float(np.sqrt(np.mean(np.maximum(var_az, 0.0))))
    unc_el_rms = float(np.sqrt(np.mean(np.maximum(var_el, 0.0))))
    neighbor_dist_weighted = float(np.sum(w * d[idx]))

    if base_azel_bank is not None and len(base_azel_bank) == len(Xbank_z):
        est_base_azel_rmse = float(np.sum(w * base_azel_bank[idx]))
        est_base_az_rmse = float(np.sum(w * base_az_bank[idx])) if base_az_bank is not None else np.nan
        est_base_el_rmse = float(np.sum(w * base_el_bank[idx])) if base_el_bank is not None else np.nan
    else:
        est_base_az_rmse = float(np.sqrt(np.mean((az_base[fit_rows:] - np.mean(az_base[fit_rows:])) ** 2))) if future_rows > 0 else 0.0
        est_base_el_rmse = float(np.sqrt(np.mean((el_base[fit_rows:] - np.mean(el_base[fit_rows:])) ** 2))) if future_rows > 0 else 0.0
        est_base_azel_rmse = 0.5 * (est_base_az_rmse + est_base_el_rmse)

    pred_res_az_future = wrap180(corr_scale * conf * pred_res_az_future)
    pred_res_el_future = corr_scale * conf * pred_res_el_future

    corr_rms_az = float(np.sqrt(np.mean((alpha_az * pred_res_az_future) ** 2))) if future_rows > 0 else 0.0
    corr_rms_el = float(np.sqrt(np.mean((alpha_el * pred_res_el_future) ** 2))) if future_rows > 0 else 0.0
    corr_rms_azel = 0.5 * (corr_rms_az + corr_rms_el)
    corr_ratio = float(corr_rms_azel / max(1e-12, est_base_azel_rmse))

    gate_flags: List[str] = []
    if auto_lowerr_th is not None and est_base_azel_rmse < auto_lowerr_th:
        corr_scale = 0.0
        gate_flags.append("lowerr")
    if auto_max_neighbor_dist is not None and d0 > auto_max_neighbor_dist:
        corr_scale = 0.0
        gate_flags.append("dist")
    if auto_max_unc_az is not None and unc_az_rms > auto_max_unc_az:
        corr_scale = 0.0
        gate_flags.append("unc_az")
    if auto_max_unc_el is not None and unc_el_rms > auto_max_unc_el:
        corr_scale = 0.0
        gate_flags.append("unc_el")
    if auto_max_corr_ratio is not None and corr_ratio > auto_max_corr_ratio:
        corr_scale = 0.0
        gate_flags.append("corr_ratio")
    gate_reason = ",".join(gate_flags) if gate_flags else "none"

    if corr_scale <= 0.0:
        pred_res_az_future[:] = 0.0
        pred_res_el_future[:] = 0.0

    if cap_deg > 0:
        pred_res_az_future = np.clip(pred_res_az_future, -cap_deg, cap_deg)
        pred_res_el_future = np.clip(pred_res_el_future, -cap_deg, cap_deg)

    # Smooth ramp at start of future segment to avoid discontinuity at day-7 boundary.
    pred_err_az = np.zeros((rows,), dtype=np.float64)
    pred_err_el = np.zeros((rows,), dtype=np.float64)
    pred_err_az[fit_rows:] = pred_res_az_future
    pred_err_el[fit_rows:] = pred_res_el_future
    ramp_rows = int(max(0, round(float(transition_h) * 60.0 / max(1, step_minutes))))
    if ramp_rows > 0:
        end = min(rows, fit_rows + ramp_rows)
        rr = end - fit_rows
        if rr > 0:
            ramp = np.linspace(0.0, 1.0, rr, endpoint=True)
            pred_err_az[fit_rows:end] = wrap180(pred_err_az[fit_rows:end] * ramp)
            pred_err_el[fit_rows:end] = pred_err_el[fit_rows:end] * ramp

    corrected_df = base_df.copy()
    corrected_df["AZ"] = wrap360(az_base - alpha_az * pred_err_az)
    corrected_df["EL"] = el_base - alpha_el * pred_err_el

    out_corr = Path(args.out_corrected_csv)
    out_pred = Path(args.out_pred_error_csv)
    if out_corr.exists() and not args.overwrite:
        raise FileExistsError(f"exists: {out_corr} (use --overwrite)")
    if out_pred.exists() and not args.overwrite:
        raise FileExistsError(f"exists: {out_pred} (use --overwrite)")
    ensure_dir(out_corr.parent)
    ensure_dir(out_pred.parent)
    corrected_df.to_csv(out_corr, index=False)

    pred_df = base_df[["No", "date", "UNIX"]].copy()
    pred_df["pred_error_AZ"] = pred_err_az
    pred_df["pred_error_EL"] = pred_err_el
    pred_df["fit_window_rows"] = int(fit_rows)
    pred_df["ood_conf"] = float(conf)
    pred_df["ood_conf_raw"] = float(conf_raw)
    pred_df["corr_scale"] = float(corr_scale)
    pred_df["corr_rms_az"] = float(corr_rms_az)
    pred_df["corr_rms_el"] = float(corr_rms_el)
    pred_df["corr_rms_azel"] = float(corr_rms_azel)
    pred_df["corr_ratio"] = float(corr_ratio)
    pred_df["unc_az_rms"] = float(unc_az_rms)
    pred_df["unc_el_rms"] = float(unc_el_rms)
    pred_df["neighbor0_dist"] = float(d0)
    pred_df["neighbor_dist_w"] = float(neighbor_dist_weighted)
    pred_df["est_base_az_rmse"] = float(est_base_az_rmse)
    pred_df["est_base_el_rmse"] = float(est_base_el_rmse)
    pred_df["est_base_azel_rmse"] = float(est_base_azel_rmse)
    pred_df["gate_reason"] = gate_reason
    pred_df["obs_assim_used"] = bool(obs_used)
    pred_df["obs_valid_rows"] = int(obs_valid_rows)
    pred_df["obs_weight_gain"] = float(obs_weight_gain)
    pred_df["obs_sign_az"] = float(obs_sign_az)
    pred_df["obs_sign_el"] = float(obs_sign_el)
    pred_df["obs_conf_floor"] = float(obs_conf_floor)
    pred_df["alpha_AZ"] = float(alpha_az)
    pred_df["alpha_EL"] = float(alpha_el)
    pred_df["predictor_mode"] = str(predictor_mode)
    pred_df["hybrid_weight"] = float(hybrid_weight)
    pred_df["k"] = int(k)
    pred_df["tau"] = float(tau)
    pred_df.to_csv(out_pred, index=False)

    neighbors = []
    for j, ww in zip(idx.tolist(), w.tolist()):
        neighbors.append({"stem": stem_list[int(j)], "weight": float(ww), "distance": float(d[int(j)])})
    nb_path = out_pred.parent / f"{out_pred.stem}_neighbors.json"
    nb_path.write_text(json.dumps({"neighbors": neighbors}, ensure_ascii=False, indent=2), encoding="utf-8")

    log(
        f"[INFO] mode={predictor_mode} nearest distance={d0:.6g} conf={conf:.6g} corr_scale={corr_scale:.6g} "
        f"est_base_azel={est_base_azel_rmse:.6g} unc_az={unc_az_rms:.6g} unc_el={unc_el_rms:.6g} "
        f"corr_ratio={corr_ratio:.6g} gate={gate_reason} "
        f"obs_used={obs_used} obs_valid_rows={obs_valid_rows} "
        f"obs_sign=({obs_sign_az:+.0f},{obs_sign_el:+.0f})"
    )
    log("[INFO] top neighbors: " + ", ".join([f"{n['stem']}({n['weight']:.3f})" for n in neighbors[:5]]))
    log(f"[SAVED] corrected: {out_corr}")
    log(f"[SAVED] pred error: {out_pred}")
    log(f"[SAVED] neighbors: {nb_path}")

    if args.truth_csv:
        truth_df = load_orbit_csv(Path(args.truth_csv), require_targets=True)
        met_full, merged_full = compute_metrics(truth_df=truth_df, baseline_df=base_df, corrected_df=corrected_df)
        met_future = compute_segment_metrics(
            truth_df=truth_df,
            baseline_df=base_df,
            corrected_df=corrected_df,
            start_no=fit_rows + 1,
        )
        met_future = met_future.copy()
        met_future["segment"] = "future_23d"
        met_full2 = met_full.copy()
        met_full2["segment"] = "full_30d"
        met_out = pd.concat([met_full2, met_future], ignore_index=True)

        mpath = Path(args.out_metrics_csv) if args.out_metrics_csv else (out_corr.parent / f"{out_corr.stem}_metrics.csv")
        if mpath.exists() and not args.overwrite:
            raise FileExistsError(f"exists: {mpath} (use --overwrite)")
        ensure_dir(mpath.parent)
        met_out.to_csv(mpath, index=False)

        bf = met_full[met_full["kind"] == "baseline"].iloc[0]
        cf = met_full[met_full["kind"] == "corrected"].iloc[0]
        bq = met_future[met_future["kind"] == "baseline"].iloc[0]
        cq = met_future[met_future["kind"] == "corrected"].iloc[0]
        log("[METRIC] AZ/EL RMSE full_30d baseline -> corrected")
        log(f"  AZ: {float(bf['AZ_RMSE']):.6g} -> {float(cf['AZ_RMSE']):.6g}")
        log(f"  EL: {float(bf['EL_RMSE']):.6g} -> {float(cf['EL_RMSE']):.6g}")
        log("[METRIC] AZ/EL RMSE future_23d baseline -> corrected")
        log(f"  AZ: {float(bq['AZ_RMSE']):.6g} -> {float(cq['AZ_RMSE']):.6g}")
        log(f"  EL: {float(bq['EL_RMSE']):.6g} -> {float(cq['EL_RMSE']):.6g}")
        log(f"[SAVED] metrics: {mpath}")

        if not args.no_plot:
            plot_dir = Path(args.plot_dir) if args.plot_dir else (out_corr.parent / "plots")
            plot_baseline_vs_corrected(
                stem=out_corr.stem,
                baseline_df=base_df,
                corrected_df=corrected_df,
                out_dir=plot_dir,
                sample_every=int(args.plot_sample_every),
            )
            plot_truth_baseline_corrected(
                stem=out_corr.stem,
                merged_df=merged_full,
                out_dir=plot_dir,
                sample_every=int(args.plot_sample_every),
            )
            plot_metric_summary(stem=out_corr.stem, metrics_df=met_full, out_dir=plot_dir)
            log(f"[SAVED] plots: {plot_dir}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GEO analog model: first 7 days (+optional previous 1 day) -> next 23 days AZ/EL correction"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train analog bank from historical datasets")
    t.add_argument("--tle-dir", default="23467_2023")
    t.add_argument("--pred-dir", default="23467_2023_csv")
    t.add_argument("--truth-csv", default="2023_calc_az_el.csv")
    t.add_argument(
        "--train-dataset",
        action="append",
        default=None,
        help=(
            "Multi-dataset train spec. Repeat this option. Format: "
            "'tle_dir|pred_dir|truth_csv' or 'name|tle_dir|pred_dir|truth_csv' "
            "(comma-separated equivalent is also accepted). "
            "If omitted, --tle-dir/--pred-dir/--truth-csv are used."
        ),
    )
    t.add_argument("--out-model", default="orbit_geo_7d23d_analog_model_v1")
    t.add_argument("--days-horizon", type=int, default=30)
    t.add_argument("--fit-days", type=int, default=7)
    t.add_argument("--pre-days", type=int, default=1)
    t.add_argument("--use-pre1d", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--step-minutes", type=int, default=1)
    t.add_argument("--sample-every", type=int, default=1)
    t.add_argument("--observer-lat", type=float, default=36.3022)
    t.add_argument("--observer-lon", type=float, default=137.9031)
    t.add_argument("--sat-name", default="23467")
    t.add_argument("--signature-downsample-minutes", type=int, default=30)
    t.add_argument("--time-weight-power", type=float, default=1.0)
    t.add_argument("--min-valid-future-rows", type=int, default=10080)
    t.add_argument("--val-split", type=float, default=0.2)
    t.add_argument("--split-mode", choices=["time", "hash"], default="time")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--k-grid", type=str, default="5,8,12,16,24")
    t.add_argument("--tau-grid", type=str, default="0.5,1,2,4,8")
    t.add_argument("--alpha-grid", type=str, default="0,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0")
    t.add_argument("--min-improved-file-ratio", type=float, default=0.55)
    t.add_argument("--alpha-infer-shrink", type=float, default=0.6,
                   help="Shrink selected alpha for inference (robustness against year shift)")
    t.add_argument("--hybrid-weight", type=float, default=0.5,
                   help="Default direct/basis blend ratio used when predictor-mode=hybrid")
    t.add_argument("--cap-deg", type=float, default=0.02)
    t.add_argument("--ood-conf-floor", type=float, default=0.1)
    t.add_argument("--min-ood-conf-apply", type=float, default=0.0)
    t.add_argument("--transition-hours", type=float, default=6.0)
    t.add_argument("--auto-lowerr-th", type=float, default=None,
                   help="If estimated baseline AZEL RMSE is lower than this, skip correction")
    t.add_argument("--auto-max-neighbor-dist", type=float, default=None,
                   help="If nearest-neighbor distance is larger than this, skip correction")
    t.add_argument("--auto-max-unc-az", type=float, default=None,
                   help="If neighbor AZ uncertainty RMS exceeds this, skip correction")
    t.add_argument("--auto-max-unc-el", type=float, default=None,
                   help="If neighbor EL uncertainty RMS exceeds this, skip correction")
    t.add_argument("--auto-max-corr-ratio", type=float, default=None,
                   help="If correction RMS / estimated baseline AZEL RMSE exceeds this, skip correction")
    t.add_argument("--max-files", type=int, default=None)
    t.add_argument("--overwrite", action="store_true")

    pr = sub.add_parser("predict", help="Predict from a single TLE")
    pr.add_argument("--model", required=True)
    pr.add_argument("--tle-file", required=True)
    pr.add_argument("--baseline-csv", default=None,
                    help="Optional 30-day baseline csv for the same TLE; if omitted, propagate with Skyfield")
    pr.add_argument("--days-horizon", type=int, default=None)
    pr.add_argument("--fit-days", type=int, default=None)
    pr.add_argument("--pre-days", type=int, default=None)
    pr.add_argument("--use-pre1d", type=lambda s: str(s).lower() in ("1", "true", "yes", "y"), default=None)
    pr.add_argument("--step-minutes", type=int, default=None)
    pr.add_argument("--sample-every", type=int, default=None)
    pr.add_argument("--observer-lat", type=float, default=36.3022)
    pr.add_argument("--observer-lon", type=float, default=137.9031)
    pr.add_argument("--sat-name", default="23467")
    pr.add_argument("--k", type=int, default=None)
    pr.add_argument("--tau", type=float, default=None)
    pr.add_argument("--alpha-az", type=float, default=None)
    pr.add_argument("--alpha-el", type=float, default=None)
    pr.add_argument("--predictor-mode", choices=["basis", "direct", "hybrid"], default="basis")
    pr.add_argument("--hybrid-weight", type=float, default=None)
    pr.add_argument("--obs-truth-csv", default=None,
                    help="Observed AZ/EL csv (truth-like). Rows before TLE time are used for online assimilation.")
    pr.add_argument("--obs-hours", type=float, default=12.0,
                    help="Hours of observation before TLE time used in online assimilation.")
    pr.add_argument("--obs-min-valid", type=int, default=60,
                    help="Minimum valid observed rows required to activate online assimilation.")
    pr.add_argument("--obs-assim-strength", type=float, default=2.0,
                    help="Weight strength for observation-consistency when reweighting neighbors.")
    pr.add_argument("--obs-pool-size", type=int, default=100,
                    help="Neighbor pool size for observation reweighting.")
    pr.add_argument("--obs-sign-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable online sign correction from pre-window observation fit.")
    pr.add_argument("--obs-sign-flip-margin", type=float, default=1e-6,
                    help="Minimum RMSE margin required to flip sign in online correction.")
    pr.add_argument("--obs-conf-floor", type=float, default=0.0,
                    help="When obs assimilation is used, raise confidence floor to this value.")
    pr.add_argument("--cap-deg", type=float, default=None)
    pr.add_argument("--ood-conf-floor", type=float, default=None)
    pr.add_argument("--min-ood-conf-apply", type=float, default=None)
    pr.add_argument("--transition-hours", type=float, default=None)
    pr.add_argument("--auto-lowerr-th", type=float, default=None)
    pr.add_argument("--auto-max-neighbor-dist", type=float, default=None)
    pr.add_argument("--auto-max-unc-az", type=float, default=None)
    pr.add_argument("--auto-max-unc-el", type=float, default=None)
    pr.add_argument("--auto-max-corr-ratio", type=float, default=None)
    pr.add_argument("--out-corrected-csv", required=True)
    pr.add_argument("--out-pred-error-csv", required=True)
    pr.add_argument("--truth-csv", default=None)
    pr.add_argument("--out-metrics-csv", default=None)
    pr.add_argument("--plot-dir", default=None)
    pr.add_argument("--plot-sample-every", type=int, default=5)
    pr.add_argument("--no-plot", action="store_true")
    pr.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    if args.cmd == "train":
        return train_cmd(args)
    if args.cmd == "predict":
        return predict_cmd(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
