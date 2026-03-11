#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

import ufo4_orbit_predict_tf as core


SIDEREAL_SECONDS = 86164.0905
SOLAR_DAY_SECONDS = 86400.0
COMMON_DAYS_PER_YEAR = 365.0


@dataclass
class ExperimentWindow:
    dates: list[str]
    unix: np.ndarray
    start_unix: int
    end_unix: int
    train_end_unix: int


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-TLE 90d baseline and 7d->83d TensorFlow AZ/EL experiment"
    )
    p.add_argument("--tle-file", default="", help="Single TLE txt file. Empty => first file in pred_tle.")
    p.add_argument("--tle-dir", default="pred_tle", help="Directory used when --tle-file is empty.")
    p.add_argument("--truth-file", default="2024_calc_az_el.csv")
    p.add_argument("--output-dir", default="single_tle_90d_tf_experiment")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--train-days", type=int, default=7)
    p.add_argument("--step-minutes", type=int, default=1)
    p.add_argument("--sat-name", default="23467")
    p.add_argument("--observer-lat", type=float, default=36.3022)
    p.add_argument("--observer-lon", type=float, default=137.9031)
    p.add_argument("--observer-alt-m", type=float, default=0.0)
    p.add_argument("--geo-radius-km", type=float, default=42164.0)
    p.add_argument("--solar-harmonics", type=int, default=2)
    p.add_argument("--sidereal-harmonics", type=int, default=4)
    p.add_argument("--yearly-harmonics", type=int, default=0)
    p.add_argument("--ridge", type=float, default=0.0, help="TensorFlow lstsq l2 regularizer")
    p.add_argument("--historical-analog-nearby-per-year", type=int, default=4)
    p.add_argument("--historical-analog-max-day-offset", type=int, default=50)
    p.add_argument("--historical-analog-min-gap-days", type=int, default=10)
    p.add_argument("--historical-analog-k-candidates", default="1,2,3,4,5,6,8,10,12,16")
    p.add_argument("--historical-analog-validation-years", default="2020,2021,2022,2023")
    p.add_argument("--stage2-poly-degrees", default="1,2,3")
    p.add_argument("--stage2-poly-ridge", type=float, default=1.0e-8)
    p.add_argument("--stage2-recent-validation-years", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-plot-points-per-month", type=int, default=12000)
    return p.parse_args()


def pick_tle_file(args: argparse.Namespace) -> Path:
    if str(args.tle_file).strip():
        p = Path(str(args.tle_file).strip())
        if not p.exists():
            raise FileNotFoundError(f"TLE file not found: {p}")
        return p
    tle_dir = Path(args.tle_dir)
    if not tle_dir.exists():
        raise FileNotFoundError(f"TLE dir not found: {tle_dir}")
    files = sorted(x for x in tle_dir.glob("*.txt") if x.is_file())
    if not files:
        raise RuntimeError(f"No TLE txt files found in {tle_dir}")
    return files[0]


def round_to_nearest_minute(d):
    floor_dt = d.replace(second=0, microsecond=0)
    if (d - floor_dt).total_seconds() >= 30.0:
        return floor_dt + timedelta(minutes=1)
    return floor_dt


def build_window(tle_path: Path, days: int, train_days: int, step_minutes: int) -> ExperimentWindow:
    d = core.parse_tle_datetime_from_stem(tle_path.stem)
    if d is None:
        raise ValueError(f"Cannot parse JST datetime from TLE filename: {tle_path.name}")
    start_jst = round_to_nearest_minute(d)
    start_unix = int(core.jst_naive_to_unix_seconds(start_jst))
    step_sec = int(step_minutes * 60)
    count = int(days * 24 * 60 // step_minutes)
    unix = start_unix + np.arange(count, dtype=np.int64) * step_sec
    dates = [core.unix_to_jst_naive(int(u)).strftime("%Y-%m-%d %H:%M:%S") for u in unix.tolist()]
    end_unix = int(unix[-1])
    train_end_unix = int(start_unix + train_days * 24 * 3600)
    return ExperimentWindow(
        dates=dates,
        unix=unix.astype(np.float64),
        start_unix=start_unix,
        end_unix=end_unix,
        train_end_unix=train_end_unix,
    )


def propagate_tle_azel_lla_at_unix(
    tle_path: Path,
    sat_name: str,
    observer_lat: float,
    observer_lon: float,
    unix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if core.EarthSatellite is None or core.load is None or core.wgs84 is None:
        raise RuntimeError("skyfield is not available")
    line1, line2 = core.read_tle_pair_lines(tle_path)
    ts = core.load.timescale()
    sat = core.EarthSatellite(line1, line2, sat_name, ts)
    obs = core.wgs84.latlon(observer_lat, observer_lon)
    yy, mo, dd, hh, mm, ss = core._unix_to_ymdhms(np.asarray(unix, dtype=np.float64))
    t = ts.utc(yy, mo, dd, hh, mm, ss)
    geo = sat.at(t)
    lat, lon = core.wgs84.latlon_of(geo)
    alt = core.wgs84.height_of(geo).km
    topo = (sat - obs).at(t)
    el, az, _ = topo.altaz()
    azel = np.column_stack([
        core.wrap360(np.asarray(az.degrees, dtype=np.float64)),
        np.asarray(el.degrees, dtype=np.float64),
    ]).astype(np.float64)
    lla = np.column_stack([
        np.asarray(lat.degrees, dtype=np.float64),
        np.asarray(lon.degrees, dtype=np.float64),
        np.asarray(alt, dtype=np.float64),
    ]).astype(np.float64)
    return azel, lla


def local_time_fields(unix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sec_local = np.asarray(np.round(unix), dtype=np.int64) + int(core.JST_OFFSET_SEC)
    dt_s = sec_local.astype("datetime64[s]")
    dt_d = dt_s.astype("datetime64[D]")
    y0 = dt_s.astype("datetime64[Y]")
    doy = (dt_d - y0).astype(np.int32)
    year_int = y0.astype(np.int64) + 1970
    is_leap = (year_int % 4 == 0) & ((year_int % 100 != 0) | (year_int % 400 == 0))
    doy = doy - ((is_leap) & (doy >= 60)).astype(np.int32)
    doy = np.clip(doy, 0, 364)
    sod = np.mod(sec_local.astype(np.float64), SOLAR_DAY_SECONDS)
    solar_phase = 2.0 * math.pi * sod / SOLAR_DAY_SECONDS
    sidereal_phase = 2.0 * math.pi * np.mod(sec_local.astype(np.float64), SIDEREAL_SECONDS) / SIDEREAL_SECONDS
    annual_phase = 2.0 * math.pi * (doy.astype(np.float64) + sod / SOLAR_DAY_SECONDS) / COMMON_DAYS_PER_YEAR
    dow = ((dt_d.astype(np.int64) + 3) % 7).astype(np.int32)
    return doy, sod, solar_phase, sidereal_phase, annual_phase, dow


def build_time_features(
    unix: np.ndarray,
    solar_h: int,
    sidereal_h: int,
    yearly_h: int,
) -> np.ndarray:
    doy, sod, solar_phase, sidereal_phase, annual_phase, dow = local_time_fields(unix)
    x = [
        np.ones_like(solar_phase),
    ]
    for k in range(1, int(solar_h) + 1):
        x += [np.sin(k * solar_phase), np.cos(k * solar_phase)]
    for k in range(1, int(sidereal_h) + 1):
        x += [np.sin(k * sidereal_phase), np.cos(k * sidereal_phase)]
    for k in range(1, int(yearly_h) + 1):
        x += [np.sin(k * annual_phase), np.cos(k * annual_phase)]
    return np.column_stack(x).astype(np.float32)


def build_trig_targets(azel: np.ndarray) -> np.ndarray:
    az = np.deg2rad(np.asarray(azel[:, 0], dtype=np.float64))
    el = np.deg2rad(np.asarray(azel[:, 1], dtype=np.float64))
    return np.column_stack([
        np.sin(az),
        np.cos(az),
        np.sin(el),
        np.cos(el),
    ]).astype(np.float32)


def decode_trig_targets(pred: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    az_pair = p[:, 0:2]
    el_pair = p[:, 2:4]
    az_norm = np.maximum(np.linalg.norm(az_pair, axis=1, keepdims=True), 1.0e-9)
    el_norm = np.maximum(np.linalg.norm(el_pair, axis=1, keepdims=True), 1.0e-9)
    az_pair = az_pair / az_norm
    el_pair = el_pair / el_norm
    az = core.wrap360(np.rad2deg(np.arctan2(az_pair[:, 0], az_pair[:, 1])))
    el = np.rad2deg(np.arctan2(el_pair[:, 0], el_pair[:, 1]))
    return np.column_stack([az, np.clip(el, -90.0, 90.0)]).astype(np.float64)


def fit_tf_linear_harmonic_model(tf, x_train: np.ndarray, y_train: np.ndarray, ridge: float) -> np.ndarray:
    x_tf = tf.constant(np.asarray(x_train, dtype=np.float32))
    y_tf = tf.constant(np.asarray(y_train, dtype=np.float32))
    w_tf = tf.linalg.lstsq(x_tf, y_tf, l2_regularizer=float(ridge), fast=False)
    return np.asarray(w_tf.numpy(), dtype=np.float64)


def build_unix_sincos_only_features(unix: np.ndarray) -> np.ndarray:
    x = np.asarray(unix, dtype=np.float64)
    return np.column_stack([np.sin(x), np.cos(x)]).astype(np.float32)


def fit_tf_unix_sincos_only_mlp(
    tf,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[object, dict]:
    tf.keras.utils.set_random_seed(int(seed))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(int(x_train.shape[1]),)),
            tf.keras.layers.Dense(128, activation="tanh"),
            tf.keras.layers.Dense(128, activation="tanh"),
            tf.keras.layers.Dense(64, activation="tanh"),
            tf.keras.layers.Dense(int(y_train.shape[1]), activation=None),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-3),
        loss="mse",
        metrics=["mae"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1.0e-5,
        ),
    ]
    hist = model.fit(
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        validation_data=(
            np.asarray(x_val, dtype=np.float32),
            np.asarray(y_val, dtype=np.float32),
        ),
        epochs=120,
        batch_size=256,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
    )
    payload = {k: [float(vv) for vv in v] for k, v in hist.history.items()}
    return model, payload


def parse_target_year_from_truth_file(truth_path: Path) -> int:
    stem = truth_path.stem
    prefix = stem.split("_", 1)[0]
    if prefix.isdigit() and len(prefix) == 4:
        return int(prefix)
    raise ValueError(f"Cannot parse target year from truth file name: {truth_path.name}")


def parse_int_csv(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def harmonic_predict_azel_from_first7(
    tf,
    window: ExperimentWindow,
    tle_azel: np.ndarray,
    solar_harmonics: int,
    sidereal_harmonics: int,
    yearly_harmonics: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = window.unix < float(window.train_end_unix)
    x_all = build_time_features(
        unix=window.unix,
        solar_h=int(solar_harmonics),
        sidereal_h=int(sidereal_harmonics),
        yearly_h=int(yearly_harmonics),
    )
    y_all = build_trig_targets(tle_azel)
    w = fit_tf_linear_harmonic_model(tf=tf, x_train=x_all[train_mask], y_train=y_all[train_mask], ridge=float(ridge))
    pred_trig = np.asarray(x_all, dtype=np.float64) @ w
    pred_azel = decode_trig_targets(pred_trig)
    pred_full = tle_azel.copy()
    pred_full[~train_mask] = pred_azel[~train_mask]
    return pred_full, w, x_all


def tle_static_features(tle_path: Path, sat_name: str) -> np.ndarray:
    line1, line2 = core.read_tle_pair_lines(tle_path)
    sat = core.EarthSatellite(line1, line2, sat_name, core.load.timescale())
    model = sat.model
    return np.asarray(
        [
            float(model.ecco),
            float(model.bstar),
            float(model.no_kozai),
            math.sin(float(model.inclo)),
            math.cos(float(model.inclo)),
            math.sin(float(model.nodeo)),
            math.cos(float(model.nodeo)),
            math.sin(float(model.argpo)),
            math.cos(float(model.argpo)),
            math.sin(float(model.mo)),
            math.cos(float(model.mo)),
        ],
        dtype=np.float64,
    )


def build_episode_descriptor(
    tle_path: Path,
    sat_name: str,
    harmonic_weights: np.ndarray,
    first7_azel: np.ndarray,
) -> np.ndarray:
    az = np.asarray(first7_azel[:, 0], dtype=np.float64)
    el = np.asarray(first7_azel[:, 1], dtype=np.float64)
    stats = np.asarray(
        [
            float(np.mean(az)),
            float(np.std(az)),
            float(np.mean(el)),
            float(np.std(el)),
            float(np.max(az) - np.min(az)),
            float(np.max(el) - np.min(el)),
        ],
        dtype=np.float64,
    )
    return np.concatenate([harmonic_weights.reshape(-1), tle_static_features(tle_path, sat_name), stats], axis=0)


def load_truth_cache(max_year: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in range(2017, int(max_year) + 1):
        p = Path(f"{year}_calc_az_el.csv")
        if p.exists():
            cache[year] = core.load_orbit_numeric_full(p)
    return cache


def pick_truth_window_from_cache(
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
    year: int,
    start_unix: int,
    end_unix: int,
    max_truth_year: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    all_u: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    for y in (year, year + 1):
        if y > int(max_truth_year):
            continue
        pair = cache.get(int(y))
        if pair is None:
            continue
        u, f = pair
        m = (u >= float(start_unix)) & (u <= float(end_unix))
        if np.any(m):
            all_u.append(u[m])
            all_f.append(f[m])
    if not all_u:
        return None, None
    u = np.concatenate(all_u)
    f = np.concatenate(all_f)
    order = np.argsort(u)
    return u[order], f[order]


def select_nearby_historical_tles(
    year: int,
    target_dt,
    nearby_per_year: int,
    max_day_offset: int,
    min_gap_days: int,
) -> list[tuple[Path, object]]:
    d = Path(f"23467_{year}")
    if not d.exists():
        return []
    target_local = target_dt.replace(year=int(year), second=0, microsecond=0)
    cand: list[tuple[float, Path, object]] = []
    for f in sorted(d.glob("*.txt")):
        dt = core.parse_tle_datetime_from_stem(f.stem)
        if dt is None:
            continue
        if abs((dt - target_local).total_seconds()) > float(max_day_offset) * 86400.0:
            continue
        cand.append((abs((dt - target_local).total_seconds()), f, dt))
    cand.sort(key=lambda x: x[0])
    out: list[tuple[Path, object]] = []
    used = []
    for _, f, dt in cand:
        if all(abs((dt - prev).total_seconds()) >= float(min_gap_days) * 86400.0 for prev in used):
            out.append((f, dt))
            used.append(dt)
        if len(out) >= int(nearby_per_year):
            break
    return out


def shift_series_edge_hold(arr: np.ndarray, shift_minutes: int) -> np.ndarray:
    src = np.asarray(arr, dtype=np.float32)
    if shift_minutes == 0:
        return src.copy()
    out = np.empty_like(src)
    if shift_minutes > 0:
        out[:-shift_minutes] = src[shift_minutes:]
        out[-shift_minutes:] = src[-1:]
        return out
    m = -int(shift_minutes)
    out[m:] = src[:-m]
    out[:m] = src[:1]
    return out


def build_historical_analog_episode(
    tf,
    tle_path: Path,
    episode_dt,
    year: int,
    truth_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    max_truth_year: int,
    sat_name: str,
    observer_lat: float,
    observer_lon: float,
    observer_alt_m: float,
    geo_radius_km: float,
    days: int,
    train_days: int,
    solar_harmonics: int,
    sidereal_harmonics: int,
    yearly_harmonics: int,
    ridge: float,
) -> dict | None:
    del episode_dt
    window = build_window(tle_path, days=int(days), train_days=int(train_days), step_minutes=1)
    truth_unix, truth_full = pick_truth_window_from_cache(
        cache=truth_cache,
        year=int(year),
        start_unix=window.start_unix,
        end_unix=window.end_unix,
        max_truth_year=int(max_truth_year),
    )
    if truth_unix is None or truth_full is None:
        return None
    raw_azel, raw_lla = propagate_tle_azel_lla_at_unix(
        tle_path=tle_path,
        sat_name=sat_name,
        observer_lat=float(observer_lat),
        observer_lon=float(observer_lon),
        unix=window.unix,
    )
    harm_azel, harm_weights, _ = harmonic_predict_azel_from_first7(
        tf=tf,
        window=window,
        tle_azel=raw_azel,
        solar_harmonics=int(solar_harmonics),
        sidereal_harmonics=int(sidereal_harmonics),
        yearly_harmonics=int(yearly_harmonics),
        ridge=float(ridge),
    )
    harm_lla = raw_lla.copy()
    forecast_mask_window = window.unix >= float(window.train_end_unix)
    harm_lla[forecast_mask_window] = core.azel_to_lla_geoshell(
        az_deg=harm_azel[forecast_mask_window, 0],
        el_deg=harm_azel[forecast_mask_window, 1],
        observer_lat_deg=float(observer_lat),
        observer_lon_deg=float(observer_lon),
        observer_alt_m=float(observer_alt_m),
        geo_radius_km=float(geo_radius_km),
    )
    pred_full = np.column_stack([harm_lla, harm_azel]).astype(np.float64)
    pred_aligned, true_aligned, unix_aligned = core.align_by_unix(window.unix, pred_full, truth_unix, truth_full)
    if pred_aligned.shape[0] < int(0.98 * len(window.unix)):
        return None
    train_mask = unix_aligned < float(window.train_end_unix)
    forecast_mask = ~train_mask
    residual = np.column_stack(
        [
            core.angle_diff_deg(true_aligned[:, 3], pred_aligned[:, 3]),
            true_aligned[:, 4] - pred_aligned[:, 4],
        ]
    ).astype(np.float32)
    desc = build_episode_descriptor(
        tle_path=tle_path,
        sat_name=sat_name,
        harmonic_weights=harm_weights,
        first7_azel=raw_azel[window.unix < float(window.train_end_unix)],
    )
    return {
        "tle_file": str(tle_path),
        "year": int(year),
        "start_unix": int(window.start_unix),
        "start_jst": window.dates[0],
        "window": window,
        "descriptor": desc,
        "residual_forecast": residual[forecast_mask],
        "pred_full_azel": harm_azel,
        "pred_full_lla": harm_lla,
        "pred_aligned": pred_aligned,
        "true_aligned": true_aligned,
        "forecast_mask": forecast_mask,
        "metrics_forecast": core.compute_metrics_azel(true_aligned[forecast_mask], pred_aligned[forecast_mask])["overall"],
    }


def apply_shifted_analog_correction(
    target_episode: dict,
    candidate_episodes: list[dict],
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = np.stack([ep["descriptor"] for ep in candidate_episodes], axis=0)
    z = np.asarray(target_episode["descriptor"], dtype=np.float64)
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    dist = np.sqrt(np.mean(np.square((x - z[None, :]) / sigma), axis=1))
    order = np.argsort(dist)[: min(int(k_neighbors), len(candidate_episodes))]
    dsel = np.asarray(dist[order], dtype=np.float64)
    dsel = dsel - np.min(dsel)
    weights = np.exp(-dsel)
    wsum = float(np.sum(weights))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        weights = np.ones_like(dsel, dtype=np.float64)
        wsum = float(np.sum(weights))
    weights = weights / wsum
    target_start_sec = int(core.unix_to_jst_naive(int(target_episode["start_unix"])).hour * 3600 +
                           core.unix_to_jst_naive(int(target_episode["start_unix"])).minute * 60 +
                           core.unix_to_jst_naive(int(target_episode["start_unix"])).second)
    correction = np.zeros_like(candidate_episodes[int(order[0])]["residual_forecast"], dtype=np.float32)
    used = []
    for w, idx in zip(weights.tolist(), order.tolist()):
        ep = candidate_episodes[int(idx)]
        ep_dt = core.unix_to_jst_naive(int(ep["start_unix"]))
        ep_sec = int(ep_dt.hour * 3600 + ep_dt.minute * 60 + ep_dt.second)
        delta_sec = ((float(target_start_sec) - float(ep_sec) + 43200.0) % 86400.0) - 43200.0
        shift_minutes = int(round(delta_sec / 60.0))
        correction += float(w) * shift_series_edge_hold(ep["residual_forecast"], shift_minutes)
        used.append(
            {
                "tle_file": ep["tle_file"],
                "distance": float(dist[int(idx)]),
                "weight": float(w),
                "shift_minutes": int(shift_minutes),
                "year": int(ep["year"]),
            }
        )
    pred = np.asarray(target_episode["pred_aligned"], dtype=np.float64).copy()
    fm = np.asarray(target_episode["forecast_mask"], dtype=bool)
    pred[fm, 3] = core.wrap360(pred[fm, 3] + correction[:, 0])
    pred[fm, 4] = pred[fm, 4] + correction[:, 1]
    metrics = core.compute_metrics_azel(target_episode["true_aligned"][fm], pred[fm])["overall"]
    return pred, correction, {"used_episodes": used, "metrics_forecast": metrics}


def choose_analog_k_forward_cv(
    episodes_by_year: dict[int, list[dict]],
    k_candidates: list[int],
    validation_years: list[int],
) -> dict:
    scores: list[dict] = []
    for k in k_candidates:
        vals = []
        for year in validation_years:
            targets = episodes_by_year.get(int(year), [])
            pool = [ep for y, items in episodes_by_year.items() if int(y) < int(year) for ep in items]
            if not targets or not pool:
                continue
            for target in targets:
                _, _, info = apply_shifted_analog_correction(target_episode=target, candidate_episodes=pool, k_neighbors=int(k))
                vals.append(info["metrics_forecast"]["max_abs_error_max"])
        if vals:
            scores.append(
                {
                    "k": int(k),
                    "mean_max_abs_error": float(np.mean(vals)),
                    "median_max_abs_error": float(np.median(vals)),
                    "count": int(len(vals)),
                }
            )
    if not scores:
        raise RuntimeError("No forward-CV scores could be computed for historical analog model")
    scores.sort(key=lambda x: (x["mean_max_abs_error"], x["median_max_abs_error"], x["k"]))
    return {"selected_k": int(scores[0]["k"]), "scores": scores}


def build_stage2_poly_entry(
    target_episode: dict,
    corrected_aligned: np.ndarray,
) -> dict:
    fm = np.asarray(target_episode["forecast_mask"], dtype=bool)
    rel_days = np.arange(int(np.sum(fm)), dtype=np.float64) / 1440.0
    return {
        "episode": target_episode,
        "corrected_aligned": np.asarray(corrected_aligned, dtype=np.float64),
        "rel_days": rel_days,
        "az_residual": core.angle_diff_deg(
            corrected_aligned[fm, 3],
            target_episode["true_aligned"][fm, 3],
        ).astype(np.float64),
        "el_residual": (
            np.asarray(corrected_aligned[fm, 4], dtype=np.float64)
            - np.asarray(target_episode["true_aligned"][fm, 4], dtype=np.float64)
        ).astype(np.float64),
    }


def build_stage2_poly_features(rel_days: np.ndarray, degree: int) -> np.ndarray:
    rel = np.asarray(rel_days, dtype=np.float64)
    return np.column_stack([np.ones_like(rel)] + [rel ** d for d in range(1, int(degree) + 1)]).astype(np.float64)


def fit_stage2_global_poly_coefficients(
    train_entries: list[dict],
    degree: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not train_entries:
        raise RuntimeError("stage2 poly training entries are empty")
    feat_dim = int(degree) + 1
    xtx = np.zeros((feat_dim, feat_dim), dtype=np.float64)
    xty_az = np.zeros(feat_dim, dtype=np.float64)
    xty_el = np.zeros(feat_dim, dtype=np.float64)
    for entry in train_entries:
        x = build_stage2_poly_features(entry["rel_days"], degree=int(degree))
        xtx += x.T @ x
        xty_az += x.T @ np.asarray(entry["az_residual"], dtype=np.float64)
        xty_el += x.T @ np.asarray(entry["el_residual"], dtype=np.float64)
    xtx += float(ridge) * np.eye(feat_dim, dtype=np.float64)
    caz = np.linalg.solve(xtx, xty_az)
    cel = np.linalg.solve(xtx, xty_el)
    return caz, cel


def build_stage2_poly_correction_series(
    count: int,
    degree: int,
    coeffs: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rel_days = np.arange(int(count), dtype=np.float64) / 1440.0
    x = build_stage2_poly_features(rel_days, degree=int(degree))
    caz, cel = coeffs
    return x @ np.asarray(caz, dtype=np.float64), x @ np.asarray(cel, dtype=np.float64)


def apply_stage2_global_poly_to_entry(
    entry: dict,
    degree: int,
    coeffs: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(entry["corrected_aligned"], dtype=np.float64).copy()
    fm = np.asarray(entry["episode"]["forecast_mask"], dtype=bool)
    az_corr, el_corr = build_stage2_poly_correction_series(int(np.sum(fm)), degree=int(degree), coeffs=coeffs)
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    pred[fm, 4] = pred[fm, 4] - el_corr
    metrics = core.compute_metrics_azel(entry["episode"]["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def choose_stage2_poly_degree_forward_cv(
    entries_by_year: dict[int, list[dict]],
    degree_candidates: list[int],
    validation_years: list[int],
    recent_year_count: int,
    ridge: float,
) -> dict:
    years = sorted(int(y) for y in validation_years if entries_by_year.get(int(y)))
    if not years:
        raise RuntimeError("No validation years available for stage2 poly selection")
    if int(recent_year_count) > 0:
        years = years[-int(recent_year_count):]
    weights = np.arange(1, len(years) + 1, dtype=np.float64)
    scores: list[dict] = []
    for degree in degree_candidates:
        per_year_scores: list[dict] = []
        weighted_values: list[float] = []
        weighted_weights: list[float] = []
        for weight, year in zip(weights.tolist(), years):
            train_entries = [ep for y, items in entries_by_year.items() if int(y) < int(year) for ep in items]
            targets = entries_by_year.get(int(year), [])
            if not train_entries or not targets:
                continue
            coeffs = fit_stage2_global_poly_coefficients(
                train_entries=train_entries,
                degree=int(degree),
                ridge=float(ridge),
            )
            vals = []
            for target in targets:
                _, metrics = apply_stage2_global_poly_to_entry(
                    entry=target,
                    degree=int(degree),
                    coeffs=coeffs,
                )
                vals.append(float(metrics["max_abs_error_max"]))
            if vals:
                mean_val = float(np.mean(vals))
                per_year_scores.append(
                    {
                        "year": int(year),
                        "weight": float(weight),
                        "mean_max_abs_error": mean_val,
                        "count": int(len(vals)),
                    }
                )
                weighted_values.append(float(weight) * mean_val)
                weighted_weights.append(float(weight))
        if per_year_scores:
            scores.append(
                {
                    "degree": int(degree),
                    "weighted_mean_max_abs_error": float(np.sum(weighted_values) / np.sum(weighted_weights)),
                    "mean_max_abs_error": float(np.mean([x["mean_max_abs_error"] for x in per_year_scores])),
                    "per_year_scores": per_year_scores,
                }
            )
    if not scores:
        raise RuntimeError("No stage2 poly CV scores could be computed")
    scores.sort(key=lambda x: (x["weighted_mean_max_abs_error"], x["mean_max_abs_error"], x["degree"]))
    return {
        "selected_degree": int(scores[0]["degree"]),
        "used_validation_years": [int(y) for y in years],
        "scores": scores,
    }


def build_stage3_overlay_item(
    episode: dict,
    pred_stage2_aligned: np.ndarray,
) -> dict:
    fm = np.asarray(episode["forecast_mask"], dtype=bool)
    residual = np.column_stack(
        [
            core.angle_diff_deg(episode["true_aligned"][fm, 3], pred_stage2_aligned[fm, 3]),
            episode["true_aligned"][fm, 4] - pred_stage2_aligned[fm, 4],
        ]
    ).astype(np.float64)
    return {
        "episode": episode,
        "pred_stage2_aligned": np.asarray(pred_stage2_aligned, dtype=np.float64),
        "residual_forecast": residual,
    }


def smooth_stage3_overlay_series(residual_forecast: np.ndarray, block_minutes: int) -> np.ndarray:
    src = np.asarray(residual_forecast, dtype=np.float64)
    if int(block_minutes) <= 1:
        return src.copy()
    out = np.empty_like(src)
    block = int(block_minutes)
    for i in range(0, src.shape[0], block):
        out[i : i + block] = np.mean(src[i : i + block], axis=0, keepdims=True)
    return out


def build_recent_stage3_overlay(
    train_items: list[dict],
    recent_years: int,
    block_minutes: int,
) -> np.ndarray:
    if not train_items:
        raise RuntimeError("stage3 overlay train_items are empty")
    years = [int(item["episode"]["year"]) for item in train_items]
    max_year = max(years)
    usable = [
        item
        for item in train_items
        if int(item["episode"]["year"]) >= int(max_year) - int(recent_years) + 1
    ]
    if not usable:
        usable = train_items
    out = np.zeros_like(np.asarray(usable[0]["residual_forecast"], dtype=np.float64))
    for item in usable:
        out += smooth_stage3_overlay_series(item["residual_forecast"], block_minutes=int(block_minutes))
    return out / float(len(usable))


def build_weighted_stage3_overlay(
    train_items: list[dict],
    target_episode: dict,
    recent_years: int,
    block_minutes: int,
    k_neighbors: int,
) -> np.ndarray:
    if not train_items:
        raise RuntimeError("stage3 weighted overlay train_items are empty")
    years = [int(item["episode"]["year"]) for item in train_items]
    max_year = max(years)
    usable = [
        item
        for item in train_items
        if int(item["episode"]["year"]) >= int(max_year) - int(recent_years) + 1
    ]
    if not usable:
        usable = train_items
    x = np.stack([np.asarray(item["episode"]["descriptor"], dtype=np.float64) for item in usable], axis=0)
    z = np.asarray(target_episode["descriptor"], dtype=np.float64)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    dist = np.sqrt(np.mean(np.square((x - z[None, :]) / sigma), axis=1))
    order = np.argsort(dist)[: min(int(k_neighbors), len(usable))]
    dsel = np.asarray(dist[order], dtype=np.float64)
    dsel = dsel - np.min(dsel)
    weights = np.exp(-dsel)
    wsum = float(np.sum(weights))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        weights = np.ones_like(dsel, dtype=np.float64)
        wsum = float(np.sum(weights))
    weights = weights / wsum
    out = np.zeros_like(np.asarray(usable[int(order[0])]["residual_forecast"], dtype=np.float64))
    for w, idx in zip(weights.tolist(), order.tolist()):
        out += float(w) * smooth_stage3_overlay_series(
            usable[int(idx)]["residual_forecast"],
            block_minutes=int(block_minutes),
        )
    return out


def apply_stage3_piecewise_overlay_to_entry(
    item: dict,
    overlay_forecast: np.ndarray,
    bounds_days: tuple[int, int],
    alpha_segments: tuple[float, float, float],
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(item["pred_stage2_aligned"], dtype=np.float64).copy()
    fm = np.asarray(item["episode"]["forecast_mask"], dtype=bool)
    corr = np.asarray(overlay_forecast, dtype=np.float64)
    count = int(np.sum(fm))
    if corr.shape[0] != count:
        raise ValueError("stage3 overlay length mismatch")
    b1 = int(bounds_days[0]) * 1440
    b2 = int(bounds_days[1]) * 1440
    b1 = max(0, min(count, b1))
    b2 = max(b1, min(count, b2))
    az_corr = np.zeros(count, dtype=np.float64)
    a1, a2, a3 = [float(x) for x in alpha_segments]
    az_corr[:b1] = a1 * corr[:b1, 0]
    az_corr[b1:b2] = a2 * corr[b1:b2, 0]
    az_corr[b2:] = a3 * corr[b2:, 0]
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    metrics = core.compute_metrics_azel(item["episode"]["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def choose_stage3_piecewise_overlay_forward_cv(
    items_by_year: dict[int, list[dict]],
    validation_years: list[int],
    recent_year_candidates: list[int],
    block_minutes_candidates: list[int],
    bounds_day_candidates: list[tuple[int, int]],
    alpha1_candidates: list[float],
    alpha2_candidates: list[float],
    alpha3_candidates: list[float],
) -> dict:
    scores: list[dict] = []
    for recent_years in recent_year_candidates:
        for block_minutes in block_minutes_candidates:
            for bounds_days in bounds_day_candidates:
                for a1 in alpha1_candidates:
                    for a2 in alpha2_candidates:
                        for a3 in alpha3_candidates:
                            vals = []
                            for year in validation_years:
                                train = [it for yy, items in items_by_year.items() if int(yy) < int(year) for it in items]
                                targets = items_by_year.get(int(year), [])
                                if not train or not targets:
                                    continue
                                overlay = build_recent_stage3_overlay(
                                    train_items=train,
                                    recent_years=int(recent_years),
                                    block_minutes=int(block_minutes),
                                )
                                for target in targets:
                                    _, metrics = apply_stage3_piecewise_overlay_to_entry(
                                        item=target,
                                        overlay_forecast=overlay,
                                        bounds_days=bounds_days,
                                        alpha_segments=(float(a1), float(a2), float(a3)),
                                    )
                                    vals.append(float(metrics["max_abs_error_max"]))
                            if vals:
                                scores.append(
                                    {
                                        "recent_years": int(recent_years),
                                        "block_minutes": int(block_minutes),
                                        "bounds_days": [int(bounds_days[0]), int(bounds_days[1])],
                                        "alpha_segments": [float(a1), float(a2), float(a3)],
                                        "val_mean_max_abs_error": float(np.mean(vals)),
                                        "val_max_abs_error": float(np.max(vals)),
                                        "count": int(len(vals)),
                                    }
                                )
    if not scores:
        raise RuntimeError("No stage3 overlay CV scores could be computed")
    scores.sort(
        key=lambda x: (
            x["val_max_abs_error"],
            x["val_mean_max_abs_error"],
            x["recent_years"],
            x["block_minutes"],
            x["bounds_days"],
            x["alpha_segments"],
        )
    )
    best = scores[0]
    return {
        "selected_recent_years": int(best["recent_years"]),
        "selected_block_minutes": int(best["block_minutes"]),
        "selected_bounds_days": [int(x) for x in best["bounds_days"]],
        "selected_alpha_segments": [float(x) for x in best["alpha_segments"]],
        "scores": scores,
    }


def choose_stage3_overlay_family_forward_cv(
    items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    # Balance worst-case with average stability across the most recent forward years.
    score_weight_mean = 0.2
    scores: list[dict] = []

    for recent_years in [2, 3]:
        for block_minutes in [60]:
            for bounds_days in [(28, 60), (30, 60), (30, 68), (35, 65)]:
                for alpha_segments in [(0.6, 0.6, 1.0), (0.6, 0.8, 1.0), (0.8, 0.6, 1.0)]:
                    vals = []
                    for year in validation_years:
                        train = [it for yy, items in items_by_year.items() if int(yy) < int(year) for it in items]
                        targets = items_by_year.get(int(year), [])
                        if not train or not targets:
                            continue
                        overlay = build_recent_stage3_overlay(
                            train_items=train,
                            recent_years=int(recent_years),
                            block_minutes=int(block_minutes),
                        )
                        for target in targets:
                            _, metrics = apply_stage3_piecewise_overlay_to_entry(
                                item=target,
                                overlay_forecast=overlay,
                                bounds_days=bounds_days,
                                alpha_segments=alpha_segments,
                            )
                            vals.append(float(metrics["max_abs_error_max"]))
                    if vals:
                        val_mean = float(np.mean(vals))
                        val_max = float(np.max(vals))
                        scores.append(
                            {
                                "family": "global",
                                "recent_years": int(recent_years),
                                "block_minutes": int(block_minutes),
                                "bounds_days": [int(bounds_days[0]), int(bounds_days[1])],
                                "alpha_segments": [float(x) for x in alpha_segments],
                                "val_mean_max_abs_error": val_mean,
                                "val_max_abs_error": val_max,
                                "selection_score": float(val_max + score_weight_mean * val_mean),
                                "count": int(len(vals)),
                            }
                        )

    for local_block_minutes in [30, 60]:
        for local_k_neighbors in [8, 12]:
            for blend_beta in [0.25, 0.5, 0.75]:
                for bounds_days in [(28, 60), (60, 74), (74, 76)]:
                    for alpha_segments in [(0.6, 0.6, 0.9), (0.6, 0.6, 1.0), (0.6, 0.8, 1.0), (0.8, 0.6, 1.0)]:
                        vals = []
                        for year in validation_years:
                            train = [it for yy, items in items_by_year.items() if int(yy) < int(year) for it in items]
                            targets = items_by_year.get(int(year), [])
                            if not train or not targets:
                                continue
                            global_overlay = build_recent_stage3_overlay(
                                train_items=train,
                                recent_years=3,
                                block_minutes=60,
                            )
                            for target in targets:
                                local_overlay = build_weighted_stage3_overlay(
                                    train_items=train,
                                    target_episode=target["episode"],
                                    recent_years=3,
                                    block_minutes=int(local_block_minutes),
                                    k_neighbors=int(local_k_neighbors),
                                )
                                blend_overlay = (
                                    (1.0 - float(blend_beta)) * np.asarray(global_overlay, dtype=np.float64)
                                    + float(blend_beta) * np.asarray(local_overlay, dtype=np.float64)
                                )
                                _, metrics = apply_stage3_piecewise_overlay_to_entry(
                                    item=target,
                                    overlay_forecast=blend_overlay,
                                    bounds_days=bounds_days,
                                    alpha_segments=alpha_segments,
                                )
                                vals.append(float(metrics["max_abs_error_max"]))
                        if vals:
                            val_mean = float(np.mean(vals))
                            val_max = float(np.max(vals))
                            scores.append(
                                {
                                    "family": "blend",
                                    "recent_years": 3,
                                    "block_minutes": 60,
                                    "local_block_minutes": int(local_block_minutes),
                                    "local_k_neighbors": int(local_k_neighbors),
                                    "blend_beta": float(blend_beta),
                                    "bounds_days": [int(bounds_days[0]), int(bounds_days[1])],
                                    "alpha_segments": [float(x) for x in alpha_segments],
                                    "val_mean_max_abs_error": val_mean,
                                    "val_max_abs_error": val_max,
                                    "selection_score": float(val_max + score_weight_mean * val_mean),
                                    "count": int(len(vals)),
                                }
                            )

    if not scores:
        raise RuntimeError("No stage3 overlay family CV scores could be computed")
    scores.sort(
        key=lambda x: (
            x["selection_score"],
            x["val_max_abs_error"],
            x["val_mean_max_abs_error"],
            x["family"],
        )
    )
    best = scores[0]
    return {
        "selected_family": str(best["family"]),
        "selected_recent_years": int(best["recent_years"]),
        "selected_block_minutes": int(best["block_minutes"]),
        "selected_local_block_minutes": int(best.get("local_block_minutes", 0)),
        "selected_local_k_neighbors": int(best.get("local_k_neighbors", 0)),
        "selected_blend_beta": float(best.get("blend_beta", 0.0)),
        "selected_bounds_days": [int(x) for x in best["bounds_days"]],
        "selected_alpha_segments": [float(x) for x in best["alpha_segments"]],
        "selection_score_weight_mean": float(score_weight_mean),
        "scores": scores,
    }


def build_stage3_overlay_from_cv(
    train_items: list[dict],
    target_episode: dict,
    stage3_cv: dict,
) -> np.ndarray:
    overlay = build_recent_stage3_overlay(
        train_items=train_items,
        recent_years=int(stage3_cv["selected_recent_years"]),
        block_minutes=int(stage3_cv["selected_block_minutes"]),
    )
    if str(stage3_cv["selected_family"]) == "blend":
        local_overlay = build_weighted_stage3_overlay(
            train_items=train_items,
            target_episode=target_episode,
            recent_years=int(stage3_cv["selected_recent_years"]),
            block_minutes=int(stage3_cv["selected_local_block_minutes"]),
            k_neighbors=int(stage3_cv["selected_local_k_neighbors"]),
        )
        overlay = (
            (1.0 - float(stage3_cv["selected_blend_beta"])) * np.asarray(overlay, dtype=np.float64)
            + float(stage3_cv["selected_blend_beta"]) * np.asarray(local_overlay, dtype=np.float64)
        )
    return np.asarray(overlay, dtype=np.float64)


def build_stage4_local_overlay(
    train_items: list[dict],
    target_episode: dict,
    recent_years: int,
    local_block_minutes: int,
    local_k_neighbors: int,
    blend_beta: float,
) -> np.ndarray:
    global_overlay = build_recent_stage3_overlay(
        train_items=train_items,
        recent_years=int(recent_years),
        block_minutes=60,
    )
    local_overlay = build_weighted_stage3_overlay(
        train_items=train_items,
        target_episode=target_episode,
        recent_years=int(recent_years),
        block_minutes=int(local_block_minutes),
        k_neighbors=int(local_k_neighbors),
    )
    return (
        (1.0 - float(blend_beta)) * np.asarray(global_overlay, dtype=np.float64)
        + float(blend_beta) * np.asarray(local_overlay, dtype=np.float64)
    )


def apply_stage4_az_overlay_to_entry(
    pred_stage3_aligned: np.ndarray,
    episode: dict,
    overlay_forecast: np.ndarray,
    start_day: int,
    alpha_az: float,
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(pred_stage3_aligned, dtype=np.float64).copy()
    fm = np.asarray(episode["forecast_mask"], dtype=bool)
    corr = np.asarray(overlay_forecast, dtype=np.float64)
    count = int(np.sum(fm))
    if corr.shape[0] != count:
        raise ValueError("stage4 overlay length mismatch")
    start = max(0, min(count, int(start_day) * 1440))
    az_corr = np.zeros(count, dtype=np.float64)
    az_corr[start:] = float(alpha_az) * corr[start:, 0]
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    metrics = core.compute_metrics_azel(episode["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def choose_stage4_az_overlay_forward_cv(
    items_by_year: dict[int, list[dict]],
    stage3_items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    weights = {2022: 1.0, 2023: 2.0}
    score_weight_mean = 0.15
    scores: list[dict] = []

    baseline_vals = []
    for year in validation_years:
        targets = items_by_year.get(int(year), [])
        for target in targets:
            metrics = core.compute_metrics_azel(
                target["episode"]["true_aligned"][target["episode"]["forecast_mask"]],
                target["pred_stage3_aligned"][target["episode"]["forecast_mask"]],
            )["overall"]
            baseline_vals.append((int(year), float(metrics["max_abs_error_max"])))
    if baseline_vals:
        weighted_mean = float(
            np.sum([weights.get(y, 1.0) * v for y, v in baseline_vals]) /
            np.sum([weights.get(y, 1.0) for y, _ in baseline_vals])
        )
        scores.append(
            {
                "family": "none",
                "val_weighted_mean_max_abs_error": weighted_mean,
                "val_max_abs_error": float(np.max([v for _, v in baseline_vals])),
                "selection_score": float(np.max([v for _, v in baseline_vals]) + score_weight_mean * weighted_mean),
                "count": int(len(baseline_vals)),
            }
        )

    for recent_years in [2, 3]:
        for local_block_minutes in [15, 30]:
            for local_k_neighbors in [4, 8, 12]:
                for blend_beta in [0.5, 0.75]:
                    for start_day in [40, 45, 50, 55, 60]:
                        for alpha_az in [0.1, 0.15, 0.2, 0.25]:
                            vals = []
                            for year in validation_years:
                                train = [it for yy, items in stage3_items_by_year.items() if int(yy) < int(year) for it in items]
                                targets = items_by_year.get(int(year), [])
                                if not train or not targets:
                                    continue
                                for target in targets:
                                    overlay = build_stage4_local_overlay(
                                        train_items=train,
                                        target_episode=target["episode"],
                                        recent_years=int(recent_years),
                                        local_block_minutes=int(local_block_minutes),
                                        local_k_neighbors=int(local_k_neighbors),
                                        blend_beta=float(blend_beta),
                                    )
                                    _, metrics = apply_stage4_az_overlay_to_entry(
                                        pred_stage3_aligned=target["pred_stage3_aligned"],
                                        episode=target["episode"],
                                        overlay_forecast=overlay,
                                        start_day=int(start_day),
                                        alpha_az=float(alpha_az),
                                    )
                                    vals.append((int(year), float(metrics["max_abs_error_max"])))
                            if vals:
                                weighted_mean = float(
                                    np.sum([weights.get(y, 1.0) * v for y, v in vals]) /
                                    np.sum([weights.get(y, 1.0) for y, _ in vals])
                                )
                                val_max = float(np.max([v for _, v in vals]))
                                scores.append(
                                    {
                                        "family": "az_late_local_overlay",
                                        "recent_years": int(recent_years),
                                        "local_block_minutes": int(local_block_minutes),
                                        "local_k_neighbors": int(local_k_neighbors),
                                        "blend_beta": float(blend_beta),
                                        "start_day": int(start_day),
                                        "alpha_az": float(alpha_az),
                                        "val_weighted_mean_max_abs_error": weighted_mean,
                                        "val_max_abs_error": val_max,
                                        "selection_score": float(val_max + score_weight_mean * weighted_mean),
                                        "count": int(len(vals)),
                                    }
                                )

    if not scores:
        raise RuntimeError("No stage4 AZ overlay CV scores could be computed")
    scores.sort(key=lambda x: (x["selection_score"], x["val_max_abs_error"], x["val_weighted_mean_max_abs_error"], x["family"]))
    best = scores[0]
    return {
        "selected_family": str(best["family"]),
        "selected_recent_years": int(best.get("recent_years", 0)),
        "selected_local_block_minutes": int(best.get("local_block_minutes", 0)),
        "selected_local_k_neighbors": int(best.get("local_k_neighbors", 0)),
        "selected_blend_beta": float(best.get("blend_beta", 0.0)),
        "selected_start_day": int(best.get("start_day", 0)),
        "selected_alpha_az": float(best.get("alpha_az", 0.0)),
        "selection_score_weight_mean": float(score_weight_mean),
        "scores": scores,
    }


def clear_png_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob("*.png"):
        p.unlink()


def evaluate_and_save(
    name: str,
    out_dir: Path,
    dates: Sequence[str],
    pred_unix: np.ndarray,
    pred_lla: np.ndarray,
    pred_azel: np.ndarray,
    truth_unix: np.ndarray,
    truth_full: np.ndarray,
    forecast_start_unix: int,
    max_plot_points_per_month: int,
    extra_json: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predicted_calc_az_el.csv"
    core.write_prediction_csv(pred_csv, dates, pred_unix, pred_lla=pred_lla, pred_azel=pred_azel)

    pred_full = np.column_stack([pred_lla, pred_azel]).astype(np.float64)
    pred_aligned, true_aligned, unix_aligned = core.align_by_unix(pred_unix, pred_full, truth_unix, truth_full)

    metrics_full = core.compute_metrics_azel(true_aligned, pred_aligned)
    monthly_full = core.compute_monthly_azel_metrics(unix_aligned, true_aligned, pred_aligned)
    metrics_lla = core.compute_metrics_lla(true_aligned, pred_aligned)

    forecast_mask = np.asarray(unix_aligned, dtype=np.float64) >= float(forecast_start_unix)
    if not np.any(forecast_mask):
        raise RuntimeError("forecast mask is empty")
    metrics_forecast = core.compute_metrics_azel(true_aligned[forecast_mask], pred_aligned[forecast_mask])
    monthly_forecast = core.compute_monthly_azel_metrics(
        unix_aligned[forecast_mask], true_aligned[forecast_mask], pred_aligned[forecast_mask]
    )

    core.write_metrics_csv(out_dir / "metrics_az_el_full.csv", metrics_full)
    core.write_monthly_metrics_csv(out_dir / "metrics_monthly_az_el_full.csv", monthly_full)
    core.write_metrics_csv(out_dir / "metrics_az_el_forecast_83d.csv", metrics_forecast)
    core.write_monthly_metrics_csv(out_dir / "metrics_monthly_az_el_forecast_83d.csv", monthly_forecast)
    core.write_lla_metrics_csv(out_dir / "metrics_lla_full.csv", metrics_lla)

    clear_png_dir(out_dir / "monthly_plots_full")
    core.plot_monthly_azel(
        unix_aligned,
        true_aligned,
        pred_aligned,
        out_dir=out_dir / "monthly_plots_full",
        max_points_per_month=max_plot_points_per_month,
    )
    clear_png_dir(out_dir / "monthly_plots_forecast_83d")
    core.plot_monthly_azel(
        unix_aligned[forecast_mask],
        true_aligned[forecast_mask],
        pred_aligned[forecast_mask],
        out_dir=out_dir / "monthly_plots_forecast_83d",
        max_points_per_month=max_plot_points_per_month,
    )

    payload = {
        "name": name,
        "rows_aligned_full": int(true_aligned.shape[0]),
        "rows_aligned_forecast_83d": int(np.sum(forecast_mask)),
        "metrics_azel_full": metrics_full,
        "metrics_azel_forecast_83d": metrics_forecast,
        "monthly_metrics_azel_full": monthly_full,
        "monthly_metrics_azel_forecast_83d": monthly_forecast,
        "metrics_lla_full": metrics_lla,
        "forecast_start_unix": int(forecast_start_unix),
    }
    if extra_json:
        payload.update(extra_json)
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    tle_path = pick_tle_file(args)
    truth_path = Path(args.truth_file)
    if not truth_path.exists():
        raise FileNotFoundError(f"truth file not found: {truth_path}")

    core.require_tensorflow()
    tf = core.tf
    tf.random.set_seed(int(args.seed))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    window = build_window(tle_path, days=int(args.days), train_days=int(args.train_days), step_minutes=int(args.step_minutes))
    log(f"Using TLE: {tle_path}")
    log(f"Window JST: {window.dates[0]} -> {window.dates[-1]} ({len(window.unix)} rows)")
    log(f"Training period end JST: {core.unix_to_jst_naive(window.train_end_unix).strftime('%Y-%m-%d %H:%M:%S')}")

    tle_azel, tle_lla = propagate_tle_azel_lla_at_unix(
        tle_path=tle_path,
        sat_name=str(args.sat_name),
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        unix=window.unix,
    )
    truth_unix, truth_full = core.load_orbit_numeric_full(truth_path)

    baseline_dir = out_root / "baseline_single_tle_90d"
    baseline_metrics = evaluate_and_save(
        name="single_tle_90d_baseline",
        out_dir=baseline_dir,
        dates=window.dates,
        pred_unix=window.unix,
        pred_lla=tle_lla,
        pred_azel=tle_azel,
        truth_unix=truth_unix,
        truth_full=truth_full,
        forecast_start_unix=window.train_end_unix,
        max_plot_points_per_month=int(args.max_plot_points_per_month),
        extra_json={
            "tle_file": str(tle_path),
            "train_days": int(args.train_days),
            "total_days": int(args.days),
        },
    )
    log(f"Baseline 90d forecast max abs (83d only): {baseline_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}")

    train_mask = window.unix < float(window.train_end_unix)
    val_mask = (window.unix >= float(window.start_unix + max(1, int(args.train_days) - 1) * 24 * 3600)) & train_mask
    fit_mask = train_mask & (~val_mask)
    if not np.any(val_mask):
        raise RuntimeError("validation split is empty")

    X_all = build_time_features(
        unix=window.unix,
        solar_h=int(args.solar_harmonics),
        sidereal_h=int(args.sidereal_harmonics),
        yearly_h=int(args.yearly_harmonics),
    )
    y_all = build_trig_targets(tle_azel)

    log("Fitting 7d TensorFlow harmonic linear model...")
    weights = fit_tf_linear_harmonic_model(
        tf=tf,
        x_train=X_all[fit_mask],
        y_train=y_all[fit_mask],
        ridge=float(args.ridge),
    )
    pred_trig = np.asarray(X_all, dtype=np.float64) @ weights
    pred_ai_azel = decode_trig_targets(pred_trig)
    forecast_mask = window.unix >= float(window.train_end_unix)
    pred_ai_azel_full = tle_azel.copy()
    pred_ai_azel_full[forecast_mask] = pred_ai_azel[forecast_mask]

    pred_ai_lla = tle_lla.copy()
    pred_ai_lla_forecast = core.azel_to_lla_geoshell(
        az_deg=pred_ai_azel_full[forecast_mask, 0],
        el_deg=pred_ai_azel_full[forecast_mask, 1],
        observer_lat_deg=float(args.observer_lat),
        observer_lon_deg=float(args.observer_lon),
        observer_alt_m=float(args.observer_alt_m),
        geo_radius_km=float(args.geo_radius_km),
    )
    pred_ai_lla[forecast_mask] = pred_ai_lla_forecast

    ai_dir = out_root / "tf_7d_train_83d_forecast"
    ai_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ai_dir / "single_tle_7d83d_tf_linear_harmonic_model.npz",
        weights=weights,
        solar_harmonics=int(args.solar_harmonics),
        sidereal_harmonics=int(args.sidereal_harmonics),
        yearly_harmonics=int(args.yearly_harmonics),
        ridge=float(args.ridge),
    )

    train_teacher_metrics = core.compute_metrics_azel(
        np.column_stack([tle_lla[fit_mask], tle_azel[fit_mask]]),
        np.column_stack([tle_lla[fit_mask], pred_ai_azel[fit_mask]]),
    )
    val_teacher_metrics = core.compute_metrics_azel(
        np.column_stack([tle_lla[val_mask], tle_azel[val_mask]]),
        np.column_stack([tle_lla[val_mask], pred_ai_azel[val_mask]]),
    )
    hist_payload = {
        "fit_method": "tensorflow_lstsq_harmonic_linear",
        "feature_dim": int(X_all.shape[1]),
        "train_rows": int(np.sum(fit_mask)),
        "val_rows": int(np.sum(val_mask)),
        "train_teacher_metrics": train_teacher_metrics,
        "val_teacher_metrics": val_teacher_metrics,
    }
    (ai_dir / "history.json").write_text(json.dumps(hist_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ai_metrics = evaluate_and_save(
        name="tf_7d_train_83d_forecast",
        out_dir=ai_dir,
        dates=window.dates,
        pred_unix=window.unix,
        pred_lla=pred_ai_lla,
        pred_azel=pred_ai_azel_full,
        truth_unix=truth_unix,
        truth_full=truth_full,
        forecast_start_unix=window.train_end_unix,
        max_plot_points_per_month=int(args.max_plot_points_per_month),
        extra_json={
            "tle_file": str(tle_path),
            "train_days": int(args.train_days),
            "total_days": int(args.days),
            "feature_config": {
                "solar_harmonics": int(args.solar_harmonics),
                "sidereal_harmonics": int(args.sidereal_harmonics),
                "yearly_harmonics": int(args.yearly_harmonics),
                "ridge": float(args.ridge),
            },
            "fit_method": "tensorflow_lstsq_harmonic_linear",
            "teacher_fit_metrics": {
                "train": train_teacher_metrics["overall"],
                "val": val_teacher_metrics["overall"],
            },
        },
    )
    log(f"TF 7d->83d forecast max abs (83d only): {ai_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}")

    log("Fitting 7d TensorFlow unix-sincos-only MLP...")
    X_unix = build_unix_sincos_only_features(window.unix)
    unix_model, unix_hist = fit_tf_unix_sincos_only_mlp(
        tf=tf,
        x_train=X_unix[fit_mask],
        y_train=y_all[fit_mask],
        x_val=X_unix[val_mask],
        y_val=y_all[val_mask],
        seed=int(args.seed),
    )
    pred_unix_trig = np.asarray(unix_model.predict(X_unix, verbose=0), dtype=np.float64)
    pred_unix_azel = decode_trig_targets(pred_unix_trig)
    pred_unix_azel_full = tle_azel.copy()
    pred_unix_azel_full[forecast_mask] = pred_unix_azel[forecast_mask]
    pred_unix_lla = tle_lla.copy()
    pred_unix_lla[forecast_mask] = core.azel_to_lla_geoshell(
        az_deg=pred_unix_azel_full[forecast_mask, 0],
        el_deg=pred_unix_azel_full[forecast_mask, 1],
        observer_lat_deg=float(args.observer_lat),
        observer_lon_deg=float(args.observer_lon),
        observer_alt_m=float(args.observer_alt_m),
        geo_radius_km=float(args.geo_radius_km),
    )
    unix_dir = out_root / "tf_7d_train_83d_forecast_unix_sincos_only"
    unix_dir.mkdir(parents=True, exist_ok=True)
    unix_model.save(unix_dir / "single_tle_7d83d_tf_unix_sincos_only.keras")
    train_unix_teacher_metrics = core.compute_metrics_azel(
        np.column_stack([tle_lla[fit_mask], tle_azel[fit_mask]]),
        np.column_stack([tle_lla[fit_mask], pred_unix_azel[fit_mask]]),
    )
    val_unix_teacher_metrics = core.compute_metrics_azel(
        np.column_stack([tle_lla[val_mask], tle_azel[val_mask]]),
        np.column_stack([tle_lla[val_mask], pred_unix_azel[val_mask]]),
    )
    unix_history_payload = {
        "fit_method": "tensorflow_mlp_unix_sincos_only",
        "feature_dim": int(X_unix.shape[1]),
        "train_rows": int(np.sum(fit_mask)),
        "val_rows": int(np.sum(val_mask)),
        "train_teacher_metrics": train_unix_teacher_metrics,
        "val_teacher_metrics": val_unix_teacher_metrics,
        "history": unix_hist,
    }
    (unix_dir / "history.json").write_text(
        json.dumps(unix_history_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unix_metrics = evaluate_and_save(
        name="tf_7d_train_83d_forecast_unix_sincos_only",
        out_dir=unix_dir,
        dates=window.dates,
        pred_unix=window.unix,
        pred_lla=pred_unix_lla,
        pred_azel=pred_unix_azel_full,
        truth_unix=truth_unix,
        truth_full=truth_full,
        forecast_start_unix=window.train_end_unix,
        max_plot_points_per_month=int(args.max_plot_points_per_month),
        extra_json={
            "tle_file": str(tle_path),
            "train_days": int(args.train_days),
            "total_days": int(args.days),
            "fit_method": "tensorflow_mlp_unix_sincos_only",
            "feature_config": {
                "features": ["sin(unix)", "cos(unix)"],
            },
            "teacher_fit_metrics": {
                "train": train_unix_teacher_metrics["overall"],
                "val": val_unix_teacher_metrics["overall"],
            },
        },
    )
    log(
        "TF unix-sincos-only 7d->83d forecast max abs (83d only): "
        f"{unix_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
    )

    target_year = parse_target_year_from_truth_file(truth_path)
    truth_cache = load_truth_cache(max_year=target_year)
    target_dt = core.parse_tle_datetime_from_stem(tle_path.stem)
    if target_dt is None:
        raise ValueError(f"Cannot parse target datetime from TLE file: {tle_path.name}")
    validation_years = [y for y in parse_int_csv(args.historical_analog_validation_years) if y < target_year]
    k_candidates = [k for k in parse_int_csv(args.historical_analog_k_candidates) if k > 0]
    episodes_by_year: dict[int, list[dict]] = {}
    log("Building historical local analog episodes...")
    for year in range(2017, target_year):
        chosen = select_nearby_historical_tles(
            year=year,
            target_dt=target_dt,
            nearby_per_year=int(args.historical_analog_nearby_per_year),
            max_day_offset=int(args.historical_analog_max_day_offset),
            min_gap_days=int(args.historical_analog_min_gap_days),
        )
        built: list[dict] = []
        for hist_tle, hist_dt in chosen:
            ep = build_historical_analog_episode(
                tf=tf,
                tle_path=hist_tle,
                episode_dt=hist_dt,
                year=year,
                truth_cache=truth_cache,
                max_truth_year=target_year - 1,
                sat_name=str(args.sat_name),
                observer_lat=float(args.observer_lat),
                observer_lon=float(args.observer_lon),
                observer_alt_m=float(args.observer_alt_m),
                geo_radius_km=float(args.geo_radius_km),
                days=int(args.days),
                train_days=int(args.train_days),
                solar_harmonics=int(args.solar_harmonics),
                sidereal_harmonics=int(args.sidereal_harmonics),
                yearly_harmonics=int(args.yearly_harmonics),
                ridge=float(args.ridge),
            )
            if ep is not None:
                built.append(ep)
        if built:
            episodes_by_year[int(year)] = built
            log(f"Historical episodes {year}: {len(built)}")
    analog_dir = out_root / "tf_7d_train_83d_forecast_historical_analog"
    analog_dir.mkdir(parents=True, exist_ok=True)
    if episodes_by_year:
        stage3_metrics = None
        cv_result = choose_analog_k_forward_cv(
            episodes_by_year=episodes_by_year,
            k_candidates=k_candidates,
            validation_years=validation_years,
        )
        selected_k = int(cv_result["selected_k"])
        log(f"Historical analog selected k={selected_k}")
        target_episode = build_historical_analog_episode(
            tf=tf,
            tle_path=tle_path,
            episode_dt=target_dt,
            year=target_year,
            truth_cache=truth_cache,
            max_truth_year=target_year,
            sat_name=str(args.sat_name),
            observer_lat=float(args.observer_lat),
            observer_lon=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
            days=int(args.days),
            train_days=int(args.train_days),
            solar_harmonics=int(args.solar_harmonics),
            sidereal_harmonics=int(args.sidereal_harmonics),
            yearly_harmonics=int(args.yearly_harmonics),
            ridge=float(args.ridge),
        )
        if target_episode is None:
            raise RuntimeError("Failed to build target episode for historical analog correction")
        candidate_pool = [ep for year, items in episodes_by_year.items() if year < target_year for ep in items]
        corrected_aligned, correction_forecast, analog_info = apply_shifted_analog_correction(
            target_episode=target_episode,
            candidate_episodes=candidate_pool,
            k_neighbors=selected_k,
        )
        analog_azel = np.asarray(target_episode["pred_full_azel"], dtype=np.float64).copy()
        analog_lla = np.asarray(target_episode["pred_full_lla"], dtype=np.float64).copy()
        target_forecast_mask = target_episode["window"].unix >= float(target_episode["window"].train_end_unix)
        analog_azel[target_forecast_mask, 0] = core.wrap360(
            analog_azel[target_forecast_mask, 0] + correction_forecast[:, 0]
        )
        analog_azel[target_forecast_mask, 1] = analog_azel[target_forecast_mask, 1] + correction_forecast[:, 1]
        analog_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
            az_deg=analog_azel[target_forecast_mask, 0],
            el_deg=analog_azel[target_forecast_mask, 1],
            observer_lat_deg=float(args.observer_lat),
            observer_lon_deg=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
        )
        analog_meta = {
            "fit_method": "historical_local_shifted_residual_analog",
            "selected_k": selected_k,
            "k_candidates": k_candidates,
            "validation_years": validation_years,
            "cv_scores": cv_result["scores"],
            "used_episodes": analog_info["used_episodes"],
            "base_harmonic_forecast_metrics": ai_metrics["metrics_azel_forecast_83d"]["overall"],
            "corrected_forecast_metrics": analog_info["metrics_forecast"],
        }
        (analog_dir / "history.json").write_text(json.dumps(analog_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        analog_metrics = evaluate_and_save(
            name="tf_7d_train_83d_forecast_historical_analog",
            out_dir=analog_dir,
            dates=window.dates,
            pred_unix=window.unix,
            pred_lla=analog_lla,
            pred_azel=analog_azel,
            truth_unix=truth_unix,
            truth_full=truth_full,
            forecast_start_unix=window.train_end_unix,
            max_plot_points_per_month=int(args.max_plot_points_per_month),
            extra_json=analog_meta,
        )
        log(
            "Historical analog corrected max abs (83d only): "
            f"{analog_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
        )

        stage2_degree_candidates = [x for x in parse_int_csv(args.stage2_poly_degrees) if x > 0]
        stage2_validation_years = [y for y in validation_years if y >= 2018]
        stage2_entries_by_year: dict[int, list[dict]] = {}
        stage2_source_by_year: dict[int, list[dict]] = {}
        if stage2_degree_candidates and stage2_validation_years:
            log("Building stage2 global polynomial drift entries...")
            for year in range(2018, target_year):
                pool = [ep for yy, items in episodes_by_year.items() if yy < year for ep in items]
                targets = episodes_by_year.get(year, [])
                built_stage2: list[dict] = []
                built_stage2_source: list[dict] = []
                if pool and targets:
                    for ep in targets:
                        corrected_hist, _, _ = apply_shifted_analog_correction(
                            target_episode=ep,
                            candidate_episodes=pool,
                            k_neighbors=selected_k,
                        )
                        built_stage2.append(build_stage2_poly_entry(ep, corrected_hist))
                        built_stage2_source.append(
                            {
                                "episode": ep,
                                "corrected_aligned": np.asarray(corrected_hist, dtype=np.float64),
                            }
                        )
                if built_stage2:
                    stage2_entries_by_year[int(year)] = built_stage2
                    stage2_source_by_year[int(year)] = built_stage2_source
                    log(f"Stage2 entries {year}: {len(built_stage2)}")

            if stage2_entries_by_year:
                stage2_cv = choose_stage2_poly_degree_forward_cv(
                    entries_by_year=stage2_entries_by_year,
                    degree_candidates=stage2_degree_candidates,
                    validation_years=stage2_validation_years,
                    recent_year_count=int(args.stage2_recent_validation_years),
                    ridge=float(args.stage2_poly_ridge),
                )
                selected_stage2_degree = int(stage2_cv["selected_degree"])
                log(f"Stage2 selected polynomial degree={selected_stage2_degree}")
                stage2_train_entries = [
                    ep for year, items in stage2_entries_by_year.items() if int(year) < target_year for ep in items
                ]
                target_stage2_entry = build_stage2_poly_entry(target_episode, corrected_aligned)
                stage2_coeffs = fit_stage2_global_poly_coefficients(
                    train_entries=stage2_train_entries,
                    degree=selected_stage2_degree,
                    ridge=float(args.stage2_poly_ridge),
                )
                target_pred_stage2, stage2_metrics_forecast = apply_stage2_global_poly_to_entry(
                    entry=target_stage2_entry,
                    degree=selected_stage2_degree,
                    coeffs=stage2_coeffs,
                )

                stage2_azel = analog_azel.copy()
                stage2_lla = analog_lla.copy()
                az_corr, el_corr = build_stage2_poly_correction_series(
                    count=int(np.sum(target_forecast_mask)),
                    degree=selected_stage2_degree,
                    coeffs=stage2_coeffs,
                )
                stage2_azel[target_forecast_mask, 0] = core.wrap360(stage2_azel[target_forecast_mask, 0] - az_corr)
                stage2_azel[target_forecast_mask, 1] = stage2_azel[target_forecast_mask, 1] - el_corr
                stage2_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                    az_deg=stage2_azel[target_forecast_mask, 0],
                    el_deg=stage2_azel[target_forecast_mask, 1],
                    observer_lat_deg=float(args.observer_lat),
                    observer_lon_deg=float(args.observer_lon),
                    observer_alt_m=float(args.observer_alt_m),
                    geo_radius_km=float(args.geo_radius_km),
                )
                stage2_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly"
                stage2_meta = {
                    **analog_meta,
                    "stage2_fit_method": "global_polynomial_drift_after_historical_analog",
                    "stage2_selected_degree": selected_stage2_degree,
                    "stage2_degree_candidates": stage2_degree_candidates,
                    "stage2_recent_validation_years": int(args.stage2_recent_validation_years),
                    "stage2_poly_ridge": float(args.stage2_poly_ridge),
                    "stage2_cv": stage2_cv,
                    "stage2_corrected_forecast_metrics": stage2_metrics_forecast,
                }
                stage2_metrics = evaluate_and_save(
                    name="tf_7d_train_83d_forecast_historical_analog_stage2_poly",
                    out_dir=stage2_dir,
                    dates=window.dates,
                    pred_unix=window.unix,
                    pred_lla=stage2_lla,
                    pred_azel=stage2_azel,
                    truth_unix=truth_unix,
                    truth_full=truth_full,
                    forecast_start_unix=window.train_end_unix,
                    max_plot_points_per_month=int(args.max_plot_points_per_month),
                    extra_json=stage2_meta,
                )
                log(
                    "Historical analog + stage2 poly max abs (83d only): "
                    f"{stage2_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                )

                stage3_items_by_year: dict[int, list[dict]] = {}
                for year, items in stage2_source_by_year.items():
                    train_entries = [
                        ep for yy, stage2_items in stage2_entries_by_year.items() if int(yy) < int(year) for ep in stage2_items
                    ]
                    if not train_entries:
                        continue
                    hist_coeffs = fit_stage2_global_poly_coefficients(
                        train_entries=train_entries,
                        degree=selected_stage2_degree,
                        ridge=float(args.stage2_poly_ridge),
                    )
                    built_stage3: list[dict] = []
                    for item in items:
                        hist_entry = build_stage2_poly_entry(item["episode"], item["corrected_aligned"])
                        hist_pred_stage2, _ = apply_stage2_global_poly_to_entry(
                            entry=hist_entry,
                            degree=selected_stage2_degree,
                            coeffs=hist_coeffs,
                        )
                        built_stage3.append(
                            build_stage3_overlay_item(
                                episode=item["episode"],
                                pred_stage2_aligned=hist_pred_stage2,
                            )
                        )
                    if built_stage3:
                        stage3_items_by_year[int(year)] = built_stage3

                stage3_validation_years = [y for y in validation_years if y >= 2022]
                if stage3_items_by_year and stage3_validation_years:
                    stage3_cv = choose_stage3_overlay_family_forward_cv(
                        items_by_year=stage3_items_by_year,
                        validation_years=stage3_validation_years,
                    )
                    log(
                        "Stage3 selected family/recent/bounds/alpha="
                        f"{stage3_cv['selected_family']}/"
                        f"{stage3_cv['selected_recent_years']}/"
                        f"{tuple(stage3_cv['selected_bounds_days'])}/"
                        f"{tuple(stage3_cv['selected_alpha_segments'])}"
                    )
                    stage3_train_items = [
                        it for yy, items in stage3_items_by_year.items() if int(yy) < target_year for it in items
                    ]
                    target_stage3_item = build_stage3_overlay_item(
                        episode=target_episode,
                        pred_stage2_aligned=target_pred_stage2,
                    )
                    stage3_overlay = build_stage3_overlay_from_cv(
                        train_items=stage3_train_items,
                        target_episode=target_episode,
                        stage3_cv=stage3_cv,
                    )
                    stage3_pred_aligned, stage3_metrics_forecast = apply_stage3_piecewise_overlay_to_entry(
                        item=target_stage3_item,
                        overlay_forecast=stage3_overlay,
                        bounds_days=tuple(int(x) for x in stage3_cv["selected_bounds_days"]),
                        alpha_segments=tuple(float(x) for x in stage3_cv["selected_alpha_segments"]),
                    )
                    stage3_azel = stage2_azel.copy()
                    stage3_lla = stage2_lla.copy()
                    stage3_azel[target_forecast_mask, 0] = stage3_pred_aligned[target_episode["forecast_mask"], 3]
                    stage3_azel[target_forecast_mask, 1] = stage3_pred_aligned[target_episode["forecast_mask"], 4]
                    stage3_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                        az_deg=stage3_azel[target_forecast_mask, 0],
                        el_deg=stage3_azel[target_forecast_mask, 1],
                        observer_lat_deg=float(args.observer_lat),
                        observer_lon_deg=float(args.observer_lon),
                        observer_alt_m=float(args.observer_alt_m),
                        geo_radius_km=float(args.geo_radius_km),
                    )
                    stage3_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay"
                    stage3_meta = {
                        **stage2_meta,
                        "stage3_fit_method": "recent_year_piecewise_overlay_after_stage2",
                        "stage3_cv": stage3_cv,
                        "stage3_recent_overlay_forecast_metrics": stage3_metrics_forecast,
                    }
                    stage3_metrics = evaluate_and_save(
                        name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay",
                        out_dir=stage3_dir,
                        dates=window.dates,
                        pred_unix=window.unix,
                        pred_lla=stage3_lla,
                        pred_azel=stage3_azel,
                        truth_unix=truth_unix,
                        truth_full=truth_full,
                        forecast_start_unix=window.train_end_unix,
                        max_plot_points_per_month=int(args.max_plot_points_per_month),
                        extra_json=stage3_meta,
                    )
                    log(
                        "Historical analog + stage2 poly + stage3 overlay max abs (83d only): "
                        f"{stage3_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                    )

                    stage4_items_by_year: dict[int, list[dict]] = {}
                    for year in range(2020, target_year):
                        train = [it for yy, items in stage3_items_by_year.items() if int(yy) < int(year) for it in items]
                        targets = stage3_items_by_year.get(int(year), [])
                        if not train or not targets:
                            continue
                        built_stage4: list[dict] = []
                        for target in targets:
                            hist_overlay = build_stage3_overlay_from_cv(
                                train_items=train,
                                target_episode=target["episode"],
                                stage3_cv=stage3_cv,
                            )
                            hist_pred_stage3, _ = apply_stage3_piecewise_overlay_to_entry(
                                item=target,
                                overlay_forecast=hist_overlay,
                                bounds_days=tuple(int(x) for x in stage3_cv["selected_bounds_days"]),
                                alpha_segments=tuple(float(x) for x in stage3_cv["selected_alpha_segments"]),
                            )
                            built_stage4.append(
                                {
                                    "episode": target["episode"],
                                    "pred_stage3_aligned": hist_pred_stage3,
                                }
                            )
                        if built_stage4:
                            stage4_items_by_year[int(year)] = built_stage4

                    stage4_validation_years = [y for y in validation_years if y >= 2022]
                    if stage4_items_by_year and stage4_validation_years:
                        stage4_cv = choose_stage4_az_overlay_forward_cv(
                            items_by_year=stage4_items_by_year,
                            stage3_items_by_year=stage3_items_by_year,
                            validation_years=stage4_validation_years,
                        )
                        log(
                            "Stage4 selected family/start/alpha="
                            f"{stage4_cv['selected_family']}/"
                            f"{stage4_cv['selected_start_day']}/"
                            f"{stage4_cv['selected_alpha_az']}"
                        )
                        if str(stage4_cv["selected_family"]) != "none":
                            stage4_overlay = build_stage4_local_overlay(
                                train_items=stage3_train_items,
                                target_episode=target_episode,
                                recent_years=int(stage4_cv["selected_recent_years"]),
                                local_block_minutes=int(stage4_cv["selected_local_block_minutes"]),
                                local_k_neighbors=int(stage4_cv["selected_local_k_neighbors"]),
                                blend_beta=float(stage4_cv["selected_blend_beta"]),
                            )
                            stage4_pred_aligned, stage4_metrics_forecast = apply_stage4_az_overlay_to_entry(
                                pred_stage3_aligned=stage3_pred_aligned,
                                episode=target_episode,
                                overlay_forecast=stage4_overlay,
                                start_day=int(stage4_cv["selected_start_day"]),
                                alpha_az=float(stage4_cv["selected_alpha_az"]),
                            )
                        else:
                            stage4_pred_aligned = np.asarray(stage3_pred_aligned, dtype=np.float64)
                            stage4_metrics_forecast = dict(stage3_metrics_forecast)

                        stage4_azel = stage3_azel.copy()
                        stage4_lla = stage3_lla.copy()
                        stage4_azel[target_forecast_mask, 0] = stage4_pred_aligned[target_episode["forecast_mask"], 3]
                        stage4_azel[target_forecast_mask, 1] = stage4_pred_aligned[target_episode["forecast_mask"], 4]
                        stage4_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                            az_deg=stage4_azel[target_forecast_mask, 0],
                            el_deg=stage4_azel[target_forecast_mask, 1],
                            observer_lat_deg=float(args.observer_lat),
                            observer_lon_deg=float(args.observer_lon),
                            observer_alt_m=float(args.observer_alt_m),
                            geo_radius_km=float(args.geo_radius_km),
                        )
                        stage4_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_az"
                        stage4_meta = {
                            **stage3_meta,
                            "stage4_fit_method": "recent_weighted_az_late_overlay_after_stage3",
                            "stage4_cv": stage4_cv,
                            "stage4_forecast_metrics": stage4_metrics_forecast,
                        }
                        stage4_metrics = evaluate_and_save(
                            name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_az",
                            out_dir=stage4_dir,
                            dates=window.dates,
                            pred_unix=window.unix,
                            pred_lla=stage4_lla,
                            pred_azel=stage4_azel,
                            truth_unix=truth_unix,
                            truth_full=truth_full,
                            forecast_start_unix=window.train_end_unix,
                            max_plot_points_per_month=int(args.max_plot_points_per_month),
                            extra_json=stage4_meta,
                        )
                        log(
                            "Historical analog + stage2 poly + stage3 overlay + stage4 AZ max abs (83d only): "
                            f"{stage4_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                        )
                    else:
                        stage4_metrics = None
                        log("Stage4 AZ overlay skipped: no stage4 items were built")
                else:
                    stage3_metrics = None
                    stage4_metrics = None
                    log("Stage3 overlay skipped: no stage3 items were built")
            else:
                stage2_metrics = None
                stage3_metrics = None
                stage4_metrics = None
                log("Stage2 poly skipped: no stage2 entries were built")
        else:
            stage2_metrics = None
            stage3_metrics = None
            stage4_metrics = None
            log("Stage2 poly skipped: no stage2 candidates or validation years")
    else:
        analog_metrics = None
        stage2_metrics = None
        stage3_metrics = None
        stage4_metrics = None
        log("Historical analog correction skipped: no historical episodes were built")

    comparison = {
        "tle_file": str(tle_path),
        "window": {
            "start_jst": window.dates[0],
            "end_jst": window.dates[-1],
            "rows": len(window.unix),
            "train_days": int(args.train_days),
            "forecast_days": int(args.days) - int(args.train_days),
        },
        "baseline_single_tle_90d": baseline_metrics['metrics_azel_forecast_83d']['overall'],
        "tf_7d_train_83d_forecast": ai_metrics['metrics_azel_forecast_83d']['overall'],
        "tf_7d_train_83d_forecast_unix_sincos_only": unix_metrics['metrics_azel_forecast_83d']['overall'],
    }
    if analog_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog"] = analog_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage2_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly"] = stage2_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage3_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay"] = stage3_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage4_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_az"] = stage4_metrics["metrics_azel_forecast_83d"]["overall"]
    (out_root / "summary.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
