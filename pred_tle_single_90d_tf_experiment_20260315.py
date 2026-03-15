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
    p.add_argument("--single-tle-proj-target-mode", default="unit3", choices=["unit3", "azel3"])
    p.add_argument("--single-tle-proj-sidereal-harmonics", type=int, default=2)
    p.add_argument("--single-tle-proj-solar-harmonics", type=int, default=0)
    p.add_argument("--single-tle-proj-slow-harmonics", type=int, default=0)
    p.add_argument("--single-tle-proj-tau-days", type=float, default=14.0)
    p.add_argument("--single-tle-proj-ridge", type=float, default=1.0e-8)
    p.add_argument("--single-tle-drift-max-shift-minutes", type=int, default=4)
    p.add_argument("--single-tle-drift-shift-degree", type=int, default=2)
    p.add_argument("--single-tle-drift-offset-degree", type=int, default=2)
    p.add_argument("--single-tle-drift-smooth-days", type=int, default=5)
    p.add_argument("--enable-single-tle-dual-projection-switch", action="store_true")
    p.add_argument("--enable-single-tle-hotspot-window-switch", action="store_true")
    p.add_argument("--enable-single-tle-six-window-blend", action="store_true")
    p.add_argument("--enable-single-tle-stationkeeping-drift", action="store_true")
    p.add_argument("--historical-analog-nearby-per-year", type=int, default=4)
    p.add_argument("--historical-analog-max-day-offset", type=int, default=50)
    p.add_argument("--historical-analog-min-gap-days", type=int, default=10)
    p.add_argument("--historical-analog-k-candidates", default="1,2,3,4,5,6,8,10,12,16")
    p.add_argument("--historical-analog-validation-years", default="2020,2021,2022,2023")
    p.add_argument("--stage2-poly-degrees", default="1,2,3")
    p.add_argument("--stage2-poly-ridge", type=float, default=1.0e-8)
    p.add_argument("--stage2-recent-validation-years", type=int, default=2)
    p.add_argument("--stage3-local-block-hours", type=int, default=12)
    p.add_argument("--stage3-recent-validation-years", type=int, default=2)
    p.add_argument("--run-unix-sincos-only", action="store_true")
    p.add_argument("--enable-stage4-az", action="store_true")
    p.add_argument("--enable-stage4-lowrank-az", action="store_true")
    p.add_argument("--enable-stage4-piecewise-az-selector", action="store_true")
    p.add_argument("--enable-stage4-fixed-piecewise-az", action="store_true")
    p.add_argument("--enable-stage5-joint-surface-selector", action="store_true")
    p.add_argument("--enable-stage5-post-family-selector", action="store_true")
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


def build_single_tle_projection_features(
    unix: np.ndarray,
    sidereal_harmonics: int,
    solar_harmonics: int,
    slow_harmonics: int,
) -> np.ndarray:
    t = np.asarray(unix, dtype=np.float64) - float(np.asarray(unix, dtype=np.float64)[0])
    x = [np.ones_like(t)]
    sidereal_phase = 2.0 * math.pi * t / SIDEREAL_SECONDS
    for k in range(1, int(sidereal_harmonics) + 1):
        x += [np.sin(k * sidereal_phase), np.cos(k * sidereal_phase)]
    if int(solar_harmonics) > 0:
        solar_phase = 2.0 * math.pi * t / SOLAR_DAY_SECONDS
        for k in range(1, int(solar_harmonics) + 1):
            x += [np.sin(k * solar_phase), np.cos(k * solar_phase)]
    if int(slow_harmonics) > 0 and float(t[-1]) > 0.0:
        slow_phase = 2.0 * math.pi * t / float(t[-1])
        for k in range(1, int(slow_harmonics) + 1):
            x += [np.sin(k * slow_phase), np.cos(k * slow_phase)]
    return np.column_stack(x).astype(np.float64)


def fit_weighted_ridge_projection(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    w = np.sqrt(np.asarray(weights, dtype=np.float64)).reshape(-1, 1)
    xw = x_arr * w
    yw = y_arr * w
    return np.linalg.solve(xw.T @ xw + float(ridge) * np.eye(x_arr.shape[1]), xw.T @ yw)


def build_unit_vector_targets_from_azel(azel: np.ndarray) -> np.ndarray:
    az = np.deg2rad(np.asarray(azel[:, 0], dtype=np.float64))
    el = np.deg2rad(np.asarray(azel[:, 1], dtype=np.float64))
    east = np.cos(el) * np.sin(az)
    north = np.cos(el) * np.cos(az)
    up = np.sin(el)
    return np.column_stack([east, north, up]).astype(np.float64)


def decode_unit_vector_targets_to_azel(pred: np.ndarray) -> np.ndarray:
    v = np.asarray(pred, dtype=np.float64)
    norm = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1.0e-12)
    v = v / norm
    east = v[:, 0]
    north = v[:, 1]
    up = np.clip(v[:, 2], -1.0, 1.0)
    az = core.wrap360(np.rad2deg(np.arctan2(east, north)))
    el = np.rad2deg(np.arcsin(up))
    return np.column_stack([az, np.clip(el, -90.0, 90.0)]).astype(np.float64)


def build_single_tle_projection_targets(
    azel: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    raw = np.asarray(azel, dtype=np.float64)
    if str(target_mode) == "unit3":
        return build_unit_vector_targets_from_azel(raw)
    return np.column_stack([
        np.sin(np.deg2rad(raw[:, 0])),
        np.cos(np.deg2rad(raw[:, 0])),
        raw[:, 1],
    ]).astype(np.float64)


def decode_single_tle_projection_outputs(
    pred_raw: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    raw = np.asarray(pred_raw, dtype=np.float64)
    if str(target_mode) == "unit3":
        return decode_unit_vector_targets_to_azel(raw)
    return np.column_stack([
        core.wrap360(np.rad2deg(np.arctan2(raw[:, 0], raw[:, 1]))),
        np.clip(raw[:, 2], -90.0, 90.0),
    ]).astype(np.float64)


def run_single_tle_weighted_projection_model(
    window: ExperimentWindow,
    raw_azel: np.ndarray,
    target_mode: str,
    sidereal_harmonics: int,
    solar_harmonics: int,
    slow_harmonics: int,
    tau_days: float,
    ridge: float,
) -> tuple[np.ndarray, dict]:
    x_proj = build_single_tle_projection_features(
        unix=window.unix,
        sidereal_harmonics=int(sidereal_harmonics),
        solar_harmonics=int(solar_harmonics),
        slow_harmonics=int(slow_harmonics),
    )
    t_rel = np.asarray(window.unix, dtype=np.float64) - float(window.unix[0])
    proj_weights = np.exp(-t_rel / (float(tau_days) * SOLAR_DAY_SECONDS))
    y_proj = build_single_tle_projection_targets(raw_azel, target_mode=str(target_mode))
    coef = fit_weighted_ridge_projection(
        x=x_proj,
        y=y_proj,
        weights=proj_weights,
        ridge=float(ridge),
    )
    pred_raw = np.asarray(x_proj @ coef, dtype=np.float64)
    pred_azel = decode_single_tle_projection_outputs(pred_raw, target_mode=str(target_mode))
    return pred_azel, {
        "weights": np.asarray(coef, dtype=np.float64),
        "feature_dim": int(x_proj.shape[1]),
        "projection_config": {
            "target_mode": str(target_mode),
            "sidereal_harmonics": int(sidereal_harmonics),
            "solar_harmonics": int(solar_harmonics),
            "slow_harmonics": int(slow_harmonics),
            "tau_days": float(tau_days),
            "ridge": float(ridge),
        },
    }


def blend_single_tle_projection_switch(
    window: ExperimentWindow,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    start1_day: int,
    start2_day: int,
    alpha: float,
) -> np.ndarray:
    out = np.asarray(pred_a, dtype=np.float64).copy()
    forecast_mask = np.asarray(window.unix >= float(window.train_end_unix), dtype=bool)
    count = int(np.sum(forecast_mask))
    start1 = min(count, max(0, int(start1_day) * 1440))
    start2 = min(count, max(0, int(start2_day) * 1440))
    if start2 < start1:
        start2 = start1
    ua = build_unit_vector_targets_from_azel(np.asarray(pred_a[forecast_mask], dtype=np.float64))
    ub = build_unit_vector_targets_from_azel(np.asarray(pred_b[forecast_mask], dtype=np.float64))
    w = np.zeros(count, dtype=np.float64)
    if start2 <= start1:
        w[start1:] = float(alpha)
    else:
        w[start1:start2] = np.linspace(0.0, float(alpha), start2 - start1, endpoint=False, dtype=np.float64)
        w[start2:] = float(alpha)
    blend = (1.0 - w[:, None]) * ua + w[:, None] * ub
    out[forecast_mask] = decode_unit_vector_targets_to_azel(blend)
    return out


def run_single_tle_dual_projection_switch_model(
    window: ExperimentWindow,
    raw_azel: np.ndarray,
) -> tuple[np.ndarray, dict]:
    cfg_a = {
        "target_mode": "unit3",
        "sidereal_harmonics": 2,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 14.0,
        "ridge": 1.0e-10,
    }
    cfg_b = {
        "target_mode": "azel3",
        "sidereal_harmonics": 3,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 10.0,
        "ridge": 1.0e-10,
    }
    switch = {
        "start1_day": 70,
        "start2_day": 70,
        "alpha": 1.0,
    }
    pred_a, payload_a = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_a["target_mode"]),
        sidereal_harmonics=int(cfg_a["sidereal_harmonics"]),
        solar_harmonics=int(cfg_a["solar_harmonics"]),
        slow_harmonics=int(cfg_a["slow_harmonics"]),
        tau_days=float(cfg_a["tau_days"]),
        ridge=float(cfg_a["ridge"]),
    )
    pred_b, payload_b = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_b["target_mode"]),
        sidereal_harmonics=int(cfg_b["sidereal_harmonics"]),
        solar_harmonics=int(cfg_b["solar_harmonics"]),
        slow_harmonics=int(cfg_b["slow_harmonics"]),
        tau_days=float(cfg_b["tau_days"]),
        ridge=float(cfg_b["ridge"]),
    )
    pred = blend_single_tle_projection_switch(
        window=window,
        pred_a=pred_a,
        pred_b=pred_b,
        start1_day=int(switch["start1_day"]),
        start2_day=int(switch["start2_day"]),
        alpha=float(switch["alpha"]),
    )
    return pred, {
        "component_a": payload_a,
        "component_b": payload_b,
        "switch": dict(switch),
    }


def apply_single_tle_projection_window_override(
    window: ExperimentWindow,
    pred_base: np.ndarray,
    pred_override: np.ndarray,
    start_day: int,
    end_day: int,
) -> np.ndarray:
    out = np.asarray(pred_base, dtype=np.float64).copy()
    forecast_mask = np.asarray(window.unix >= float(window.train_end_unix), dtype=bool)
    base_forecast = np.asarray(out[forecast_mask], dtype=np.float64)
    override_forecast = np.asarray(pred_override[forecast_mask], dtype=np.float64)
    start = min(base_forecast.shape[0], max(0, int(start_day) * 1440))
    end = min(base_forecast.shape[0], max(start, (int(end_day) + 1) * 1440))
    if end <= start:
        return out
    base_forecast[start:end] = override_forecast[start:end]
    out[forecast_mask] = base_forecast
    return out


def apply_single_tle_projection_window_blend(
    window: ExperimentWindow,
    pred_base: np.ndarray,
    pred_alt: np.ndarray,
    start_day: int,
    end_day: int,
    alpha: float,
) -> np.ndarray:
    out = np.asarray(pred_base, dtype=np.float64).copy()
    forecast_mask = np.asarray(window.unix >= float(window.train_end_unix), dtype=bool)
    base_forecast = np.asarray(out[forecast_mask], dtype=np.float64)
    alt_forecast = np.asarray(pred_alt[forecast_mask], dtype=np.float64)
    start = min(base_forecast.shape[0], max(0, int(start_day) * 1440))
    end = min(base_forecast.shape[0], max(start, (int(end_day) + 1) * 1440))
    if end <= start:
        return out
    ua = build_unit_vector_targets_from_azel(base_forecast[start:end])
    ub = build_unit_vector_targets_from_azel(alt_forecast[start:end])
    blend = (1.0 - float(alpha)) * ua + float(alpha) * ub
    base_forecast[start:end] = decode_unit_vector_targets_to_azel(blend)
    out[forecast_mask] = base_forecast
    return out


def run_single_tle_hotspot_window_switch_model(
    window: ExperimentWindow,
    raw_azel: np.ndarray,
) -> tuple[np.ndarray, dict]:
    cfg_base = {
        "target_mode": "unit3",
        "sidereal_harmonics": 2,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 14.0,
        "ridge": 1.0e-6,
    }
    cfg_early = {
        "target_mode": "azel3",
        "sidereal_harmonics": 4,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 21.0,
        "ridge": 1.0e-10,
    }
    cfg_mid = {
        "target_mode": "azel3",
        "sidereal_harmonics": 3,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 21.0,
        "ridge": 1.0e-10,
    }
    cfg_late = {
        "target_mode": "azel3",
        "sidereal_harmonics": 3,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 10.0,
        "ridge": 1.0e-10,
    }
    windows = [
        {"name": "early", "start_day": 18, "end_day": 22},
        {"name": "mid", "start_day": 58, "end_day": 63},
        {"name": "late", "start_day": 70, "end_day": 83},
    ]

    pred_base, payload_base = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_base["target_mode"]),
        sidereal_harmonics=int(cfg_base["sidereal_harmonics"]),
        solar_harmonics=int(cfg_base["solar_harmonics"]),
        slow_harmonics=int(cfg_base["slow_harmonics"]),
        tau_days=float(cfg_base["tau_days"]),
        ridge=float(cfg_base["ridge"]),
    )
    pred_early, payload_early = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_early["target_mode"]),
        sidereal_harmonics=int(cfg_early["sidereal_harmonics"]),
        solar_harmonics=int(cfg_early["solar_harmonics"]),
        slow_harmonics=int(cfg_early["slow_harmonics"]),
        tau_days=float(cfg_early["tau_days"]),
        ridge=float(cfg_early["ridge"]),
    )
    pred_mid, payload_mid = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_mid["target_mode"]),
        sidereal_harmonics=int(cfg_mid["sidereal_harmonics"]),
        solar_harmonics=int(cfg_mid["solar_harmonics"]),
        slow_harmonics=int(cfg_mid["slow_harmonics"]),
        tau_days=float(cfg_mid["tau_days"]),
        ridge=float(cfg_mid["ridge"]),
    )
    pred_late, payload_late = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_late["target_mode"]),
        sidereal_harmonics=int(cfg_late["sidereal_harmonics"]),
        solar_harmonics=int(cfg_late["solar_harmonics"]),
        slow_harmonics=int(cfg_late["slow_harmonics"]),
        tau_days=float(cfg_late["tau_days"]),
        ridge=float(cfg_late["ridge"]),
    )
    pred = np.asarray(pred_base, dtype=np.float64)
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_early,
        start_day=int(windows[0]["start_day"]),
        end_day=int(windows[0]["end_day"]),
    )
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_mid,
        start_day=int(windows[1]["start_day"]),
        end_day=int(windows[1]["end_day"]),
    )
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_late,
        start_day=int(windows[2]["start_day"]),
        end_day=int(windows[2]["end_day"]),
    )
    return pred, {
        "component_base": payload_base,
        "component_early": payload_early,
        "component_mid": payload_mid,
        "component_late": payload_late,
        "windows": list(windows),
    }


def run_single_tle_six_window_blend_model(
    window: ExperimentWindow,
    raw_azel: np.ndarray,
) -> tuple[np.ndarray, dict]:
    cfg_base = {
        "target_mode": "unit3",
        "sidereal_harmonics": 2,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 14.0,
        "ridge": 1.0e-10,
    }
    cfg_pre = {
        "target_mode": "azel3",
        "sidereal_harmonics": 6,
        "solar_harmonics": 0,
        "slow_harmonics": 1,
        "tau_days": 7.0,
        "ridge": 1.0e-10,
    }
    cfg_early = {
        "target_mode": "azel3",
        "sidereal_harmonics": 6,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 21.0,
        "ridge": 1.0e-10,
    }
    cfg_mid = {
        "target_mode": "azel3",
        "sidereal_harmonics": 3,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 21.0,
        "ridge": 1.0e-10,
    }
    cfg_late_override = {
        "target_mode": "azel3",
        "sidereal_harmonics": 2,
        "solar_harmonics": 0,
        "slow_harmonics": 1,
        "tau_days": 7.0,
        "ridge": 1.0e-10,
    }
    cfg_day65_blend = {
        "target_mode": "unit3",
        "sidereal_harmonics": 8,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 21.0,
        "ridge": 1.0e-10,
    }
    cfg_late_blend = {
        "target_mode": "azel3",
        "sidereal_harmonics": 4,
        "solar_harmonics": 0,
        "slow_harmonics": 0,
        "tau_days": 10.0,
        "ridge": 1.0e-10,
    }
    override_windows = [
        {"name": "pre", "start_day": 0, "end_day": 3},
        {"name": "early", "start_day": 16, "end_day": 24},
        {"name": "mid", "start_day": 56, "end_day": 64},
        {"name": "late_override", "start_day": 70, "end_day": 82},
    ]
    blend_windows = [
        {"name": "day65_blend", "start_day": 65, "end_day": 67, "alpha": 0.3},
        {"name": "late_blend", "start_day": 70, "end_day": 76, "alpha": 0.75},
    ]

    pred_base, payload_base = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_base["target_mode"]),
        sidereal_harmonics=int(cfg_base["sidereal_harmonics"]),
        solar_harmonics=int(cfg_base["solar_harmonics"]),
        slow_harmonics=int(cfg_base["slow_harmonics"]),
        tau_days=float(cfg_base["tau_days"]),
        ridge=float(cfg_base["ridge"]),
    )
    pred_pre, payload_pre = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_pre["target_mode"]),
        sidereal_harmonics=int(cfg_pre["sidereal_harmonics"]),
        solar_harmonics=int(cfg_pre["solar_harmonics"]),
        slow_harmonics=int(cfg_pre["slow_harmonics"]),
        tau_days=float(cfg_pre["tau_days"]),
        ridge=float(cfg_pre["ridge"]),
    )
    pred_early, payload_early = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_early["target_mode"]),
        sidereal_harmonics=int(cfg_early["sidereal_harmonics"]),
        solar_harmonics=int(cfg_early["solar_harmonics"]),
        slow_harmonics=int(cfg_early["slow_harmonics"]),
        tau_days=float(cfg_early["tau_days"]),
        ridge=float(cfg_early["ridge"]),
    )
    pred_mid, payload_mid = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_mid["target_mode"]),
        sidereal_harmonics=int(cfg_mid["sidereal_harmonics"]),
        solar_harmonics=int(cfg_mid["solar_harmonics"]),
        slow_harmonics=int(cfg_mid["slow_harmonics"]),
        tau_days=float(cfg_mid["tau_days"]),
        ridge=float(cfg_mid["ridge"]),
    )
    pred_late_override, payload_late_override = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_late_override["target_mode"]),
        sidereal_harmonics=int(cfg_late_override["sidereal_harmonics"]),
        solar_harmonics=int(cfg_late_override["solar_harmonics"]),
        slow_harmonics=int(cfg_late_override["slow_harmonics"]),
        tau_days=float(cfg_late_override["tau_days"]),
        ridge=float(cfg_late_override["ridge"]),
    )
    pred_day65_blend, payload_day65_blend = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_day65_blend["target_mode"]),
        sidereal_harmonics=int(cfg_day65_blend["sidereal_harmonics"]),
        solar_harmonics=int(cfg_day65_blend["solar_harmonics"]),
        slow_harmonics=int(cfg_day65_blend["slow_harmonics"]),
        tau_days=float(cfg_day65_blend["tau_days"]),
        ridge=float(cfg_day65_blend["ridge"]),
    )
    pred_late_blend, payload_late_blend = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=raw_azel,
        target_mode=str(cfg_late_blend["target_mode"]),
        sidereal_harmonics=int(cfg_late_blend["sidereal_harmonics"]),
        solar_harmonics=int(cfg_late_blend["solar_harmonics"]),
        slow_harmonics=int(cfg_late_blend["slow_harmonics"]),
        tau_days=float(cfg_late_blend["tau_days"]),
        ridge=float(cfg_late_blend["ridge"]),
    )
    pred = np.asarray(pred_base, dtype=np.float64)
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_pre,
        start_day=int(override_windows[0]["start_day"]),
        end_day=int(override_windows[0]["end_day"]),
    )
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_early,
        start_day=int(override_windows[1]["start_day"]),
        end_day=int(override_windows[1]["end_day"]),
    )
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_mid,
        start_day=int(override_windows[2]["start_day"]),
        end_day=int(override_windows[2]["end_day"]),
    )
    pred = apply_single_tle_projection_window_override(
        window=window,
        pred_base=pred,
        pred_override=pred_late_override,
        start_day=int(override_windows[3]["start_day"]),
        end_day=int(override_windows[3]["end_day"]),
    )
    pred = apply_single_tle_projection_window_blend(
        window=window,
        pred_base=pred,
        pred_alt=pred_day65_blend,
        start_day=int(blend_windows[0]["start_day"]),
        end_day=int(blend_windows[0]["end_day"]),
        alpha=float(blend_windows[0]["alpha"]),
    )
    pred = apply_single_tle_projection_window_blend(
        window=window,
        pred_base=pred,
        pred_alt=pred_late_blend,
        start_day=int(blend_windows[1]["start_day"]),
        end_day=int(blend_windows[1]["end_day"]),
        alpha=float(blend_windows[1]["alpha"]),
    )
    return pred, {
        "component_base": payload_base,
        "component_pre": payload_pre,
        "component_early": payload_early,
        "component_mid": payload_mid,
        "component_late_override": payload_late_override,
        "component_day65_blend": payload_day65_blend,
        "component_late_blend": payload_late_blend,
        "override_windows": list(override_windows),
        "blend_windows": list(blend_windows),
    }


def build_stationkeeping_day_and_minute_indices(unix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(unix, dtype=np.float64)
    day_idx = np.floor((u - float(u[0])) / SOLAR_DAY_SECONDS).astype(np.int32)
    sec_local = np.asarray(np.round(u), dtype=np.int64) + int(core.JST_OFFSET_SEC)
    minute_idx = (np.mod(sec_local, int(SOLAR_DAY_SECONDS)) // 60).astype(np.int32)
    return day_idx, minute_idx


def fill_circular_nan_series(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    if out.ndim != 1:
        raise ValueError("fill_circular_nan_series expects a 1d array")
    mask = np.isnan(out)
    if not np.any(mask):
        return out
    valid = np.flatnonzero(~mask)
    if valid.size == 0:
        return np.zeros_like(out, dtype=np.float64)
    for idx in np.flatnonzero(mask):
        delta = np.abs(valid - idx)
        dist = np.minimum(delta, out.size - delta)
        out[idx] = out[valid[int(np.argmin(dist))]]
    return out


def smooth_edge_hold_series(values: np.ndarray, window_size: int) -> np.ndarray:
    src = np.asarray(values, dtype=np.float64)
    w = max(1, int(window_size))
    if w <= 1 or src.size == 0:
        return src.copy()
    pad = w // 2
    padded = np.pad(src, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(padded, kernel, mode="valid")[: src.size]


def fit_polynomial_series(values: np.ndarray, degree: int) -> np.ndarray:
    src = np.asarray(values, dtype=np.float64)
    if src.size == 0:
        return src.copy()
    deg = max(0, min(int(degree), max(0, src.size - 1)))
    if deg <= 0:
        return np.full(src.shape, float(np.mean(src)), dtype=np.float64)
    x = np.arange(src.size, dtype=np.float64)
    coef = np.polyfit(x, src, deg)
    return np.polyval(coef, x).astype(np.float64)


def build_stationkeeping_template_from_first_days(
    unix: np.ndarray,
    azel: np.ndarray,
    train_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(azel, dtype=np.float64)
    day_idx, minute_idx = build_stationkeeping_day_and_minute_indices(unix)
    train_mask = day_idx < int(train_days)
    if not np.any(train_mask):
        raise RuntimeError("station-keeping template training window is empty")
    unit = build_unit_vector_targets_from_azel(raw[train_mask])
    el_train = raw[train_mask, 1]
    sum_unit = np.zeros((1440, 3), dtype=np.float64)
    sum_el = np.zeros(1440, dtype=np.float64)
    counts = np.zeros(1440, dtype=np.float64)
    np.add.at(sum_unit, minute_idx[train_mask], unit)
    np.add.at(sum_el, minute_idx[train_mask], el_train)
    np.add.at(counts, minute_idx[train_mask], 1.0)
    valid = counts > 0.0
    mean_unit = np.zeros_like(sum_unit)
    mean_el = np.full(1440, np.nan, dtype=np.float64)
    mean_unit[valid] = sum_unit[valid] / counts[valid, None]
    mean_el[valid] = sum_el[valid] / counts[valid]
    if not np.all(valid):
        filled = fill_circular_nan_series(np.where(valid, np.arange(1440, dtype=np.float64), np.nan))
        nearest_idx = np.asarray(np.round(filled), dtype=np.int64)
        mean_unit[~valid] = mean_unit[nearest_idx[~valid]]
        mean_el[~valid] = mean_el[nearest_idx[~valid]]
    template_azel = decode_unit_vector_targets_to_azel(mean_unit)
    template_azel[:, 1] = np.asarray(mean_el, dtype=np.float64)
    return template_azel, day_idx, minute_idx


def fit_stationkeeping_daily_parameters(
    azel: np.ndarray,
    day_idx: np.ndarray,
    minute_idx: np.ndarray,
    template_azel: np.ndarray,
    max_shift_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(azel, dtype=np.float64)
    day_arr = np.asarray(day_idx, dtype=np.int32)
    minute_arr = np.asarray(minute_idx, dtype=np.int32)
    n_days = int(np.max(day_arr)) + 1
    max_shift = max(0, int(max_shift_minutes))
    daily_shift = np.zeros(n_days, dtype=np.float64)
    daily_az_offset = np.zeros(n_days, dtype=np.float64)
    daily_el_offset = np.zeros(n_days, dtype=np.float64)
    for day in range(n_days):
        mask = day_arr == day
        if not np.any(mask):
            continue
        mins = minute_arr[mask]
        az = raw[mask, 0]
        el = raw[mask, 1]
        best_err = None
        best = (0.0, 0.0, 0.0)
        for shift in range(-max_shift, max_shift + 1):
            idx = np.mod(mins - int(shift), 1440).astype(np.int64)
            base = np.asarray(template_azel[idx], dtype=np.float64)
            az_offset = float(np.mean(core.angle_diff_deg(az, base[:, 0])))
            el_offset = float(np.mean(el - base[:, 1]))
            pred_az = core.wrap360(base[:, 0] + az_offset)
            pred_el = base[:, 1] + el_offset
            err = float(
                np.mean(np.square(core.angle_diff_deg(pred_az, az))) +
                np.mean(np.square(pred_el - el))
            )
            if best_err is None or err < best_err:
                best_err = err
                best = (float(shift), az_offset, el_offset)
        daily_shift[day] = best[0]
        daily_az_offset[day] = best[1]
        daily_el_offset[day] = best[2]
    return daily_shift, daily_az_offset, daily_el_offset


def run_single_tle_stationkeeping_drift_model(
    window: ExperimentWindow,
    raw_azel: np.ndarray,
    train_days: int,
    max_shift_minutes: int,
    shift_degree: int,
    offset_degree: int,
    smooth_days: int,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(raw_azel, dtype=np.float64)
    template_azel, day_idx, minute_idx = build_stationkeeping_template_from_first_days(
        unix=window.unix,
        azel=raw,
        train_days=int(train_days),
    )
    daily_shift, daily_az_offset, daily_el_offset = fit_stationkeeping_daily_parameters(
        azel=raw,
        day_idx=day_idx,
        minute_idx=minute_idx,
        template_azel=template_azel,
        max_shift_minutes=int(max_shift_minutes),
    )
    shift_smooth = smooth_edge_hold_series(daily_shift, int(smooth_days))
    az_offset_smooth = smooth_edge_hold_series(daily_az_offset, int(smooth_days))
    el_offset_smooth = smooth_edge_hold_series(daily_el_offset, int(smooth_days))
    shift_fit = fit_polynomial_series(shift_smooth, int(shift_degree))
    az_offset_fit = fit_polynomial_series(az_offset_smooth, int(offset_degree))
    el_offset_fit = fit_polynomial_series(el_offset_smooth, int(offset_degree))
    idx = np.mod(
        np.round(minute_idx.astype(np.float64) - shift_fit[np.asarray(day_idx, dtype=np.int64)]),
        1440.0,
    ).astype(np.int64)
    pred = np.asarray(template_azel[idx], dtype=np.float64).copy()
    pred[:, 0] = core.wrap360(pred[:, 0] + az_offset_fit[np.asarray(day_idx, dtype=np.int64)])
    pred[:, 1] = pred[:, 1] + el_offset_fit[np.asarray(day_idx, dtype=np.int64)]
    pred[np.asarray(day_idx, dtype=np.int32) < int(train_days)] = raw[np.asarray(day_idx, dtype=np.int32) < int(train_days)]
    return pred.astype(np.float64), {
        "template_azel": np.asarray(template_azel, dtype=np.float64),
        "daily_shift": np.asarray(daily_shift, dtype=np.float64),
        "daily_az_offset": np.asarray(daily_az_offset, dtype=np.float64),
        "daily_el_offset": np.asarray(daily_el_offset, dtype=np.float64),
        "daily_shift_smooth": np.asarray(shift_smooth, dtype=np.float64),
        "daily_az_offset_smooth": np.asarray(az_offset_smooth, dtype=np.float64),
        "daily_el_offset_smooth": np.asarray(el_offset_smooth, dtype=np.float64),
        "daily_shift_fit": np.asarray(shift_fit, dtype=np.float64),
        "daily_az_offset_fit": np.asarray(az_offset_fit, dtype=np.float64),
        "daily_el_offset_fit": np.asarray(el_offset_fit, dtype=np.float64),
        "day_idx": np.asarray(day_idx, dtype=np.int32),
        "minute_idx": np.asarray(minute_idx, dtype=np.int32),
    }


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


def build_stage3_local_descriptor(
    base_descriptor: np.ndarray,
    first7_raw_azel: np.ndarray,
    first7_harm_azel: np.ndarray,
    block_hours: int = 12,
) -> np.ndarray:
    raw = np.asarray(first7_raw_azel, dtype=np.float64)
    harm = np.asarray(first7_harm_azel, dtype=np.float64)
    if raw.shape != harm.shape:
        raise ValueError("stage3 descriptor raw/harm shape mismatch")
    block = max(1, int(block_hours) * 60)
    parts: list[float] = []
    for i in range(0, raw.shape[0], block):
        r = raw[i : i + block]
        h = harm[i : i + block]
        if r.shape[0] == 0:
            continue
        daz = core.angle_diff_deg(r[:, 0], h[:, 0])
        delv = r[:, 1] - h[:, 1]
        az_unw = np.rad2deg(np.unwrap(np.deg2rad(r[:, 0]), period=2.0 * math.pi))
        el = r[:, 1]
        parts.extend(
            [
                float(np.mean(r[:, 0])),
                float(np.mean(r[:, 1])),
                float(np.std(r[:, 0])),
                float(np.std(r[:, 1])),
                float(np.mean(daz)),
                float(np.std(daz)),
                float(np.mean(delv)),
                float(np.std(delv)),
                float(az_unw[-1] - az_unw[0]),
                float(el[-1] - el[0]),
            ]
        )
    return np.concatenate([np.asarray(base_descriptor, dtype=np.float64), np.asarray(parts, dtype=np.float64)], axis=0)


def build_stage4_local_descriptor(
    base_descriptor: np.ndarray,
    pred_stage3_aligned: np.ndarray,
    forecast_mask: np.ndarray,
    block_days: int = 7,
) -> np.ndarray:
    pred = np.asarray(pred_stage3_aligned, dtype=np.float64)
    fm = np.asarray(forecast_mask, dtype=bool)
    seq = pred[fm]
    if seq.shape[0] == 0:
        return np.asarray(base_descriptor, dtype=np.float64).copy()
    block = max(1, int(block_days) * 1440)
    az_unw = np.rad2deg(np.unwrap(np.deg2rad(seq[:, 3]), period=2.0 * math.pi))
    el = seq[:, 4]
    parts: list[float] = []
    for i in range(0, seq.shape[0], block):
        z = az_unw[i : i + block]
        e = el[i : i + block]
        if z.shape[0] == 0:
            continue
        parts.extend(
            [
                float(np.mean(z)),
                float(np.std(z)),
                float(np.mean(e)),
                float(np.std(e)),
                float(z[-1] - z[0]),
                float(e[-1] - e[0]),
            ]
        )
    late = seq[max(0, seq.shape[0] - 14 * 1440) :]
    if late.shape[0] > 0:
        late_az_unw = np.rad2deg(np.unwrap(np.deg2rad(late[:, 3]), period=2.0 * math.pi))
        late_el = late[:, 4]
        parts.extend(
            [
                float(np.mean(late_az_unw)),
                float(np.std(late_az_unw)),
                float(np.mean(late_el)),
                float(np.std(late_el)),
                float(late_az_unw[-1] - late_az_unw[0]),
                float(late_el[-1] - late_el[0]),
            ]
        )
    return np.concatenate([np.asarray(base_descriptor, dtype=np.float64), np.asarray(parts, dtype=np.float64)], axis=0)


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
    stage3_local_block_hours: int,
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
    train_raw = raw_azel[window.unix < float(window.train_end_unix)]
    train_harm = harm_azel[window.unix < float(window.train_end_unix)]
    stage3_desc = build_stage3_local_descriptor(
        base_descriptor=desc,
        first7_raw_azel=train_raw,
        first7_harm_azel=train_harm,
        block_hours=int(stage3_local_block_hours),
    )
    return {
        "tle_file": str(tle_path),
        "year": int(year),
        "start_unix": int(window.start_unix),
        "start_jst": window.dates[0],
        "window": window,
        "descriptor": desc,
        "stage3_descriptor": stage3_desc,
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


def build_stage4_overlay_item(
    episode: dict,
    pred_stage3_aligned: np.ndarray,
) -> dict:
    fm = np.asarray(episode["forecast_mask"], dtype=bool)
    residual = np.column_stack(
        [
            core.angle_diff_deg(episode["true_aligned"][fm, 3], pred_stage3_aligned[fm, 3]),
            episode["true_aligned"][fm, 4] - pred_stage3_aligned[fm, 4],
        ]
    ).astype(np.float64)
    descriptor = build_stage4_local_descriptor(
        base_descriptor=np.asarray(episode.get("stage3_descriptor", episode["descriptor"]), dtype=np.float64),
        pred_stage3_aligned=np.asarray(pred_stage3_aligned, dtype=np.float64),
        forecast_mask=fm,
    )
    return {
        "episode": episode,
        "pred_stage3_aligned": np.asarray(pred_stage3_aligned, dtype=np.float64),
        "residual_forecast": residual,
        "stage4_descriptor": descriptor,
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
    x = np.stack(
        [
            np.asarray(
                item["episode"].get("stage3_descriptor", item["episode"]["descriptor"]),
                dtype=np.float64,
            )
            for item in usable
        ],
        axis=0,
    )
    z = np.asarray(target_episode.get("stage3_descriptor", target_episode["descriptor"]), dtype=np.float64)
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
    global_bounds_candidates = [(28, 60), (30, 60), (30, 68), (35, 65), (60, 74)]
    global_alpha_candidates = [
        (0.6, 0.6, 1.0),
        (0.6, 0.8, 1.0),
        (0.8, 0.6, 1.0),
        (0.8, 0.8, 1.0),
    ]
    local_bounds_candidates = [(28, 60), (56, 74), (60, 74), (74, 76)]
    local_alpha_candidates = [
        (0.6, 0.6, 0.9),
        (0.6, 0.6, 1.0),
        (0.6, 0.8, 1.0),
        (0.8, 0.6, 1.0),
        (0.8, 0.8, 1.0),
        (0.8, 0.8, 1.1),
    ]

    for recent_years in [2, 3]:
        for block_minutes in [60]:
            for bounds_days in global_bounds_candidates:
                for alpha_segments in global_alpha_candidates:
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

    for local_block_minutes in [15, 30, 60]:
        for local_k_neighbors in [6, 8, 12]:
            for blend_beta in [0.5, 0.75, 0.9, 1.0]:
                for bounds_days in local_bounds_candidates:
                    for alpha_segments in local_alpha_candidates:
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
                                    "family": "local" if float(blend_beta) >= 0.999 else "blend",
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
    target_item: dict,
    recent_years: int,
    local_block_minutes: int,
    local_k_neighbors: int,
    blend_beta: float,
    family: str = "mean",
) -> np.ndarray:
    global_overlay = build_recent_stage3_overlay(
        train_items=train_items,
        recent_years=int(recent_years),
        block_minutes=60,
    )
    years = [int(item["episode"]["year"]) for item in train_items]
    max_year = max(years)
    usable = [
        item
        for item in train_items
        if int(item["episode"]["year"]) >= int(max_year) - int(recent_years) + 1
    ]
    if not usable:
        usable = train_items
    x = np.stack([np.asarray(item["stage4_descriptor"], dtype=np.float64) for item in usable], axis=0)
    z = np.asarray(target_item["stage4_descriptor"], dtype=np.float64)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    dist = np.sqrt(np.mean(np.square((x - z[None, :]) / sigma), axis=1))
    order = np.argsort(dist)[: min(int(local_k_neighbors), len(usable))]
    dsel = np.asarray(dist[order], dtype=np.float64)
    dsel = dsel - np.min(dsel)
    weights = np.exp(-dsel)
    wsum = float(np.sum(weights))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        weights = np.ones_like(dsel, dtype=np.float64)
        wsum = float(np.sum(weights))
    weights = weights / wsum
    selected = [
        smooth_stage3_overlay_series(
            usable[int(idx)]["residual_forecast"],
            block_minutes=int(local_block_minutes),
        )
        for idx in order.tolist()
    ]
    if str(family) == "consensus":
        stack = np.stack(selected, axis=0)
        az = stack[:, :, 0]
        el = stack[:, :, 1]
        agreement = np.abs(np.mean(np.sign(az), axis=0))
        az_med = np.median(az, axis=0) * agreement
        el_med = np.median(el, axis=0)
        local_overlay = np.column_stack([az_med, el_med]).astype(np.float64)
    else:
        local_overlay = np.zeros_like(np.asarray(usable[int(order[0])]["residual_forecast"], dtype=np.float64))
        for w, arr in zip(weights.tolist(), selected):
            local_overlay += float(w) * np.asarray(arr, dtype=np.float64)
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


STAGE4_LOW_RANK_BASE = {
    "family": "mean",
    "recent_years": 2,
    "local_block_minutes": 15,
    "local_k_neighbors": 4,
    "blend_beta": 0.75,
    "start_day": 40,
    "alpha_az": 0.25,
}


def build_stage4_source_item_from_stage3(stage3_item: dict) -> dict:
    return build_stage4_overlay_item(
        episode=stage3_item["episode"],
        pred_stage3_aligned=stage3_item["pred_stage3_aligned"],
    )


def apply_fixed_stage4_base_to_stage3_item(
    stage3_item: dict,
    train_stage4_sources: list[dict],
) -> dict:
    target_item = build_stage4_source_item_from_stage3(stage3_item)
    overlay = build_stage4_local_overlay(
        train_items=train_stage4_sources,
        target_item=target_item,
        recent_years=int(STAGE4_LOW_RANK_BASE["recent_years"]),
        local_block_minutes=int(STAGE4_LOW_RANK_BASE["local_block_minutes"]),
        local_k_neighbors=int(STAGE4_LOW_RANK_BASE["local_k_neighbors"]),
        blend_beta=float(STAGE4_LOW_RANK_BASE["blend_beta"]),
        family=str(STAGE4_LOW_RANK_BASE["family"]),
    )
    pred_stage4_aligned, metrics = apply_stage4_az_overlay_to_entry(
        pred_stage3_aligned=stage3_item["pred_stage3_aligned"],
        episode=stage3_item["episode"],
        overlay_forecast=overlay,
        start_day=int(STAGE4_LOW_RANK_BASE["start_day"]),
        alpha_az=float(STAGE4_LOW_RANK_BASE["alpha_az"]),
    )
    out = build_stage4_overlay_item(
        episode=stage3_item["episode"],
        pred_stage3_aligned=pred_stage4_aligned,
    )
    out["pred_stage4_aligned"] = np.asarray(pred_stage4_aligned, dtype=np.float64)
    out["base_metrics"] = metrics
    return out


def build_stage4_lowrank_surface(item: dict, block_minutes: int) -> np.ndarray:
    block = max(1, int(block_minutes))
    residual = np.asarray(item["residual_forecast"][:, 0], dtype=np.float64)
    usable = (residual.shape[0] // block) * block
    if usable <= 0:
        return np.zeros((1, 1), dtype=np.float64)
    r = residual[:usable].reshape(-1, block).mean(axis=1)
    per_day = max(1, 1440 // block)
    days = max(1, r.shape[0] // per_day)
    return r[: days * per_day].reshape(days, per_day)


def stage4_surface_to_series(surface: np.ndarray, total_minutes: int, block_minutes: int) -> np.ndarray:
    block = max(1, int(block_minutes))
    flat = np.repeat(np.asarray(surface, dtype=np.float64).reshape(-1), block)
    if flat.shape[0] < int(total_minutes):
        pad = np.full(int(total_minutes) - flat.shape[0], flat[-1], dtype=np.float64)
        flat = np.concatenate([flat, pad], axis=0)
    return flat[: int(total_minutes)]


def fit_stage4_lowrank_surface_from_descriptors(
    train_items: list[dict],
    block_minutes: int,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not train_items:
        raise RuntimeError("stage4 lowrank training items are empty")
    surfaces = np.stack([build_stage4_lowrank_surface(item, block_minutes) for item in train_items], axis=0)
    flat = surfaces.reshape(surfaces.shape[0], -1)
    mean_surface = np.mean(flat, axis=0)
    centered = flat - mean_surface[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_rank = max(1, min(int(rank), int(vt.shape[0])))
    basis = np.asarray(vt[:use_rank], dtype=np.float64)
    coeff = centered @ basis.T

    x = np.stack([np.asarray(item["stage4_descriptor"], dtype=np.float64) for item in train_items], axis=0)
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    xz = (x - mu) / sigma
    X = np.column_stack([np.ones(xz.shape[0]), xz])
    xtx = X.T @ X + float(ridge) * np.eye(X.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xtx, X.T @ coeff)
    return mean_surface, basis, mu, sigma, coef


def predict_stage4_lowrank_surface_for_item(
    item: dict,
    mean_surface: np.ndarray,
    basis: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    z = np.asarray(item["stage4_descriptor"], dtype=np.float64)[None, :]
    xz = (z - mu) / sigma
    X = np.column_stack([np.ones(xz.shape[0]), xz])
    coeff = X @ coef
    flat = mean_surface[None, :] + coeff @ basis
    return np.asarray(flat.reshape(-1), dtype=np.float64)


def apply_stage4_lowrank_surface_to_item(
    item: dict,
    series: np.ndarray,
    alpha: float,
    start_day: int,
) -> tuple[np.ndarray, dict]:
    ep = item["episode"]
    fm = np.asarray(ep["forecast_mask"], dtype=bool)
    pred = np.asarray(item["pred_stage4_aligned"], dtype=np.float64).copy()
    count = int(np.sum(fm))
    corr = np.asarray(series, dtype=np.float64)[:count]
    start = max(0, min(count, int(start_day) * 1440))
    az_corr = np.zeros(count, dtype=np.float64)
    az_corr[start:] = float(alpha) * corr[start:]
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    metrics = core.compute_metrics_azel(ep["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def choose_stage4_lowrank_surface_forward_cv(
    items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    years = sorted(int(y) for y in validation_years if items_by_year.get(int(y)))
    if years:
        years = years[-1:]
    scores: list[dict] = []
    baseline_vals = []
    for year in years:
        targets = items_by_year.get(int(year), [])
        for item in targets:
            baseline_vals.append(float(item["base_metrics"]["max_abs_error_max"]))
    if baseline_vals:
        scores.append(
            {
                "family": "none",
                "val_max_abs_error": float(np.max(baseline_vals)),
                "val_mean_max_abs_error": float(np.mean(baseline_vals)),
                "count": int(len(baseline_vals)),
            }
        )

    for block_minutes in [60]:
        for rank in [2, 3]:
            for ridge in [10.0, 1.0]:
                for start_day in [35, 42, 50, 60]:
                    for alpha in [0.2, 0.3, 0.4]:
                        vals = []
                        for year in years:
                            train_items = [
                                it for yy, items in items_by_year.items() if int(yy) < int(year) for it in items
                            ]
                            targets = items_by_year.get(int(year), [])
                            if not train_items or not targets:
                                continue
                            params = fit_stage4_lowrank_surface_from_descriptors(
                                train_items=train_items,
                                block_minutes=int(block_minutes),
                                rank=int(rank),
                                ridge=float(ridge),
                            )
                            for item in targets:
                                surf = predict_stage4_lowrank_surface_for_item(item, *params)
                                series = stage4_surface_to_series(
                                    surface=surf,
                                    total_minutes=int(np.sum(item["episode"]["forecast_mask"])),
                                    block_minutes=int(block_minutes),
                                )
                                _, metrics = apply_stage4_lowrank_surface_to_item(
                                    item=item,
                                    series=series,
                                    alpha=float(alpha),
                                    start_day=int(start_day),
                                )
                                vals.append(float(metrics["max_abs_error_max"]))
                        if vals:
                            scores.append(
                                {
                                    "family": "lowrank",
                                    "block_minutes": int(block_minutes),
                                    "rank": int(rank),
                                    "ridge": float(ridge),
                                    "start_day": int(start_day),
                                    "alpha": float(alpha),
                                    "val_max_abs_error": float(np.max(vals)),
                                    "val_mean_max_abs_error": float(np.mean(vals)),
                                    "count": int(len(vals)),
                                }
                            )
    if not scores:
        raise RuntimeError("No stage4 lowrank surface CV scores could be computed")
    scores.sort(
        key=lambda x: (
            float(x["val_max_abs_error"]),
            0 if str(x["family"]) == "lowrank" else 1,
            float(x["val_mean_max_abs_error"]),
            int(x.get("rank", 9999)),
            -float(x.get("ridge", 0.0)),
            int(x.get("start_day", 0)),
            float(x.get("alpha", 0.0)),
        )
    )
    best = scores[0]
    return {
        "selected_family": str(best["family"]),
        "selected_block_minutes": int(best.get("block_minutes", 0)),
        "selected_rank": int(best.get("rank", 0)),
        "selected_ridge": float(best.get("ridge", 0.0)),
        "selected_start_day": int(best.get("start_day", 0)),
        "selected_alpha": float(best.get("alpha", 0.0)),
        "scores": scores,
    }


def choose_stage4_az_overlay_forward_cv(
    items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    years = sorted(int(y) for y in validation_years if items_by_year.get(int(y)))
    if years:
        years = years[-1:]
    weights = {int(y): 1.0 + 0.5 * i for i, y in enumerate(years, start=1)}
    score_weight_mean = 0.15
    scores: list[dict] = []

    baseline_vals = []
    for year in years:
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

    for overlay_family in ["mean", "consensus"]:
        for recent_years in [2]:
            for local_block_minutes in [15, 30]:
                for local_k_neighbors in [4, 8, 12]:
                    for blend_beta in [0.75, 0.9, 1.0]:
                        for start_day in [35, 40, 45, 50]:
                            for alpha_az in [0.2, 0.25, 0.35, 0.38, 0.4, 0.5, 0.6]:
                                vals = []
                                for year in years:
                                    train = [it for yy, items in items_by_year.items() if int(yy) < int(year) for it in items]
                                    targets = items_by_year.get(int(year), [])
                                    if not train or not targets:
                                        continue
                                    for target in targets:
                                        overlay = build_stage4_local_overlay(
                                            train_items=train,
                                            target_item=target,
                                            recent_years=int(recent_years),
                                            local_block_minutes=int(local_block_minutes),
                                            local_k_neighbors=int(local_k_neighbors),
                                            blend_beta=float(blend_beta),
                                            family=str(overlay_family),
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
                                        "overlay_family": str(overlay_family),
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
    # Prioritize worst-case validation error. When multiple candidates tie on val_max,
    # prefer sharper local corrections because the residual blocks are typically
    # under-corrected after stage3.
    scores.sort(
        key=lambda x: (
            float(x["val_max_abs_error"]),
            float(x["val_weighted_mean_max_abs_error"]),
            1 if str(x["family"]) == "none" else 0,
            1 if str(x.get("overlay_family", "mean")) != "mean" else 0,
            int(x.get("local_k_neighbors", 9999)),
            -float(x.get("blend_beta", 0.0)),
            -int(x.get("start_day", 0)),
            -float(x.get("alpha_az", 0.0)),
            x["selection_score"],
        )
    )
    best = scores[0]
    return {
        "selected_family": str(best["family"]),
        "selected_overlay_family": str(best.get("overlay_family", "mean")),
        "selected_recent_years": int(best.get("recent_years", 0)),
        "selected_local_block_minutes": int(best.get("local_block_minutes", 0)),
        "selected_local_k_neighbors": int(best.get("local_k_neighbors", 0)),
        "selected_blend_beta": float(best.get("blend_beta", 0.0)),
        "selected_start_day": int(best.get("start_day", 0)),
        "selected_alpha_az": float(best.get("alpha_az", 0.0)),
        "selection_score_weight_mean": float(score_weight_mean),
        "scores": scores,
    }


STAGE4_PIECEWISE_SELECTOR_BASE = {
    "overlay_family": "mean",
    "recent_years": 2,
    "local_block_minutes": 15,
    "local_k_neighbors": 4,
}

STAGE4_PIECEWISE_SELECTOR_CANDIDATES = [
    {
        "blend_beta": float(blend_beta),
        "start1": int(start1),
        "start2": int(start2),
        "alpha1": float(alpha1),
        "alpha2": float(alpha2),
    }
    for blend_beta in [0.9, 1.0]
    for start1 in [40, 45]
    for start2 in [55, 60, 65]
    for alpha1 in [0.2, 0.25, 0.3, 0.35, 0.38, 0.4]
    for alpha2 in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    if alpha2 >= alpha1
]


STAGE4_FIXED_PIECEWISE_SEARCH = {
    "overlay_families": ["mean"],
    "recent_years": [2],
    "local_block_minutes": [15, 30],
    "local_k_neighbors": [4, 8],
    "blend_beta": [0.75, 0.9],
    "start1": [35, 40],
    "start2": [50, 55],
    "alpha1": [0.15, 0.2, 0.25],
    "alpha2": [0.35, 0.4, 0.45, 0.5],
}

STAGE5_JOINT_SURFACE_BASE = {
    "overlay_family": "mean",
    "recent_years": 2,
    "local_block_minutes": 15,
    "local_k_neighbors": 4,
}

STAGE5_JOINT_SURFACE_FIXED_CAND = {
    "blend_beta": 0.9,
    "start1": 45,
    "start2": 55,
    "alpha1": 0.2,
    "alpha2": 0.4,
}

STAGE5_JOINT_SURFACE_CANDIDATES = [
    {
        "block_minutes": int(block_minutes),
        "rank": int(rank),
        "ridge": float(ridge),
        "az_start_day": int(az_start_day),
        "el_start_day": int(el_start_day),
        "alpha_az": float(alpha_az),
        "alpha_el": float(alpha_el),
    }
    for block_minutes in [15]
    for rank in [2, 4]
    for ridge in [0.1, 10.0]
    for az_start_day in [35]
    for el_start_day in [0, 7]
    for alpha_az in [0.05, 0.1, 0.15]
    for alpha_el in [0.0, 0.05, 0.1, 0.15]
]


def apply_stage4_piecewise_az_overlay_to_entry(
    pred_stage3_aligned: np.ndarray,
    episode: dict,
    overlay_forecast: np.ndarray,
    start1_day: int,
    start2_day: int,
    alpha1: float,
    alpha2: float,
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(pred_stage3_aligned, dtype=np.float64).copy()
    fm = np.asarray(episode["forecast_mask"], dtype=bool)
    corr = np.asarray(overlay_forecast, dtype=np.float64)
    count = int(np.sum(fm))
    if corr.shape[0] != count:
        raise ValueError("stage4 piecewise overlay length mismatch")
    az_corr = np.zeros(count, dtype=np.float64)
    s1 = max(0, min(count, int(start1_day) * 1440))
    s2 = max(s1, min(count, int(start2_day) * 1440))
    az_corr[s1:s2] = float(alpha1) * corr[s1:s2, 0]
    az_corr[s2:] = float(alpha2) * corr[s2:, 0]
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    metrics = core.compute_metrics_azel(episode["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def stage4_piecewise_selector_distance(a: np.ndarray, b: np.ndarray, sigma: np.ndarray) -> float:
    z = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / sigma
    return float(np.sqrt(np.mean(np.square(z))))


def choose_stage4_piecewise_az_selector_forward_cv(
    items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    episodes = []
    years_all = sorted(int(y) for y in items_by_year)
    for year in years_all:
        train_items = [it for yy, items in items_by_year.items() if int(yy) < int(year) for it in items]
        if not train_items:
            continue
        for item in items_by_year.get(int(year), []):
            overlays = {}
            for blend_beta in [0.9, 1.0]:
                overlays[blend_beta] = build_stage4_local_overlay(
                    train_items=train_items,
                    target_item=item,
                    recent_years=int(STAGE4_PIECEWISE_SELECTOR_BASE["recent_years"]),
                    local_block_minutes=int(STAGE4_PIECEWISE_SELECTOR_BASE["local_block_minutes"]),
                    local_k_neighbors=int(STAGE4_PIECEWISE_SELECTOR_BASE["local_k_neighbors"]),
                    blend_beta=float(blend_beta),
                    family=str(STAGE4_PIECEWISE_SELECTOR_BASE["overlay_family"]),
                )
            cand_errors = []
            for cand in STAGE4_PIECEWISE_SELECTOR_CANDIDATES:
                _, metrics = apply_stage4_piecewise_az_overlay_to_entry(
                    pred_stage3_aligned=item["pred_stage3_aligned"],
                    episode=item["episode"],
                    overlay_forecast=overlays[float(cand["blend_beta"])],
                    start1_day=int(cand["start1"]),
                    start2_day=int(cand["start2"]),
                    alpha1=float(cand["alpha1"]),
                    alpha2=float(cand["alpha2"]),
                )
                cand_errors.append(float(metrics["max_abs_error_max"]))
            episodes.append(
                {
                    "year": int(year),
                    "descriptor": np.asarray(item["stage4_descriptor"], dtype=np.float64),
                    "candidate_errors": np.asarray(cand_errors, dtype=np.float64),
                }
            )

    years = sorted(int(y) for y in validation_years if any(int(ep["year"]) == int(y) for ep in episodes))
    if not years:
        raise RuntimeError("No stage4 piecewise selector validation years could be computed")
    descriptor_stack = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in episodes], axis=0)
    sigma = np.std(descriptor_stack, axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)

    scores = []
    for k in [1, 2, 3, 4, 6, 8]:
        vals = []
        for year in years:
            year_eps = [ep for ep in episodes if int(ep["year"]) == int(year)]
            train_eps = [ep for ep in episodes if int(ep["year"]) < int(year)]
            if not year_eps or not train_eps:
                continue
            for ep in year_eps:
                dist = np.asarray(
                    [
                        stage4_piecewise_selector_distance(ep["descriptor"], tr["descriptor"], sigma)
                        for tr in train_eps
                    ],
                    dtype=np.float64,
                )
                order = np.argsort(dist)[: min(int(k), len(train_eps))]
                dsel = dist[order] - np.min(dist[order])
                weights = np.exp(-dsel)
                wsum = float(np.sum(weights))
                if (not np.isfinite(wsum)) or wsum <= 0.0:
                    weights = np.ones_like(dsel, dtype=np.float64)
                    wsum = float(np.sum(weights))
                weights = weights / wsum
                err = np.zeros(len(STAGE4_PIECEWISE_SELECTOR_CANDIDATES), dtype=np.float64)
                for ww, idx in zip(weights.tolist(), order.tolist()):
                    err += float(ww) * np.asarray(train_eps[int(idx)]["candidate_errors"], dtype=np.float64)
                best_idx = int(np.argmin(err))
                vals.append(float(ep["candidate_errors"][best_idx]))
        if vals:
            scores.append(
                {
                    "k": int(k),
                    "val_max_abs_error": float(np.max(vals)),
                    "val_mean_max_abs_error": float(np.mean(vals)),
                    "count": int(len(vals)),
                }
            )

    if not scores:
        raise RuntimeError("No stage4 piecewise selector CV scores could be computed")
    scores.sort(key=lambda x: (x["val_max_abs_error"], x["val_mean_max_abs_error"], x["k"]))
    best = scores[0]
    return {
        "selected_k": int(best["k"]),
        "base_overlay": dict(STAGE4_PIECEWISE_SELECTOR_BASE),
        "candidates": STAGE4_PIECEWISE_SELECTOR_CANDIDATES,
        "episode_bank": episodes,
        "scores": scores,
    }


def predict_stage4_piecewise_az_selector_candidate(
    episode_bank: list[dict],
    target_item: dict,
    stage4_piecewise_cv: dict,
) -> dict:
    if not episode_bank:
        raise RuntimeError("stage4 piecewise selector episode bank is empty")
    sigma = np.std(np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in episode_bank], axis=0), axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    z = np.asarray(target_item["stage4_descriptor"], dtype=np.float64)
    dist = np.asarray(
        [stage4_piecewise_selector_distance(z, ep["descriptor"], sigma) for ep in episode_bank],
        dtype=np.float64,
    )
    order = np.argsort(dist)[: min(int(stage4_piecewise_cv["selected_k"]), len(episode_bank))]
    dsel = dist[order] - np.min(dist[order])
    weights = np.exp(-dsel)
    wsum = float(np.sum(weights))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        weights = np.ones_like(dsel, dtype=np.float64)
        wsum = float(np.sum(weights))
    weights = weights / wsum
    err = np.zeros(len(stage4_piecewise_cv["candidates"]), dtype=np.float64)
    for ww, idx in zip(weights.tolist(), order.tolist()):
        err += float(ww) * np.asarray(episode_bank[int(idx)]["candidate_errors"], dtype=np.float64)
    best_idx = int(np.argmin(err))
    return dict(stage4_piecewise_cv["candidates"][best_idx])


def choose_stage4_fixed_piecewise_az_forward_cv(
    items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    years = sorted(int(y) for y in validation_years if items_by_year.get(int(y)))
    if not years:
        raise RuntimeError("No stage4 fixed piecewise validation years could be computed")
    weights = {int(y): float(i + 1) for i, y in enumerate(years)}
    scores = []
    for overlay_family in STAGE4_FIXED_PIECEWISE_SEARCH["overlay_families"]:
        for recent_years in STAGE4_FIXED_PIECEWISE_SEARCH["recent_years"]:
            for local_block_minutes in STAGE4_FIXED_PIECEWISE_SEARCH["local_block_minutes"]:
                for local_k_neighbors in STAGE4_FIXED_PIECEWISE_SEARCH["local_k_neighbors"]:
                    for blend_beta in STAGE4_FIXED_PIECEWISE_SEARCH["blend_beta"]:
                        for start1 in STAGE4_FIXED_PIECEWISE_SEARCH["start1"]:
                            for start2 in STAGE4_FIXED_PIECEWISE_SEARCH["start2"]:
                                if int(start2) < int(start1):
                                    continue
                                for alpha1 in STAGE4_FIXED_PIECEWISE_SEARCH["alpha1"]:
                                    for alpha2 in STAGE4_FIXED_PIECEWISE_SEARCH["alpha2"]:
                                        if float(alpha2) < float(alpha1):
                                            continue
                                        vals = []
                                        for year in years:
                                            train = [
                                                it
                                                for yy, items in items_by_year.items()
                                                if int(yy) < int(year)
                                                for it in items
                                            ]
                                            targets = items_by_year.get(int(year), [])
                                            if not train or not targets:
                                                continue
                                            for target in targets:
                                                overlay = build_stage4_local_overlay(
                                                    train_items=train,
                                                    target_item=target,
                                                    recent_years=int(recent_years),
                                                    local_block_minutes=int(local_block_minutes),
                                                    local_k_neighbors=int(local_k_neighbors),
                                                    blend_beta=float(blend_beta),
                                                    family=str(overlay_family),
                                                )
                                                _, metrics = apply_stage4_piecewise_az_overlay_to_entry(
                                                    pred_stage3_aligned=target["pred_stage3_aligned"],
                                                    episode=target["episode"],
                                                    overlay_forecast=overlay,
                                                    start1_day=int(start1),
                                                    start2_day=int(start2),
                                                    alpha1=float(alpha1),
                                                    alpha2=float(alpha2),
                                                )
                                                vals.append((int(year), float(metrics["max_abs_error_max"])))
                                        if vals:
                                            weighted_mean = float(
                                                np.sum([weights.get(y, 1.0) * v for y, v in vals]) /
                                                np.sum([weights.get(y, 1.0) for y, _ in vals])
                                            )
                                            val_max = float(np.max([v for _, v in vals]))
                                            val_mean = float(np.mean([v for _, v in vals]))
                                            scores.append(
                                                {
                                                    "overlay_family": str(overlay_family),
                                                    "recent_years": int(recent_years),
                                                    "local_block_minutes": int(local_block_minutes),
                                                    "local_k_neighbors": int(local_k_neighbors),
                                                    "blend_beta": float(blend_beta),
                                                    "start1": int(start1),
                                                    "start2": int(start2),
                                                    "alpha1": float(alpha1),
                                                    "alpha2": float(alpha2),
                                                    "val_weighted_mean_max_abs_error": weighted_mean,
                                                    "val_mean_max_abs_error": val_mean,
                                                    "val_max_abs_error": val_max,
                                                    "count": int(len(vals)),
                                                }
                                            )
    if not scores:
        raise RuntimeError("No stage4 fixed piecewise CV scores could be computed")
    scores.sort(
        key=lambda x: (
            float(x["val_max_abs_error"]),
            float(x["val_weighted_mean_max_abs_error"]),
            float(x["val_mean_max_abs_error"]),
            x["start1"],
            x["start2"],
            x["local_k_neighbors"],
            -x["blend_beta"],
            x["alpha1"],
            x["alpha2"],
        )
    )
    best = scores[0]
    return {
        "selected_overlay_family": str(best["overlay_family"]),
        "selected_recent_years": int(best["recent_years"]),
        "selected_local_block_minutes": int(best["local_block_minutes"]),
        "selected_local_k_neighbors": int(best["local_k_neighbors"]),
        "selected_blend_beta": float(best["blend_beta"]),
        "selected_start1": int(best["start1"]),
        "selected_start2": int(best["start2"]),
        "selected_alpha1": float(best["alpha1"]),
        "selected_alpha2": float(best["alpha2"]),
        "search_space": dict(STAGE4_FIXED_PIECEWISE_SEARCH),
        "scores": scores,
    }


def build_stage5_fixed_piecewise_item(
    stage4_item: dict,
    train_stage4_items: list[dict],
) -> dict:
    if train_stage4_items:
        overlay = build_stage4_local_overlay(
            train_items=train_stage4_items,
            target_item=stage4_item,
            recent_years=int(STAGE5_JOINT_SURFACE_BASE["recent_years"]),
            local_block_minutes=int(STAGE5_JOINT_SURFACE_BASE["local_block_minutes"]),
            local_k_neighbors=int(STAGE5_JOINT_SURFACE_BASE["local_k_neighbors"]),
            blend_beta=float(STAGE5_JOINT_SURFACE_FIXED_CAND["blend_beta"]),
            family=str(STAGE5_JOINT_SURFACE_BASE["overlay_family"]),
        )
        pred_piecewise_aligned, metrics = apply_stage4_piecewise_az_overlay_to_entry(
            pred_stage3_aligned=stage4_item["pred_stage3_aligned"],
            episode=stage4_item["episode"],
            overlay_forecast=overlay,
            start1_day=int(STAGE5_JOINT_SURFACE_FIXED_CAND["start1"]),
            start2_day=int(STAGE5_JOINT_SURFACE_FIXED_CAND["start2"]),
            alpha1=float(STAGE5_JOINT_SURFACE_FIXED_CAND["alpha1"]),
            alpha2=float(STAGE5_JOINT_SURFACE_FIXED_CAND["alpha2"]),
        )
    else:
        pred_piecewise_aligned = np.asarray(stage4_item["pred_stage3_aligned"], dtype=np.float64)
        fm = np.asarray(stage4_item["episode"]["forecast_mask"], dtype=bool)
        metrics = core.compute_metrics_azel(
            stage4_item["episode"]["true_aligned"][fm],
            pred_piecewise_aligned[fm],
        )["overall"]
    out = build_stage4_overlay_item(
        episode=stage4_item["episode"],
        pred_stage3_aligned=pred_piecewise_aligned,
    )
    out["pred_piecewise_aligned"] = np.asarray(pred_piecewise_aligned, dtype=np.float64)
    out["piecewise_metrics"] = metrics
    return out


def build_stage5_joint_surface(item: dict, block_minutes: int) -> np.ndarray:
    block = max(1, int(block_minutes))
    residual = np.asarray(item["residual_forecast"], dtype=np.float64)
    usable = (residual.shape[0] // block) * block
    if usable <= 0:
        return np.zeros((1, 2), dtype=np.float64)
    return residual[:usable].reshape(-1, block, 2).mean(axis=1)


def fit_stage5_joint_surface_from_descriptors(
    train_items: list[dict],
    block_minutes: int,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not train_items:
        raise RuntimeError("stage5 joint surface training items are empty")
    surfaces = np.stack([build_stage5_joint_surface(item, block_minutes) for item in train_items], axis=0)
    flat = surfaces.reshape(surfaces.shape[0], -1)
    mean_surface = np.mean(flat, axis=0)
    centered = flat - mean_surface[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    use_rank = max(1, min(int(rank), int(vt.shape[0])))
    basis = np.asarray(vt[:use_rank], dtype=np.float64)
    coeff = centered @ basis.T

    x = np.stack([np.asarray(item["stage4_descriptor"], dtype=np.float64) for item in train_items], axis=0)
    mu = np.mean(x, axis=0, keepdims=True)
    sigma = np.std(x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    xz = (x - mu) / sigma
    X = np.column_stack([np.ones(xz.shape[0]), xz])
    xtx = X.T @ X + float(ridge) * np.eye(X.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xtx, X.T @ coeff)
    return mean_surface, basis, mu, sigma, coef


def predict_stage5_joint_surface_for_item(
    item: dict,
    mean_surface: np.ndarray,
    basis: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    coef: np.ndarray,
) -> np.ndarray:
    z = np.asarray(item["stage4_descriptor"], dtype=np.float64)[None, :]
    xz = (z - mu) / sigma
    X = np.column_stack([np.ones(xz.shape[0]), xz])
    coeff = X @ coef
    flat = mean_surface[None, :] + coeff @ basis
    return np.asarray(flat.reshape(-1, 2), dtype=np.float64)


def apply_stage5_joint_surface_to_item(
    item: dict,
    surface: np.ndarray,
    block_minutes: int,
    az_start_day: int,
    el_start_day: int,
    alpha_az: float,
    alpha_el: float,
) -> tuple[np.ndarray, dict]:
    ep = item["episode"]
    fm = np.asarray(ep["forecast_mask"], dtype=bool)
    pred = np.asarray(item["pred_piecewise_aligned"], dtype=np.float64).copy()
    count = int(np.sum(fm))
    series = np.repeat(np.asarray(surface, dtype=np.float64), int(block_minutes), axis=0)
    if series.shape[0] < count:
        series = np.concatenate([series, np.repeat(series[-1:], count - series.shape[0], axis=0)], axis=0)
    series = series[:count]
    saz = min(count, int(az_start_day) * 1440)
    sel = min(count, int(el_start_day) * 1440)
    az_corr = np.zeros(count, dtype=np.float64)
    el_corr = np.zeros(count, dtype=np.float64)
    az_corr[saz:] = float(alpha_az) * series[saz:, 0]
    el_corr[sel:] = float(alpha_el) * series[sel:, 1]
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    pred[fm, 4] = pred[fm, 4] - el_corr
    metrics = core.compute_metrics_azel(ep["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def choose_stage5_joint_surface_selector_forward_cv(
    stage4_items_by_year: dict[int, list[dict]],
    validation_years: list[int],
) -> dict:
    years_all = sorted(int(y) for y in stage4_items_by_year)
    piece_by_year: dict[int, list[dict]] = {}
    for year in years_all:
        train = [it for yy, items in stage4_items_by_year.items() if int(yy) < int(year) for it in items]
        targets = stage4_items_by_year.get(int(year), [])
        if not targets:
            continue
        piece_by_year[int(year)] = [build_stage5_fixed_piecewise_item(it, train) for it in targets]

    validation_years = [int(y) for y in validation_years if int(y) in piece_by_year]
    if not validation_years:
        raise RuntimeError("No stage5 joint surface validation years could be computed")

    candidate_errors_by_year: dict[int, list[dict]] = {}
    for year in validation_years:
        train = [it for yy, items in piece_by_year.items() if int(yy) < int(year) for it in items]
        targets = piece_by_year.get(int(year), [])
        if not train or not targets:
            continue
        models = [
            fit_stage5_joint_surface_from_descriptors(
                train_items=train,
                block_minutes=int(cand["block_minutes"]),
                rank=int(cand["rank"]),
                ridge=float(cand["ridge"]),
            )
            for cand in STAGE5_JOINT_SURFACE_CANDIDATES
        ]
        bank_items: list[dict] = []
        for item in targets:
            errs = []
            for cand, params in zip(STAGE5_JOINT_SURFACE_CANDIDATES, models):
                surf = predict_stage5_joint_surface_for_item(item, *params)
                _, metrics = apply_stage5_joint_surface_to_item(
                    item=item,
                    surface=surf,
                    block_minutes=int(cand["block_minutes"]),
                    az_start_day=int(cand["az_start_day"]),
                    el_start_day=int(cand["el_start_day"]),
                    alpha_az=float(cand["alpha_az"]),
                    alpha_el=float(cand["alpha_el"]),
                )
                errs.append(float(metrics["max_abs_error_max"]))
            bank_items.append(
                {
                    "year": int(year),
                    "descriptor": np.asarray(item["stage4_descriptor"], dtype=np.float64),
                    "candidate_errors": np.asarray(errs, dtype=np.float64),
                }
            )
        if bank_items:
            candidate_errors_by_year[int(year)] = bank_items

    all_eps = [ep for yy, items in candidate_errors_by_year.items() for ep in items]
    if not all_eps:
        raise RuntimeError("No stage5 joint surface selector episode bank could be computed")
    Xall = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in all_eps], axis=0)
    sigma = np.std(Xall, axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)

    def predict_idx(train_eps: list[dict], z: np.ndarray, k: int, temp: float) -> int:
        X = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in train_eps], axis=0)
        dz = (X - z[None, :]) / sigma[None, :]
        dist = np.sqrt(np.mean(np.square(dz), axis=1))
        order = np.argsort(dist)[: min(int(k), len(train_eps))]
        dsel = np.asarray(dist[order], dtype=np.float64) - float(np.min(dist[order]))
        weights = np.exp(-float(temp) * dsel)
        wsum = float(np.sum(weights))
        if (not np.isfinite(wsum)) or wsum <= 0.0:
            weights = np.ones_like(dsel, dtype=np.float64)
            wsum = float(np.sum(weights))
        weights = weights / wsum
        err = np.zeros(len(STAGE5_JOINT_SURFACE_CANDIDATES), dtype=np.float64)
        for ww, idx in zip(weights.tolist(), order.tolist()):
            err += float(ww) * np.asarray(train_eps[int(idx)]["candidate_errors"], dtype=np.float64)
        return int(np.argmin(err))

    scores = []
    for k in [1, 2, 3, 4, 6]:
        for temp in [0.25, 0.5, 1.0, 2.0]:
            vals = []
            for year in validation_years:
                train_eps = [ep for ep in all_eps if int(ep["year"]) < int(year)]
                year_eps = [ep for ep in all_eps if int(ep["year"]) == int(year)]
                if not train_eps or not year_eps:
                    continue
                for ep in year_eps:
                    idx = predict_idx(train_eps, np.asarray(ep["descriptor"], dtype=np.float64), int(k), float(temp))
                    vals.append(float(ep["candidate_errors"][idx]))
            if vals:
                scores.append(
                    {
                        "k": int(k),
                        "temp": float(temp),
                        "val_max_abs_error": float(np.max(vals)),
                        "val_mean_max_abs_error": float(np.mean(vals)),
                        "count": int(len(vals)),
                    }
                )
    if not scores:
        raise RuntimeError("No stage5 joint surface selector scores could be computed")
    scores.sort(key=lambda x: (x["val_max_abs_error"], x["val_mean_max_abs_error"], x["k"], x["temp"]))
    best = scores[0]
    return {
        "piece_items_by_year": piece_by_year,
        "episode_bank": all_eps,
        "selected_k": int(best["k"]),
        "selected_temp": float(best["temp"]),
        "candidate_space": list(STAGE5_JOINT_SURFACE_CANDIDATES),
        "fixed_base_overlay": dict(STAGE5_JOINT_SURFACE_BASE),
        "fixed_piecewise_candidate": dict(STAGE5_JOINT_SURFACE_FIXED_CAND),
        "scores": scores,
    }


def predict_stage5_joint_surface_candidate(
    episode_bank: list[dict],
    target_item: dict,
    selected_k: int,
    selected_temp: float,
) -> dict:
    Xall = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in episode_bank], axis=0)
    sigma = np.std(Xall, axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    z = np.asarray(target_item["stage4_descriptor"], dtype=np.float64)
    dist = np.sqrt(np.mean(np.square((Xall - z[None, :]) / sigma[None, :]), axis=1))
    order = np.argsort(dist)[: min(int(selected_k), len(episode_bank))]
    dsel = np.asarray(dist[order], dtype=np.float64) - float(np.min(dist[order]))
    weights = np.exp(-float(selected_temp) * dsel)
    wsum = float(np.sum(weights))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        weights = np.ones_like(dsel, dtype=np.float64)
        wsum = float(np.sum(weights))
    weights = weights / wsum
    err = np.zeros(len(STAGE5_JOINT_SURFACE_CANDIDATES), dtype=np.float64)
    for ww, idx in zip(weights.tolist(), order.tolist()):
        err += float(ww) * np.asarray(episode_bank[int(idx)]["candidate_errors"], dtype=np.float64)
    best_idx = int(np.argmin(err))
    return dict(STAGE5_JOINT_SURFACE_CANDIDATES[best_idx])


STAGE5_POST_FAMILY_CANDIDATES = (
    [
        {
            "family": "none",
            "recent_years": 0,
            "early_window": None,
            "late_window": None,
            "alpha_az_early": 0.0,
            "alpha_el_early": 0.0,
            "alpha_az_late": 0.0,
            "alpha_el_late": 0.0,
        }
    ]
    + [
        {
            "family": "window_recent_mean",
            "recent_years": int(recent_years),
            "early_window": list(early_window) if early_window is not None else None,
            "late_window": list(late_window) if late_window is not None else None,
            "alpha_az_early": float(alpha_az_early),
            "alpha_el_early": float(alpha_el_early),
            "alpha_az_late": float(alpha_az_late),
            "alpha_el_late": float(alpha_el_late),
        }
        for recent_years in [2, 3]
        for early_window in [None, (14, 26)]
        for late_window in [None, (70, 80)]
        for alpha_az_early in ([0.0] if early_window is None else [0.05, 0.1])
        for alpha_el_early in ([0.0] if early_window is None else [0.0, 0.05])
        for alpha_az_late in ([0.0] if late_window is None else [0.05, 0.1])
        for alpha_el_late in ([0.0] if late_window is None else [0.0])
        if early_window is not None or late_window is not None
    ]
)


def apply_selected_stage5_joint_surface_candidate(
    item: dict,
    train_items: list[dict],
    episode_bank: list[dict],
    selected_k: int,
    selected_temp: float,
) -> tuple[np.ndarray, dict, dict]:
    cand = predict_stage5_joint_surface_candidate(
        episode_bank=episode_bank,
        target_item=item,
        selected_k=int(selected_k),
        selected_temp=float(selected_temp),
    )
    params = fit_stage5_joint_surface_from_descriptors(
        train_items=train_items,
        block_minutes=int(cand["block_minutes"]),
        rank=int(cand["rank"]),
        ridge=float(cand["ridge"]),
    )
    surface = predict_stage5_joint_surface_for_item(item, *params)
    pred, metrics = apply_stage5_joint_surface_to_item(
        item=item,
        surface=surface,
        block_minutes=int(cand["block_minutes"]),
        az_start_day=int(cand["az_start_day"]),
        el_start_day=int(cand["el_start_day"]),
        alpha_az=float(cand["alpha_az"]),
        alpha_el=float(cand["alpha_el"]),
    )
    return pred, metrics, cand


def build_stage5_post_family_item(
    piece_item: dict,
    pred_stage5_aligned: np.ndarray,
) -> dict:
    ep = piece_item["episode"]
    fm = np.asarray(ep["forecast_mask"], dtype=bool)
    pred = np.asarray(pred_stage5_aligned, dtype=np.float64)
    residual = np.column_stack(
        [
            core.angle_diff_deg(ep["true_aligned"][fm, 3], pred[fm, 3]),
            ep["true_aligned"][fm, 4] - pred[fm, 4],
        ]
    ).astype(np.float64)
    metrics = core.compute_metrics_azel(ep["true_aligned"][fm], pred[fm])["overall"]
    return {
        "episode": ep,
        "pred_stage5_aligned": pred,
        "residual_forecast": residual,
        "stage4_descriptor": np.asarray(piece_item["stage4_descriptor"], dtype=np.float64),
        "base_metrics": metrics,
    }


def build_stage5_recent_post_overlay(
    train_items: list[dict],
    recent_years: int,
) -> np.ndarray:
    if not train_items:
        raise RuntimeError("stage5 post-family train_items are empty")
    years = [int(item["episode"]["year"]) for item in train_items]
    max_year = max(years)
    usable = [
        item
        for item in train_items
        if int(item["episode"]["year"]) >= int(max_year) - int(recent_years) + 1
    ]
    if not usable:
        usable = train_items
    return np.mean(
        np.stack([np.asarray(item["residual_forecast"], dtype=np.float64) for item in usable], axis=0),
        axis=0,
    )


def build_stage5_window_mask(count: int, window_days: Sequence[int] | None) -> np.ndarray:
    if window_days is None:
        return np.zeros(int(count), dtype=bool)
    start_day = int(window_days[0])
    end_day = int(window_days[1])
    day_idx = np.arange(int(count), dtype=np.int32) // 1440
    return (day_idx >= start_day) & (day_idx <= end_day)


def apply_stage5_post_family_to_item(
    item: dict,
    overlay_forecast: np.ndarray,
    candidate: dict,
) -> tuple[np.ndarray, dict]:
    pred = np.asarray(item["pred_stage5_aligned"], dtype=np.float64).copy()
    ep = item["episode"]
    fm = np.asarray(ep["forecast_mask"], dtype=bool)
    count = int(np.sum(fm))
    corr = np.asarray(overlay_forecast, dtype=np.float64)
    if corr.shape[0] != count:
        raise ValueError("stage5 post-family overlay length mismatch")
    az_corr = np.zeros(count, dtype=np.float64)
    el_corr = np.zeros(count, dtype=np.float64)
    early_mask = build_stage5_window_mask(count, candidate.get("early_window"))
    late_mask = build_stage5_window_mask(count, candidate.get("late_window"))
    az_corr[early_mask] += float(candidate.get("alpha_az_early", 0.0)) * corr[early_mask, 0]
    el_corr[early_mask] += float(candidate.get("alpha_el_early", 0.0)) * corr[early_mask, 1]
    az_corr[late_mask] += float(candidate.get("alpha_az_late", 0.0)) * corr[late_mask, 0]
    el_corr[late_mask] += float(candidate.get("alpha_el_late", 0.0)) * corr[late_mask, 1]
    pred[fm, 3] = core.wrap360(pred[fm, 3] - az_corr)
    pred[fm, 4] = pred[fm, 4] - el_corr
    metrics = core.compute_metrics_azel(ep["true_aligned"][fm], pred[fm])["overall"]
    return pred, metrics


def build_stage5_post_base_items_by_year(
    stage5_joint_cv: dict,
    validation_years: list[int],
) -> dict[int, list[dict]]:
    piece_by_year = {
        int(year): list(items)
        for year, items in stage5_joint_cv["piece_items_by_year"].items()
    }
    out: dict[int, list[dict]] = {}
    years = sorted(int(y) for y in validation_years if int(y) in piece_by_year)
    for year in years:
        train_items = [it for yy, items in piece_by_year.items() if int(yy) < int(year) for it in items]
        train_eps = [ep for ep in stage5_joint_cv["episode_bank"] if int(ep["year"]) < int(year)]
        targets = piece_by_year.get(int(year), [])
        if not train_items or not train_eps or not targets:
            continue
        built: list[dict] = []
        for item in targets:
            pred, _, _ = apply_selected_stage5_joint_surface_candidate(
                item=item,
                train_items=train_items,
                episode_bank=train_eps,
                selected_k=int(stage5_joint_cv["selected_k"]),
                selected_temp=float(stage5_joint_cv["selected_temp"]),
            )
            built.append(build_stage5_post_family_item(piece_item=item, pred_stage5_aligned=pred))
        if built:
            out[int(year)] = built
    return out


def choose_stage5_post_family_selector_forward_cv(
    stage5_joint_cv: dict,
    validation_years: list[int],
) -> dict:
    base_items_by_year = build_stage5_post_base_items_by_year(
        stage5_joint_cv=stage5_joint_cv,
        validation_years=validation_years,
    )
    validation_years = [int(y) for y in validation_years if int(y) in base_items_by_year]
    if not validation_years:
        raise RuntimeError("No stage5 post-family validation years could be computed")

    candidate_errors_by_year: dict[int, list[dict]] = {}
    for year in validation_years:
        train_items = [it for yy, items in base_items_by_year.items() if int(yy) < int(year) for it in items]
        targets = base_items_by_year.get(int(year), [])
        if not train_items or not targets:
            continue
        year_bank: list[dict] = []
        for item in targets:
            errs = []
            for cand in STAGE5_POST_FAMILY_CANDIDATES:
                if str(cand["family"]) == "none":
                    metrics = dict(item["base_metrics"])
                else:
                    overlay = build_stage5_recent_post_overlay(
                        train_items=train_items,
                        recent_years=int(cand["recent_years"]),
                    )
                    _, metrics = apply_stage5_post_family_to_item(
                        item=item,
                        overlay_forecast=overlay,
                        candidate=cand,
                    )
                errs.append(float(metrics["max_abs_error_max"]))
            year_bank.append(
                {
                    "year": int(year),
                    "descriptor": np.asarray(item["stage4_descriptor"], dtype=np.float64),
                    "candidate_errors": np.asarray(errs, dtype=np.float64),
                }
            )
        if year_bank:
            candidate_errors_by_year[int(year)] = year_bank

    all_eps = [ep for yy, items in candidate_errors_by_year.items() for ep in items]
    selector_mode = "forward_year_split"
    if not all_eps:
        flat_items = [it for yy, items in base_items_by_year.items() for it in items]
        if len(flat_items) < 2:
            raise RuntimeError("No stage5 post-family selector episode bank could be computed")
        selector_mode = "fallback_leave_one_out"
        fallback_eps = []
        for idx, item in enumerate(flat_items):
            train_items = [it for jj, it in enumerate(flat_items) if int(jj) != int(idx)]
            errs = []
            for cand in STAGE5_POST_FAMILY_CANDIDATES:
                if str(cand["family"]) == "none":
                    metrics = dict(item["base_metrics"])
                else:
                    overlay = build_stage5_recent_post_overlay(
                        train_items=train_items,
                        recent_years=int(cand["recent_years"]),
                    )
                    _, metrics = apply_stage5_post_family_to_item(
                        item=item,
                        overlay_forecast=overlay,
                        candidate=cand,
                    )
                errs.append(float(metrics["max_abs_error_max"]))
            fallback_eps.append(
                {
                    "year": int(item["episode"]["year"]),
                    "descriptor": np.asarray(item["stage4_descriptor"], dtype=np.float64),
                    "candidate_errors": np.asarray(errs, dtype=np.float64),
                }
            )
        all_eps = fallback_eps
    x_all = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in all_eps], axis=0)
    sigma = np.std(x_all, axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)

    scores = []
    for k in [1, 2, 3, 4, 6]:
        for temp in [0.25, 0.5, 1.0, 2.0]:
            vals = []
            if selector_mode == "forward_year_split":
                for year in validation_years:
                    train_eps = [ep for ep in all_eps if int(ep["year"]) < int(year)]
                    year_eps = [ep for ep in all_eps if int(ep["year"]) == int(year)]
                    if not train_eps or not year_eps:
                        continue
                    train_x = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in train_eps], axis=0)
                    for ep in year_eps:
                        z = np.asarray(ep["descriptor"], dtype=np.float64)
                        dist = np.sqrt(np.mean(np.square((train_x - z[None, :]) / sigma[None, :]), axis=1))
                        order = np.argsort(dist)[: min(int(k), len(train_eps))]
                        dsel = np.asarray(dist[order], dtype=np.float64) - float(np.min(dist[order]))
                        weights = np.exp(-float(temp) * dsel)
                        wsum = float(np.sum(weights))
                        if (not np.isfinite(wsum)) or wsum <= 0.0:
                            weights = np.ones_like(dsel, dtype=np.float64)
                            wsum = float(np.sum(weights))
                        weights = weights / wsum
                        err = np.zeros(len(STAGE5_POST_FAMILY_CANDIDATES), dtype=np.float64)
                        for ww, idx in zip(weights.tolist(), order.tolist()):
                            err += float(ww) * np.asarray(train_eps[int(idx)]["candidate_errors"], dtype=np.float64)
                        vals.append(float(ep["candidate_errors"][int(np.argmin(err))]))
            else:
                for i, ep in enumerate(all_eps):
                    train_eps = [tr for j, tr in enumerate(all_eps) if int(j) != int(i)]
                    train_x = np.stack([np.asarray(tr["descriptor"], dtype=np.float64) for tr in train_eps], axis=0)
                    z = np.asarray(ep["descriptor"], dtype=np.float64)
                    dist = np.sqrt(np.mean(np.square((train_x - z[None, :]) / sigma[None, :]), axis=1))
                    order = np.argsort(dist)[: min(int(k), len(train_eps))]
                    dsel = np.asarray(dist[order], dtype=np.float64) - float(np.min(dist[order]))
                    weights = np.exp(-float(temp) * dsel)
                    wsum = float(np.sum(weights))
                    if (not np.isfinite(wsum)) or wsum <= 0.0:
                        weights = np.ones_like(dsel, dtype=np.float64)
                        wsum = float(np.sum(weights))
                    weights = weights / wsum
                    err = np.zeros(len(STAGE5_POST_FAMILY_CANDIDATES), dtype=np.float64)
                    for ww, idx in zip(weights.tolist(), order.tolist()):
                        err += float(ww) * np.asarray(train_eps[int(idx)]["candidate_errors"], dtype=np.float64)
                    vals.append(float(ep["candidate_errors"][int(np.argmin(err))]))
            if vals:
                scores.append(
                    {
                        "k": int(k),
                        "temp": float(temp),
                        "val_max_abs_error": float(np.max(vals)),
                        "val_mean_max_abs_error": float(np.mean(vals)),
                        "count": int(len(vals)),
                    }
                )
    if not scores:
        raise RuntimeError("No stage5 post-family selector scores could be computed")
    scores.sort(key=lambda x: (x["val_max_abs_error"], x["val_mean_max_abs_error"], x["k"], x["temp"]))
    best = scores[0]
    return {
        "base_items_by_year": base_items_by_year,
        "episode_bank": all_eps,
        "selector_mode": str(selector_mode),
        "selected_k": int(best["k"]),
        "selected_temp": float(best["temp"]),
        "candidate_space": list(STAGE5_POST_FAMILY_CANDIDATES),
        "scores": scores,
    }


def predict_stage5_post_family_candidate(
    episode_bank: list[dict],
    target_item: dict,
    selected_k: int,
    selected_temp: float,
) -> dict:
    x_all = np.stack([np.asarray(ep["descriptor"], dtype=np.float64) for ep in episode_bank], axis=0)
    sigma = np.std(x_all, axis=0)
    sigma = np.where(sigma < 1.0e-6, 1.0, sigma)
    z = np.asarray(target_item["stage4_descriptor"], dtype=np.float64)
    dist = np.sqrt(np.mean(np.square((x_all - z[None, :]) / sigma[None, :]), axis=1))
    order = np.argsort(dist)[: min(int(selected_k), len(episode_bank))]
    dsel = np.asarray(dist[order], dtype=np.float64) - float(np.min(dist[order]))
    weights = np.exp(-float(selected_temp) * dsel)
    wsum = float(np.sum(weights))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        weights = np.ones_like(dsel, dtype=np.float64)
        wsum = float(np.sum(weights))
    weights = weights / wsum
    err = np.zeros(len(STAGE5_POST_FAMILY_CANDIDATES), dtype=np.float64)
    for ww, idx in zip(weights.tolist(), order.tolist()):
        err += float(ww) * np.asarray(episode_bank[int(idx)]["candidate_errors"], dtype=np.float64)
    best_idx = int(np.argmin(err))
    return dict(STAGE5_POST_FAMILY_CANDIDATES[best_idx])


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

    log("Fitting single-TLE weighted 90d projection model...")
    pred_proj_azel_full, proj_payload = run_single_tle_weighted_projection_model(
        window=window,
        raw_azel=tle_azel,
        target_mode=str(args.single_tle_proj_target_mode),
        sidereal_harmonics=int(args.single_tle_proj_sidereal_harmonics),
        solar_harmonics=int(args.single_tle_proj_solar_harmonics),
        slow_harmonics=int(args.single_tle_proj_slow_harmonics),
        tau_days=float(args.single_tle_proj_tau_days),
        ridge=float(args.single_tle_proj_ridge),
    )
    proj_target_mode = str(proj_payload["projection_config"]["target_mode"])
    proj_coef = np.asarray(proj_payload["weights"], dtype=np.float64)
    pred_proj_lla = core.azel_to_lla_geoshell(
        az_deg=pred_proj_azel_full[:, 0],
        el_deg=pred_proj_azel_full[:, 1],
        observer_lat_deg=float(args.observer_lat),
        observer_lon_deg=float(args.observer_lon),
        observer_alt_m=float(args.observer_alt_m),
        geo_radius_km=float(args.geo_radius_km),
    )
    proj_dir = out_root / "tf_single_tle_full90_weighted_projection"
    proj_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        proj_dir / "single_tle_full90_weighted_projection_model.npz",
        weights=proj_coef,
        sidereal_harmonics=int(proj_payload["projection_config"]["sidereal_harmonics"]),
        solar_harmonics=int(proj_payload["projection_config"]["solar_harmonics"]),
        slow_harmonics=int(proj_payload["projection_config"]["slow_harmonics"]),
        tau_days=float(proj_payload["projection_config"]["tau_days"]),
        ridge=float(proj_payload["projection_config"]["ridge"]),
    )
    proj_teacher_metrics = core.compute_metrics_azel(
        np.column_stack([tle_lla, tle_azel]),
        np.column_stack([pred_proj_lla, pred_proj_azel_full]),
    )
    (proj_dir / "history.json").write_text(
        json.dumps(
            {
                "fit_method": "single_tle_weighted_ridge_projection",
                "feature_dim": int(proj_payload["feature_dim"]),
                "projection_config": dict(proj_payload["projection_config"]),
                "teacher_fit_metrics": proj_teacher_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    single_tle_proj_metrics = evaluate_and_save(
        name="tf_single_tle_full90_weighted_projection",
        out_dir=proj_dir,
        dates=window.dates,
        pred_unix=window.unix,
        pred_lla=pred_proj_lla,
        pred_azel=pred_proj_azel_full,
        truth_unix=truth_unix,
        truth_full=truth_full,
        forecast_start_unix=window.train_end_unix,
        max_plot_points_per_month=int(args.max_plot_points_per_month),
        extra_json={
            "tle_file": str(tle_path),
            "train_days": int(args.train_days),
            "total_days": int(args.days),
            "fit_method": "single_tle_weighted_ridge_projection",
            "projection_config": dict(proj_payload["projection_config"]),
            "teacher_fit_metrics": proj_teacher_metrics["overall"],
        },
    )
    log(
        "Single-TLE weighted projection max abs (83d only): "
        f"{single_tle_proj_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
    )

    single_tle_dual_switch_metrics = None
    if bool(args.enable_single_tle_dual_projection_switch):
        log("Fitting single-TLE dual projection switch model...")
        pred_dual_switch_azel_full, dual_switch_payload = run_single_tle_dual_projection_switch_model(
            window=window,
            raw_azel=tle_azel,
        )
        pred_dual_switch_lla = core.azel_to_lla_geoshell(
            az_deg=pred_dual_switch_azel_full[:, 0],
            el_deg=pred_dual_switch_azel_full[:, 1],
            observer_lat_deg=float(args.observer_lat),
            observer_lon_deg=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
        )
        dual_switch_dir = out_root / "tf_single_tle_dual_projection_switch"
        dual_switch_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            dual_switch_dir / "single_tle_dual_projection_switch_model.npz",
            component_a_weights=np.asarray(dual_switch_payload["component_a"]["weights"], dtype=np.float64),
            component_b_weights=np.asarray(dual_switch_payload["component_b"]["weights"], dtype=np.float64),
            component_a_sidereal_harmonics=int(dual_switch_payload["component_a"]["projection_config"]["sidereal_harmonics"]),
            component_a_solar_harmonics=int(dual_switch_payload["component_a"]["projection_config"]["solar_harmonics"]),
            component_a_slow_harmonics=int(dual_switch_payload["component_a"]["projection_config"]["slow_harmonics"]),
            component_a_tau_days=float(dual_switch_payload["component_a"]["projection_config"]["tau_days"]),
            component_a_ridge=float(dual_switch_payload["component_a"]["projection_config"]["ridge"]),
            component_b_sidereal_harmonics=int(dual_switch_payload["component_b"]["projection_config"]["sidereal_harmonics"]),
            component_b_solar_harmonics=int(dual_switch_payload["component_b"]["projection_config"]["solar_harmonics"]),
            component_b_slow_harmonics=int(dual_switch_payload["component_b"]["projection_config"]["slow_harmonics"]),
            component_b_tau_days=float(dual_switch_payload["component_b"]["projection_config"]["tau_days"]),
            component_b_ridge=float(dual_switch_payload["component_b"]["projection_config"]["ridge"]),
            switch_start1_day=int(dual_switch_payload["switch"]["start1_day"]),
            switch_start2_day=int(dual_switch_payload["switch"]["start2_day"]),
            switch_alpha=float(dual_switch_payload["switch"]["alpha"]),
        )
        dual_switch_teacher_metrics = core.compute_metrics_azel(
            np.column_stack([tle_lla, tle_azel]),
            np.column_stack([pred_dual_switch_lla, pred_dual_switch_azel_full]),
        )
        (dual_switch_dir / "history.json").write_text(
            json.dumps(
                {
                    "fit_method": "single_tle_dual_projection_switch",
                    "component_a": {
                        "feature_dim": int(dual_switch_payload["component_a"]["feature_dim"]),
                        "projection_config": dict(dual_switch_payload["component_a"]["projection_config"]),
                    },
                    "component_b": {
                        "feature_dim": int(dual_switch_payload["component_b"]["feature_dim"]),
                        "projection_config": dict(dual_switch_payload["component_b"]["projection_config"]),
                    },
                    "switch": dict(dual_switch_payload["switch"]),
                    "teacher_fit_metrics": dual_switch_teacher_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        single_tle_dual_switch_metrics = evaluate_and_save(
            name="tf_single_tle_dual_projection_switch",
            out_dir=dual_switch_dir,
            dates=window.dates,
            pred_unix=window.unix,
            pred_lla=pred_dual_switch_lla,
            pred_azel=pred_dual_switch_azel_full,
            truth_unix=truth_unix,
            truth_full=truth_full,
            forecast_start_unix=window.train_end_unix,
            max_plot_points_per_month=int(args.max_plot_points_per_month),
            extra_json={
                "tle_file": str(tle_path),
                "train_days": int(args.train_days),
                "total_days": int(args.days),
                "fit_method": "single_tle_dual_projection_switch",
                "component_a": {
                    "feature_dim": int(dual_switch_payload["component_a"]["feature_dim"]),
                    "projection_config": dict(dual_switch_payload["component_a"]["projection_config"]),
                },
                "component_b": {
                    "feature_dim": int(dual_switch_payload["component_b"]["feature_dim"]),
                    "projection_config": dict(dual_switch_payload["component_b"]["projection_config"]),
                },
                "switch": dict(dual_switch_payload["switch"]),
                "teacher_fit_metrics": dual_switch_teacher_metrics["overall"],
            },
        )
        log(
            "Single-TLE dual projection switch max abs (83d only): "
            f"{single_tle_dual_switch_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
        )
    else:
        log("Single-TLE dual projection switch skipped by default")

    single_tle_hotspot_switch_metrics = None
    if bool(args.enable_single_tle_hotspot_window_switch):
        log("Fitting single-TLE hotspot window switch model...")
        pred_hotspot_switch_azel_full, hotspot_switch_payload = run_single_tle_hotspot_window_switch_model(
            window=window,
            raw_azel=tle_azel,
        )
        pred_hotspot_switch_lla = core.azel_to_lla_geoshell(
            az_deg=pred_hotspot_switch_azel_full[:, 0],
            el_deg=pred_hotspot_switch_azel_full[:, 1],
            observer_lat_deg=float(args.observer_lat),
            observer_lon_deg=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
        )
        hotspot_switch_dir = out_root / "tf_single_tle_hotspot_window_switch"
        hotspot_switch_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            hotspot_switch_dir / "single_tle_hotspot_window_switch_model.npz",
            base_weights=np.asarray(hotspot_switch_payload["component_base"]["weights"], dtype=np.float64),
            early_weights=np.asarray(hotspot_switch_payload["component_early"]["weights"], dtype=np.float64),
            mid_weights=np.asarray(hotspot_switch_payload["component_mid"]["weights"], dtype=np.float64),
            late_weights=np.asarray(hotspot_switch_payload["component_late"]["weights"], dtype=np.float64),
            base_sidereal_harmonics=int(hotspot_switch_payload["component_base"]["projection_config"]["sidereal_harmonics"]),
            base_solar_harmonics=int(hotspot_switch_payload["component_base"]["projection_config"]["solar_harmonics"]),
            base_slow_harmonics=int(hotspot_switch_payload["component_base"]["projection_config"]["slow_harmonics"]),
            base_tau_days=float(hotspot_switch_payload["component_base"]["projection_config"]["tau_days"]),
            base_ridge=float(hotspot_switch_payload["component_base"]["projection_config"]["ridge"]),
            early_sidereal_harmonics=int(hotspot_switch_payload["component_early"]["projection_config"]["sidereal_harmonics"]),
            early_solar_harmonics=int(hotspot_switch_payload["component_early"]["projection_config"]["solar_harmonics"]),
            early_slow_harmonics=int(hotspot_switch_payload["component_early"]["projection_config"]["slow_harmonics"]),
            early_tau_days=float(hotspot_switch_payload["component_early"]["projection_config"]["tau_days"]),
            early_ridge=float(hotspot_switch_payload["component_early"]["projection_config"]["ridge"]),
            mid_sidereal_harmonics=int(hotspot_switch_payload["component_mid"]["projection_config"]["sidereal_harmonics"]),
            mid_solar_harmonics=int(hotspot_switch_payload["component_mid"]["projection_config"]["solar_harmonics"]),
            mid_slow_harmonics=int(hotspot_switch_payload["component_mid"]["projection_config"]["slow_harmonics"]),
            mid_tau_days=float(hotspot_switch_payload["component_mid"]["projection_config"]["tau_days"]),
            mid_ridge=float(hotspot_switch_payload["component_mid"]["projection_config"]["ridge"]),
            late_sidereal_harmonics=int(hotspot_switch_payload["component_late"]["projection_config"]["sidereal_harmonics"]),
            late_solar_harmonics=int(hotspot_switch_payload["component_late"]["projection_config"]["solar_harmonics"]),
            late_slow_harmonics=int(hotspot_switch_payload["component_late"]["projection_config"]["slow_harmonics"]),
            late_tau_days=float(hotspot_switch_payload["component_late"]["projection_config"]["tau_days"]),
            late_ridge=float(hotspot_switch_payload["component_late"]["projection_config"]["ridge"]),
            windows=np.asarray(
                [
                    [
                        int(win["start_day"]),
                        int(win["end_day"]),
                    ]
                    for win in hotspot_switch_payload["windows"]
                ],
                dtype=np.int32,
            ),
        )
        hotspot_switch_teacher_metrics = core.compute_metrics_azel(
            np.column_stack([tle_lla, tle_azel]),
            np.column_stack([pred_hotspot_switch_lla, pred_hotspot_switch_azel_full]),
        )
        (hotspot_switch_dir / "history.json").write_text(
            json.dumps(
                {
                    "fit_method": "single_tle_hotspot_window_switch",
                    "component_base": {
                        "feature_dim": int(hotspot_switch_payload["component_base"]["feature_dim"]),
                        "projection_config": dict(hotspot_switch_payload["component_base"]["projection_config"]),
                    },
                    "component_early": {
                        "feature_dim": int(hotspot_switch_payload["component_early"]["feature_dim"]),
                        "projection_config": dict(hotspot_switch_payload["component_early"]["projection_config"]),
                    },
                    "component_mid": {
                        "feature_dim": int(hotspot_switch_payload["component_mid"]["feature_dim"]),
                        "projection_config": dict(hotspot_switch_payload["component_mid"]["projection_config"]),
                    },
                    "component_late": {
                        "feature_dim": int(hotspot_switch_payload["component_late"]["feature_dim"]),
                        "projection_config": dict(hotspot_switch_payload["component_late"]["projection_config"]),
                    },
                    "windows": list(hotspot_switch_payload["windows"]),
                    "teacher_fit_metrics": hotspot_switch_teacher_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        single_tle_hotspot_switch_metrics = evaluate_and_save(
            name="tf_single_tle_hotspot_window_switch",
            out_dir=hotspot_switch_dir,
            dates=window.dates,
            pred_unix=window.unix,
            pred_lla=pred_hotspot_switch_lla,
            pred_azel=pred_hotspot_switch_azel_full,
            truth_unix=truth_unix,
            truth_full=truth_full,
            forecast_start_unix=window.train_end_unix,
            max_plot_points_per_month=int(args.max_plot_points_per_month),
            extra_json={
                "tle_file": str(tle_path),
                "train_days": int(args.train_days),
                "total_days": int(args.days),
                "fit_method": "single_tle_hotspot_window_switch",
                "component_base": {
                    "feature_dim": int(hotspot_switch_payload["component_base"]["feature_dim"]),
                    "projection_config": dict(hotspot_switch_payload["component_base"]["projection_config"]),
                },
                "component_early": {
                    "feature_dim": int(hotspot_switch_payload["component_early"]["feature_dim"]),
                    "projection_config": dict(hotspot_switch_payload["component_early"]["projection_config"]),
                },
                "component_mid": {
                    "feature_dim": int(hotspot_switch_payload["component_mid"]["feature_dim"]),
                    "projection_config": dict(hotspot_switch_payload["component_mid"]["projection_config"]),
                },
                "component_late": {
                    "feature_dim": int(hotspot_switch_payload["component_late"]["feature_dim"]),
                    "projection_config": dict(hotspot_switch_payload["component_late"]["projection_config"]),
                },
                "windows": list(hotspot_switch_payload["windows"]),
                "teacher_fit_metrics": hotspot_switch_teacher_metrics["overall"],
            },
        )
        log(
            "Single-TLE hotspot window switch max abs (83d only): "
            f"{single_tle_hotspot_switch_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
        )
    else:
        log("Single-TLE hotspot window switch skipped by default")

    single_tle_six_window_blend_metrics = None
    if bool(args.enable_single_tle_six_window_blend):
        log("Fitting single-TLE six-window blend model...")
        pred_six_window_blend_azel_full, six_window_blend_payload = run_single_tle_six_window_blend_model(
            window=window,
            raw_azel=tle_azel,
        )
        pred_six_window_blend_lla = core.azel_to_lla_geoshell(
            az_deg=pred_six_window_blend_azel_full[:, 0],
            el_deg=pred_six_window_blend_azel_full[:, 1],
            observer_lat_deg=float(args.observer_lat),
            observer_lon_deg=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
        )
        six_window_blend_dir = out_root / "tf_single_tle_six_window_blend"
        six_window_blend_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            six_window_blend_dir / "single_tle_six_window_blend_model.npz",
            base_weights=np.asarray(six_window_blend_payload["component_base"]["weights"], dtype=np.float64),
            pre_weights=np.asarray(six_window_blend_payload["component_pre"]["weights"], dtype=np.float64),
            early_weights=np.asarray(six_window_blend_payload["component_early"]["weights"], dtype=np.float64),
            mid_weights=np.asarray(six_window_blend_payload["component_mid"]["weights"], dtype=np.float64),
            late_override_weights=np.asarray(six_window_blend_payload["component_late_override"]["weights"], dtype=np.float64),
            day65_blend_weights=np.asarray(six_window_blend_payload["component_day65_blend"]["weights"], dtype=np.float64),
            late_blend_weights=np.asarray(six_window_blend_payload["component_late_blend"]["weights"], dtype=np.float64),
            override_windows=np.asarray(
                [
                    [int(win["start_day"]), int(win["end_day"])]
                    for win in six_window_blend_payload["override_windows"]
                ],
                dtype=np.int32,
            ),
            blend_windows=np.asarray(
                [
                    [int(win["start_day"]), int(win["end_day"]), int(round(float(win["alpha"]) * 1000.0))]
                    for win in six_window_blend_payload["blend_windows"]
                ],
                dtype=np.int32,
            ),
        )
        six_window_blend_teacher_metrics = core.compute_metrics_azel(
            np.column_stack([tle_lla, tle_azel]),
            np.column_stack([pred_six_window_blend_lla, pred_six_window_blend_azel_full]),
        )
        (six_window_blend_dir / "history.json").write_text(
            json.dumps(
                {
                    "fit_method": "single_tle_six_window_blend",
                    "component_base": {
                        "feature_dim": int(six_window_blend_payload["component_base"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_base"]["projection_config"]),
                    },
                    "component_pre": {
                        "feature_dim": int(six_window_blend_payload["component_pre"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_pre"]["projection_config"]),
                    },
                    "component_early": {
                        "feature_dim": int(six_window_blend_payload["component_early"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_early"]["projection_config"]),
                    },
                    "component_mid": {
                        "feature_dim": int(six_window_blend_payload["component_mid"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_mid"]["projection_config"]),
                    },
                    "component_late_override": {
                        "feature_dim": int(six_window_blend_payload["component_late_override"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_late_override"]["projection_config"]),
                    },
                    "component_day65_blend": {
                        "feature_dim": int(six_window_blend_payload["component_day65_blend"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_day65_blend"]["projection_config"]),
                    },
                    "component_late_blend": {
                        "feature_dim": int(six_window_blend_payload["component_late_blend"]["feature_dim"]),
                        "projection_config": dict(six_window_blend_payload["component_late_blend"]["projection_config"]),
                    },
                    "override_windows": list(six_window_blend_payload["override_windows"]),
                    "blend_windows": list(six_window_blend_payload["blend_windows"]),
                    "teacher_fit_metrics": six_window_blend_teacher_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        single_tle_six_window_blend_metrics = evaluate_and_save(
            name="tf_single_tle_six_window_blend",
            out_dir=six_window_blend_dir,
            dates=window.dates,
            pred_unix=window.unix,
            pred_lla=pred_six_window_blend_lla,
            pred_azel=pred_six_window_blend_azel_full,
            truth_unix=truth_unix,
            truth_full=truth_full,
            forecast_start_unix=window.train_end_unix,
            max_plot_points_per_month=int(args.max_plot_points_per_month),
            extra_json={
                "tle_file": str(tle_path),
                "train_days": int(args.train_days),
                "total_days": int(args.days),
                "fit_method": "single_tle_six_window_blend",
                "component_base": {
                    "feature_dim": int(six_window_blend_payload["component_base"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_base"]["projection_config"]),
                },
                "component_pre": {
                    "feature_dim": int(six_window_blend_payload["component_pre"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_pre"]["projection_config"]),
                },
                "component_early": {
                    "feature_dim": int(six_window_blend_payload["component_early"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_early"]["projection_config"]),
                },
                "component_mid": {
                    "feature_dim": int(six_window_blend_payload["component_mid"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_mid"]["projection_config"]),
                },
                "component_late_override": {
                    "feature_dim": int(six_window_blend_payload["component_late_override"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_late_override"]["projection_config"]),
                },
                "component_day65_blend": {
                    "feature_dim": int(six_window_blend_payload["component_day65_blend"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_day65_blend"]["projection_config"]),
                },
                "component_late_blend": {
                    "feature_dim": int(six_window_blend_payload["component_late_blend"]["feature_dim"]),
                    "projection_config": dict(six_window_blend_payload["component_late_blend"]["projection_config"]),
                },
                "override_windows": list(six_window_blend_payload["override_windows"]),
                "blend_windows": list(six_window_blend_payload["blend_windows"]),
                "teacher_fit_metrics": six_window_blend_teacher_metrics["overall"],
            },
        )
        log(
            "Single-TLE six-window blend max abs (83d only): "
            f"{single_tle_six_window_blend_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
        )
    else:
        log("Single-TLE six-window blend skipped by default")

    single_tle_drift_metrics = None
    if bool(args.enable_single_tle_stationkeeping_drift):
        log("Fitting single-TLE station-keeping drift parameter model...")
        pred_drift_azel_full, drift_payload = run_single_tle_stationkeeping_drift_model(
            window=window,
            raw_azel=tle_azel,
            train_days=int(args.train_days),
            max_shift_minutes=int(args.single_tle_drift_max_shift_minutes),
            shift_degree=int(args.single_tle_drift_shift_degree),
            offset_degree=int(args.single_tle_drift_offset_degree),
            smooth_days=int(args.single_tle_drift_smooth_days),
        )
        pred_drift_lla = core.azel_to_lla_geoshell(
            az_deg=pred_drift_azel_full[:, 0],
            el_deg=pred_drift_azel_full[:, 1],
            observer_lat_deg=float(args.observer_lat),
            observer_lon_deg=float(args.observer_lon),
            observer_alt_m=float(args.observer_alt_m),
            geo_radius_km=float(args.geo_radius_km),
        )
        drift_dir = out_root / "tf_single_tle_stationkeeping_drift_params"
        drift_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            drift_dir / "single_tle_stationkeeping_drift_params_model.npz",
            template_azel=np.asarray(drift_payload["template_azel"], dtype=np.float64),
            daily_shift=np.asarray(drift_payload["daily_shift"], dtype=np.float64),
            daily_az_offset=np.asarray(drift_payload["daily_az_offset"], dtype=np.float64),
            daily_el_offset=np.asarray(drift_payload["daily_el_offset"], dtype=np.float64),
            daily_shift_fit=np.asarray(drift_payload["daily_shift_fit"], dtype=np.float64),
            daily_az_offset_fit=np.asarray(drift_payload["daily_az_offset_fit"], dtype=np.float64),
            daily_el_offset_fit=np.asarray(drift_payload["daily_el_offset_fit"], dtype=np.float64),
            max_shift_minutes=int(args.single_tle_drift_max_shift_minutes),
            shift_degree=int(args.single_tle_drift_shift_degree),
            offset_degree=int(args.single_tle_drift_offset_degree),
            smooth_days=int(args.single_tle_drift_smooth_days),
        )
        drift_teacher_metrics = core.compute_metrics_azel(
            np.column_stack([tle_lla, tle_azel]),
            np.column_stack([pred_drift_lla, pred_drift_azel_full]),
        )
        (drift_dir / "history.json").write_text(
            json.dumps(
                {
                    "fit_method": "single_tle_stationkeeping_drift_parameters",
                    "drift_config": {
                        "max_shift_minutes": int(args.single_tle_drift_max_shift_minutes),
                        "shift_degree": int(args.single_tle_drift_shift_degree),
                        "offset_degree": int(args.single_tle_drift_offset_degree),
                        "smooth_days": int(args.single_tle_drift_smooth_days),
                    },
                    "teacher_fit_metrics": drift_teacher_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        single_tle_drift_metrics = evaluate_and_save(
            name="tf_single_tle_stationkeeping_drift_params",
            out_dir=drift_dir,
            dates=window.dates,
            pred_unix=window.unix,
            pred_lla=pred_drift_lla,
            pred_azel=pred_drift_azel_full,
            truth_unix=truth_unix,
            truth_full=truth_full,
            forecast_start_unix=window.train_end_unix,
            max_plot_points_per_month=int(args.max_plot_points_per_month),
            extra_json={
                "tle_file": str(tle_path),
                "train_days": int(args.train_days),
                "total_days": int(args.days),
                "fit_method": "single_tle_stationkeeping_drift_parameters",
                "drift_config": {
                    "max_shift_minutes": int(args.single_tle_drift_max_shift_minutes),
                    "shift_degree": int(args.single_tle_drift_shift_degree),
                    "offset_degree": int(args.single_tle_drift_offset_degree),
                    "smooth_days": int(args.single_tle_drift_smooth_days),
                },
                "teacher_fit_metrics": drift_teacher_metrics["overall"],
            },
        )
        log(
            "Single-TLE station-keeping drift params max abs (83d only): "
            f"{single_tle_drift_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
        )
    else:
        log("Single-TLE station-keeping drift params skipped by default")

    unix_metrics = None
    if bool(args.run_unix_sincos_only):
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
                stage3_local_block_hours=int(args.stage3_local_block_hours),
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
            stage3_local_block_hours=int(args.stage3_local_block_hours),
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
                if stage3_validation_years:
                    recent_count = max(1, int(args.stage3_recent_validation_years))
                    stage3_validation_years = stage3_validation_years[-recent_count:]
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
                                build_stage4_overlay_item(
                                    episode=target["episode"],
                                    pred_stage3_aligned=hist_pred_stage3,
                                )
                            )
                        if built_stage4:
                            stage4_items_by_year[int(year)] = built_stage4

                    stage4_validation_years = [y for y in validation_years if y >= 2022]
                    if bool(args.enable_stage4_az):
                        if stage4_items_by_year and stage4_validation_years:
                            stage4_cv = choose_stage4_az_overlay_forward_cv(
                                items_by_year=stage4_items_by_year,
                                validation_years=stage4_validation_years,
                            )
                            log(
                                "Stage4 selected family/start/alpha="
                                f"{stage4_cv['selected_family']}/"
                                f"{stage4_cv['selected_start_day']}/"
                                f"{stage4_cv['selected_alpha_az']}"
                            )
                            if str(stage4_cv["selected_family"]) != "none":
                                stage4_train_items = [
                                    it for yy, items in stage4_items_by_year.items() if int(yy) < target_year for it in items
                                ]
                                target_stage4_item = build_stage4_overlay_item(
                                    episode=target_episode,
                                    pred_stage3_aligned=stage3_pred_aligned,
                                )
                                stage4_overlay = build_stage4_local_overlay(
                                    train_items=stage4_train_items,
                                    target_item=target_stage4_item,
                                    recent_years=int(stage4_cv["selected_recent_years"]),
                                    local_block_minutes=int(stage4_cv["selected_local_block_minutes"]),
                                    local_k_neighbors=int(stage4_cv["selected_local_k_neighbors"]),
                                    blend_beta=float(stage4_cv["selected_blend_beta"]),
                                    family=str(stage4_cv.get("selected_overlay_family", "mean")),
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
                        stage4_metrics = None
                        log("Stage4 AZ overlay skipped by default")

                    if bool(args.enable_stage4_piecewise_az_selector):
                        if stage4_items_by_year and stage4_validation_years:
                            stage4_piecewise_cv = choose_stage4_piecewise_az_selector_forward_cv(
                                items_by_year=stage4_items_by_year,
                                validation_years=stage4_validation_years,
                            )
                            target_stage4_piecewise_item = build_stage4_overlay_item(
                                episode=target_episode,
                                pred_stage3_aligned=stage3_pred_aligned,
                            )
                            stage4_piecewise_cand = predict_stage4_piecewise_az_selector_candidate(
                                episode_bank=list(stage4_piecewise_cv["episode_bank"]),
                                target_item=target_stage4_piecewise_item,
                                stage4_piecewise_cv=stage4_piecewise_cv,
                            )
                            stage4_piecewise_train_items = [
                                it for yy, items in stage4_items_by_year.items() if int(yy) < target_year for it in items
                            ]
                            stage4_piecewise_overlay = build_stage4_local_overlay(
                                train_items=stage4_piecewise_train_items,
                                target_item=target_stage4_piecewise_item,
                                recent_years=int(stage4_piecewise_cv["base_overlay"]["recent_years"]),
                                local_block_minutes=int(stage4_piecewise_cv["base_overlay"]["local_block_minutes"]),
                                local_k_neighbors=int(stage4_piecewise_cv["base_overlay"]["local_k_neighbors"]),
                                blend_beta=float(stage4_piecewise_cand["blend_beta"]),
                                family=str(stage4_piecewise_cv["base_overlay"]["overlay_family"]),
                            )
                            stage4_piecewise_pred_aligned, stage4_piecewise_metrics_forecast = (
                                apply_stage4_piecewise_az_overlay_to_entry(
                                    pred_stage3_aligned=stage3_pred_aligned,
                                    episode=target_episode,
                                    overlay_forecast=stage4_piecewise_overlay,
                                    start1_day=int(stage4_piecewise_cand["start1"]),
                                    start2_day=int(stage4_piecewise_cand["start2"]),
                                    alpha1=float(stage4_piecewise_cand["alpha1"]),
                                    alpha2=float(stage4_piecewise_cand["alpha2"]),
                                )
                            )
                            stage4_piecewise_azel = stage3_azel.copy()
                            stage4_piecewise_lla = stage3_lla.copy()
                            stage4_piecewise_azel[target_forecast_mask, 0] = stage4_piecewise_pred_aligned[
                                target_episode["forecast_mask"], 3
                            ]
                            stage4_piecewise_azel[target_forecast_mask, 1] = stage4_piecewise_pred_aligned[
                                target_episode["forecast_mask"], 4
                            ]
                            stage4_piecewise_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                                az_deg=stage4_piecewise_azel[target_forecast_mask, 0],
                                el_deg=stage4_piecewise_azel[target_forecast_mask, 1],
                                observer_lat_deg=float(args.observer_lat),
                                observer_lon_deg=float(args.observer_lon),
                                observer_alt_m=float(args.observer_alt_m),
                                geo_radius_km=float(args.geo_radius_km),
                            )
                            stage4_piecewise_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_piecewise_az_selector"
                            stage4_piecewise_meta = {
                                **stage3_meta,
                                "stage4_piecewise_fit_method": "descriptor_weighted_piecewise_az_selector_after_stage3",
                                "stage4_piecewise_selector_base": dict(STAGE4_PIECEWISE_SELECTOR_BASE),
                                "stage4_piecewise_cv": {
                                    "selected_k": int(stage4_piecewise_cv["selected_k"]),
                                    "scores": stage4_piecewise_cv["scores"],
                                },
                                "stage4_piecewise_selected_candidate": stage4_piecewise_cand,
                                "stage4_piecewise_forecast_metrics": stage4_piecewise_metrics_forecast,
                            }
                            stage4_piecewise_metrics = evaluate_and_save(
                                name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_piecewise_az_selector",
                                out_dir=stage4_piecewise_dir,
                                dates=window.dates,
                                pred_unix=window.unix,
                                pred_lla=stage4_piecewise_lla,
                                pred_azel=stage4_piecewise_azel,
                                truth_unix=truth_unix,
                                truth_full=truth_full,
                                forecast_start_unix=window.train_end_unix,
                                max_plot_points_per_month=int(args.max_plot_points_per_month),
                                extra_json=stage4_piecewise_meta,
                            )
                            log(
                                "Historical analog + stage2 poly + stage3 overlay + stage4 piecewise AZ selector max abs (83d only): "
                                f"{stage4_piecewise_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                            )
                        else:
                            stage4_piecewise_metrics = None
                            log("Stage4 piecewise AZ selector skipped: no stage4 items were built")
                    else:
                        stage4_piecewise_metrics = None
                        log("Stage4 piecewise AZ selector skipped by default")

                    if bool(args.enable_stage5_joint_surface_selector):
                        if stage4_items_by_year and stage4_validation_years:
                            stage5_joint_cv = choose_stage5_joint_surface_selector_forward_cv(
                                stage4_items_by_year=stage4_items_by_year,
                                validation_years=stage4_validation_years,
                            )
                            stage5_target_stage4_item = build_stage4_overlay_item(
                                episode=target_episode,
                                pred_stage3_aligned=stage3_pred_aligned,
                            )
                            stage5_target_piece_item = build_stage5_fixed_piecewise_item(
                                stage4_item=stage5_target_stage4_item,
                                train_stage4_items=[
                                    it
                                    for yy, items in stage4_items_by_year.items()
                                    if int(yy) < target_year
                                    for it in items
                                ],
                            )
                            stage5_piece_train_items = [
                                it
                                for yy, items in stage5_joint_cv["piece_items_by_year"].items()
                                if int(yy) < target_year
                                for it in items
                            ]
                            stage5_joint_pred_aligned, stage5_joint_metrics_forecast, stage5_joint_cand = (
                                apply_selected_stage5_joint_surface_candidate(
                                    item=stage5_target_piece_item,
                                    train_items=stage5_piece_train_items,
                                    episode_bank=list(stage5_joint_cv["episode_bank"]),
                                    selected_k=int(stage5_joint_cv["selected_k"]),
                                    selected_temp=float(stage5_joint_cv["selected_temp"]),
                                )
                            )
                            stage5_joint_azel = stage3_azel.copy()
                            stage5_joint_lla = stage3_lla.copy()
                            stage5_joint_azel[target_forecast_mask, 0] = stage5_joint_pred_aligned[
                                target_episode["forecast_mask"], 3
                            ]
                            stage5_joint_azel[target_forecast_mask, 1] = stage5_joint_pred_aligned[
                                target_episode["forecast_mask"], 4
                            ]
                            stage5_joint_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                                az_deg=stage5_joint_azel[target_forecast_mask, 0],
                                el_deg=stage5_joint_azel[target_forecast_mask, 1],
                                observer_lat_deg=float(args.observer_lat),
                                observer_lon_deg=float(args.observer_lon),
                                observer_alt_m=float(args.observer_alt_m),
                                geo_radius_km=float(args.geo_radius_km),
                            )
                            stage5_joint_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage5_joint_surface_selector"
                            stage5_joint_meta = {
                                **stage3_meta,
                                "stage5_joint_surface_fit_method": "selector_over_joint_dayphase_surface_after_fixed_piecewise_base",
                                "stage5_joint_surface_base_overlay": dict(STAGE5_JOINT_SURFACE_BASE),
                                "stage5_joint_surface_fixed_piecewise": dict(STAGE5_JOINT_SURFACE_FIXED_CAND),
                                "stage5_joint_surface_cv": {
                                    "selected_k": int(stage5_joint_cv["selected_k"]),
                                    "selected_temp": float(stage5_joint_cv["selected_temp"]),
                                    "scores": stage5_joint_cv["scores"],
                                },
                                "stage5_joint_surface_selected_candidate": stage5_joint_cand,
                                "stage5_joint_surface_forecast_metrics": stage5_joint_metrics_forecast,
                            }
                            stage5_joint_surface_metrics = evaluate_and_save(
                                name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage5_joint_surface_selector",
                                out_dir=stage5_joint_dir,
                                dates=window.dates,
                                pred_unix=window.unix,
                                pred_lla=stage5_joint_lla,
                                pred_azel=stage5_joint_azel,
                                truth_unix=truth_unix,
                                truth_full=truth_full,
                                forecast_start_unix=window.train_end_unix,
                                max_plot_points_per_month=int(args.max_plot_points_per_month),
                                extra_json=stage5_joint_meta,
                            )
                            log(
                                "Historical analog + stage2 poly + stage3 overlay + stage5 joint surface selector max abs (83d only): "
                                f"{stage5_joint_surface_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                            )

                            if bool(args.enable_stage5_post_family_selector):
                                stage5_post_cv = choose_stage5_post_family_selector_forward_cv(
                                    stage5_joint_cv=stage5_joint_cv,
                                    validation_years=stage4_validation_years,
                                )
                                stage5_target_post_item = build_stage5_post_family_item(
                                    piece_item=stage5_target_piece_item,
                                    pred_stage5_aligned=stage5_joint_pred_aligned,
                                )
                                stage5_post_cand = predict_stage5_post_family_candidate(
                                    episode_bank=list(stage5_post_cv["episode_bank"]),
                                    target_item=stage5_target_post_item,
                                    selected_k=int(stage5_post_cv["selected_k"]),
                                    selected_temp=float(stage5_post_cv["selected_temp"]),
                                )
                                stage5_post_train_items = [
                                    it
                                    for yy, items in stage5_post_cv["base_items_by_year"].items()
                                    if int(yy) < target_year
                                    for it in items
                                ]
                                if str(stage5_post_cand["family"]) == "none" or not stage5_post_train_items:
                                    stage5_post_pred_aligned = np.asarray(stage5_joint_pred_aligned, dtype=np.float64)
                                    stage5_post_metrics_forecast = dict(stage5_joint_metrics_forecast)
                                else:
                                    stage5_post_overlay = build_stage5_recent_post_overlay(
                                        train_items=stage5_post_train_items,
                                        recent_years=int(stage5_post_cand["recent_years"]),
                                    )
                                    stage5_post_pred_aligned, stage5_post_metrics_forecast = (
                                        apply_stage5_post_family_to_item(
                                            item=stage5_target_post_item,
                                            overlay_forecast=stage5_post_overlay,
                                            candidate=stage5_post_cand,
                                        )
                                    )
                                stage5_post_azel = stage5_joint_azel.copy()
                                stage5_post_lla = stage5_joint_lla.copy()
                                stage5_post_azel[target_forecast_mask, 0] = stage5_post_pred_aligned[
                                    target_episode["forecast_mask"], 3
                                ]
                                stage5_post_azel[target_forecast_mask, 1] = stage5_post_pred_aligned[
                                    target_episode["forecast_mask"], 4
                                ]
                                stage5_post_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                                    az_deg=stage5_post_azel[target_forecast_mask, 0],
                                    el_deg=stage5_post_azel[target_forecast_mask, 1],
                                    observer_lat_deg=float(args.observer_lat),
                                    observer_lon_deg=float(args.observer_lon),
                                    observer_alt_m=float(args.observer_alt_m),
                                    geo_radius_km=float(args.geo_radius_km),
                                )
                                stage5_post_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage5_joint_surface_post_family_selector"
                                stage5_post_meta = {
                                    **stage5_joint_meta,
                                    "stage5_post_family_fit_method": "descriptor_selected_window_family_after_stage5_joint_surface",
                                    "stage5_post_family_cv": {
                                        "selected_k": int(stage5_post_cv["selected_k"]),
                                        "selected_temp": float(stage5_post_cv["selected_temp"]),
                                        "scores": stage5_post_cv["scores"],
                                    },
                                    "stage5_post_family_selected_candidate": stage5_post_cand,
                                    "stage5_post_family_forecast_metrics": stage5_post_metrics_forecast,
                                }
                                stage5_post_family_metrics = evaluate_and_save(
                                    name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage5_joint_surface_post_family_selector",
                                    out_dir=stage5_post_dir,
                                    dates=window.dates,
                                    pred_unix=window.unix,
                                    pred_lla=stage5_post_lla,
                                    pred_azel=stage5_post_azel,
                                    truth_unix=truth_unix,
                                    truth_full=truth_full,
                                    forecast_start_unix=window.train_end_unix,
                                    max_plot_points_per_month=int(args.max_plot_points_per_month),
                                    extra_json=stage5_post_meta,
                                )
                                log(
                                    "Historical analog + stage2 poly + stage3 overlay + stage5 joint surface post-family selector max abs (83d only): "
                                    f"{stage5_post_family_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                                )
                            else:
                                stage5_post_family_metrics = None
                                log("Stage5 post-family selector skipped by default")
                        else:
                            stage5_joint_surface_metrics = None
                            stage5_post_family_metrics = None
                            log("Stage5 joint surface selector skipped: no stage4 items were built")
                    else:
                        stage5_joint_surface_metrics = None
                        stage5_post_family_metrics = None
                        log("Stage5 joint surface selector skipped by default")

                    if bool(args.enable_stage4_fixed_piecewise_az):
                        if stage4_items_by_year and stage4_validation_years:
                            stage4_fixed_piecewise_cv = choose_stage4_fixed_piecewise_az_forward_cv(
                                items_by_year=stage4_items_by_year,
                                validation_years=stage4_validation_years,
                            )
                            target_stage4_fixed_item = build_stage4_overlay_item(
                                episode=target_episode,
                                pred_stage3_aligned=stage3_pred_aligned,
                            )
                            stage4_fixed_overlay = build_stage4_local_overlay(
                                train_items=[
                                    it
                                    for yy, items in stage4_items_by_year.items()
                                    if int(yy) < target_year
                                    for it in items
                                ],
                                target_item=target_stage4_fixed_item,
                                recent_years=int(stage4_fixed_piecewise_cv["selected_recent_years"]),
                                local_block_minutes=int(stage4_fixed_piecewise_cv["selected_local_block_minutes"]),
                                local_k_neighbors=int(stage4_fixed_piecewise_cv["selected_local_k_neighbors"]),
                                blend_beta=float(stage4_fixed_piecewise_cv["selected_blend_beta"]),
                                family=str(stage4_fixed_piecewise_cv["selected_overlay_family"]),
                            )
                            stage4_fixed_pred_aligned, stage4_fixed_metrics_forecast = (
                                apply_stage4_piecewise_az_overlay_to_entry(
                                    pred_stage3_aligned=stage3_pred_aligned,
                                    episode=target_episode,
                                    overlay_forecast=stage4_fixed_overlay,
                                    start1_day=int(stage4_fixed_piecewise_cv["selected_start1"]),
                                    start2_day=int(stage4_fixed_piecewise_cv["selected_start2"]),
                                    alpha1=float(stage4_fixed_piecewise_cv["selected_alpha1"]),
                                    alpha2=float(stage4_fixed_piecewise_cv["selected_alpha2"]),
                                )
                            )
                            stage4_fixed_azel = stage3_azel.copy()
                            stage4_fixed_lla = stage3_lla.copy()
                            stage4_fixed_azel[target_forecast_mask, 0] = stage4_fixed_pred_aligned[
                                target_episode["forecast_mask"], 3
                            ]
                            stage4_fixed_azel[target_forecast_mask, 1] = stage4_fixed_pred_aligned[
                                target_episode["forecast_mask"], 4
                            ]
                            stage4_fixed_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                                az_deg=stage4_fixed_azel[target_forecast_mask, 0],
                                el_deg=stage4_fixed_azel[target_forecast_mask, 1],
                                observer_lat_deg=float(args.observer_lat),
                                observer_lon_deg=float(args.observer_lon),
                                observer_alt_m=float(args.observer_alt_m),
                                geo_radius_km=float(args.geo_radius_km),
                            )
                            stage4_fixed_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_fixed_piecewise_az"
                            stage4_fixed_meta = {
                                **stage3_meta,
                                "stage4_fixed_piecewise_fit_method": "validation_selected_fixed_piecewise_az_after_stage3",
                                "stage4_fixed_piecewise_cv": stage4_fixed_piecewise_cv,
                                "stage4_fixed_piecewise_forecast_metrics": stage4_fixed_metrics_forecast,
                            }
                            stage4_fixed_piecewise_metrics = evaluate_and_save(
                                name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_fixed_piecewise_az",
                                out_dir=stage4_fixed_dir,
                                dates=window.dates,
                                pred_unix=window.unix,
                                pred_lla=stage4_fixed_lla,
                                pred_azel=stage4_fixed_azel,
                                truth_unix=truth_unix,
                                truth_full=truth_full,
                                forecast_start_unix=window.train_end_unix,
                                max_plot_points_per_month=int(args.max_plot_points_per_month),
                                extra_json=stage4_fixed_meta,
                            )
                            log(
                                "Historical analog + stage2 poly + stage3 overlay + stage4 fixed piecewise AZ max abs (83d only): "
                                f"{stage4_fixed_piecewise_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                            )
                        else:
                            stage4_fixed_piecewise_metrics = None
                            log("Stage4 fixed piecewise AZ skipped: no stage4 items were built")
                    else:
                        stage4_fixed_piecewise_metrics = None
                        log("Stage4 fixed piecewise AZ skipped by default")

                    if bool(args.enable_stage4_lowrank_az):
                        stage4_lowrank_items_by_year: dict[int, list[dict]] = {}
                        for year in range(2020, target_year):
                            train_stage4_sources = [
                                build_stage4_source_item_from_stage3(it)
                                for yy, items in stage3_items_by_year.items()
                                if int(yy) < int(year)
                                for it in items
                            ]
                            targets = stage3_items_by_year.get(int(year), [])
                            if not train_stage4_sources or not targets:
                                continue
                            built_stage4_lowrank: list[dict] = []
                            for target in targets:
                                built_stage4_lowrank.append(
                                    apply_fixed_stage4_base_to_stage3_item(
                                        stage3_item=target,
                                        train_stage4_sources=train_stage4_sources,
                                    )
                                )
                            if built_stage4_lowrank:
                                stage4_lowrank_items_by_year[int(year)] = built_stage4_lowrank

                        stage4_lowrank_validation_years = [y for y in validation_years if y >= 2022]
                        if stage4_lowrank_items_by_year and stage4_lowrank_validation_years:
                            stage4_lowrank_cv = choose_stage4_lowrank_surface_forward_cv(
                                items_by_year=stage4_lowrank_items_by_year,
                                validation_years=stage4_lowrank_validation_years,
                            )
                            log(
                                "Stage4 lowrank selected family/start/alpha="
                                f"{stage4_lowrank_cv['selected_family']}/"
                                f"{stage4_lowrank_cv['selected_start_day']}/"
                                f"{stage4_lowrank_cv['selected_alpha']}"
                            )
                            train_stage4_sources = [
                                build_stage4_source_item_from_stage3(it)
                                for yy, items in stage3_items_by_year.items()
                                if int(yy) < target_year
                                for it in items
                            ]
                            target_stage4_lowrank_base = apply_fixed_stage4_base_to_stage3_item(
                                stage3_item=target_stage3_item,
                                train_stage4_sources=train_stage4_sources,
                            )
                            if str(stage4_lowrank_cv["selected_family"]) == "lowrank":
                                stage4_lowrank_train_items = [
                                    it
                                    for yy, items in stage4_lowrank_items_by_year.items()
                                    if int(yy) < target_year
                                    for it in items
                                ]
                                stage4_lowrank_params = fit_stage4_lowrank_surface_from_descriptors(
                                    train_items=stage4_lowrank_train_items,
                                    block_minutes=int(stage4_lowrank_cv["selected_block_minutes"]),
                                    rank=int(stage4_lowrank_cv["selected_rank"]),
                                    ridge=float(stage4_lowrank_cv["selected_ridge"]),
                                )
                                stage4_lowrank_surface = predict_stage4_lowrank_surface_for_item(
                                    item=target_stage4_lowrank_base,
                                    mean_surface=stage4_lowrank_params[0],
                                    basis=stage4_lowrank_params[1],
                                    mu=stage4_lowrank_params[2],
                                    sigma=stage4_lowrank_params[3],
                                    coef=stage4_lowrank_params[4],
                                )
                                stage4_lowrank_series = stage4_surface_to_series(
                                    surface=stage4_lowrank_surface,
                                    total_minutes=int(np.sum(target_episode["forecast_mask"])),
                                    block_minutes=int(stage4_lowrank_cv["selected_block_minutes"]),
                                )
                                stage4_lowrank_pred_aligned, stage4_lowrank_metrics_forecast = apply_stage4_lowrank_surface_to_item(
                                    item=target_stage4_lowrank_base,
                                    series=stage4_lowrank_series,
                                    alpha=float(stage4_lowrank_cv["selected_alpha"]),
                                    start_day=int(stage4_lowrank_cv["selected_start_day"]),
                                )
                            else:
                                stage4_lowrank_pred_aligned = np.asarray(
                                    target_stage4_lowrank_base["pred_stage4_aligned"],
                                    dtype=np.float64,
                                )
                                stage4_lowrank_metrics_forecast = dict(target_stage4_lowrank_base["base_metrics"])

                            stage4_lowrank_azel = stage3_azel.copy()
                            stage4_lowrank_lla = stage3_lla.copy()
                            stage4_lowrank_azel[target_forecast_mask, 0] = stage4_lowrank_pred_aligned[
                                target_episode["forecast_mask"], 3
                            ]
                            stage4_lowrank_azel[target_forecast_mask, 1] = stage4_lowrank_pred_aligned[
                                target_episode["forecast_mask"], 4
                            ]
                            stage4_lowrank_lla[target_forecast_mask] = core.azel_to_lla_geoshell(
                                az_deg=stage4_lowrank_azel[target_forecast_mask, 0],
                                el_deg=stage4_lowrank_azel[target_forecast_mask, 1],
                                observer_lat_deg=float(args.observer_lat),
                                observer_lon_deg=float(args.observer_lon),
                                observer_alt_m=float(args.observer_alt_m),
                                geo_radius_km=float(args.geo_radius_km),
                            )
                            stage4_lowrank_dir = out_root / "tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_lowrank_az"
                            stage4_lowrank_meta = {
                                **stage3_meta,
                                "stage4_lowrank_fit_method": "fixed_stage4_base_plus_lowrank_az_surface",
                                "stage4_lowrank_base": STAGE4_LOW_RANK_BASE,
                                "stage4_lowrank_cv": stage4_lowrank_cv,
                                "stage4_lowrank_forecast_metrics": stage4_lowrank_metrics_forecast,
                            }
                            stage4_lowrank_metrics = evaluate_and_save(
                                name="tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_lowrank_az",
                                out_dir=stage4_lowrank_dir,
                                dates=window.dates,
                                pred_unix=window.unix,
                                pred_lla=stage4_lowrank_lla,
                                pred_azel=stage4_lowrank_azel,
                                truth_unix=truth_unix,
                                truth_full=truth_full,
                                forecast_start_unix=window.train_end_unix,
                                max_plot_points_per_month=int(args.max_plot_points_per_month),
                                extra_json=stage4_lowrank_meta,
                            )
                            log(
                                "Historical analog + stage2 poly + stage3 overlay + stage4 lowrank AZ max abs (83d only): "
                                f"{stage4_lowrank_metrics['metrics_azel_forecast_83d']['overall']['max_abs_error_max']:.6f}"
                            )
                        else:
                            stage4_lowrank_metrics = None
                            log("Stage4 lowrank AZ surface skipped: no stage4 lowrank items were built")
                    else:
                        stage4_lowrank_metrics = None
                        log("Stage4 lowrank AZ surface skipped by default")
                else:
                    stage3_metrics = None
                    stage4_metrics = None
                    stage4_piecewise_metrics = None
                    stage5_joint_surface_metrics = None
                    stage5_post_family_metrics = None
                    stage4_fixed_piecewise_metrics = None
                    stage4_lowrank_metrics = None
                    log("Stage3 overlay skipped: no stage3 items were built")
            else:
                stage2_metrics = None
                stage3_metrics = None
                stage4_metrics = None
                stage4_piecewise_metrics = None
                stage5_joint_surface_metrics = None
                stage5_post_family_metrics = None
                stage4_fixed_piecewise_metrics = None
                stage4_lowrank_metrics = None
                log("Stage2 poly skipped: no stage2 entries were built")
        else:
            stage2_metrics = None
            stage3_metrics = None
            stage4_metrics = None
            stage4_piecewise_metrics = None
            stage5_joint_surface_metrics = None
            stage5_post_family_metrics = None
            stage4_fixed_piecewise_metrics = None
            stage4_lowrank_metrics = None
            log("Stage2 poly skipped: no stage2 candidates or validation years")
    else:
        analog_metrics = None
        stage2_metrics = None
        stage3_metrics = None
        stage4_metrics = None
        stage4_piecewise_metrics = None
        stage5_joint_surface_metrics = None
        stage5_post_family_metrics = None
        stage4_fixed_piecewise_metrics = None
        stage4_lowrank_metrics = None
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
        "tf_single_tle_full90_weighted_projection": single_tle_proj_metrics["metrics_azel_forecast_83d"]["overall"],
    }
    if single_tle_dual_switch_metrics is not None:
        comparison["tf_single_tle_dual_projection_switch"] = single_tle_dual_switch_metrics["metrics_azel_forecast_83d"]["overall"]
    if single_tle_hotspot_switch_metrics is not None:
        comparison["tf_single_tle_hotspot_window_switch"] = single_tle_hotspot_switch_metrics["metrics_azel_forecast_83d"]["overall"]
    if single_tle_six_window_blend_metrics is not None:
        comparison["tf_single_tle_six_window_blend"] = single_tle_six_window_blend_metrics["metrics_azel_forecast_83d"]["overall"]
    if single_tle_drift_metrics is not None:
        comparison["tf_single_tle_stationkeeping_drift_params"] = single_tle_drift_metrics["metrics_azel_forecast_83d"]["overall"]
    if unix_metrics is not None:
        comparison["tf_7d_train_83d_forecast_unix_sincos_only"] = unix_metrics["metrics_azel_forecast_83d"]["overall"]
    if analog_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog"] = analog_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage2_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly"] = stage2_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage3_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay"] = stage3_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage4_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_az"] = stage4_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage4_piecewise_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_piecewise_az_selector"] = stage4_piecewise_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage5_joint_surface_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage5_joint_surface_selector"] = stage5_joint_surface_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage5_post_family_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage5_joint_surface_post_family_selector"] = stage5_post_family_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage4_fixed_piecewise_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_fixed_piecewise_az"] = stage4_fixed_piecewise_metrics["metrics_azel_forecast_83d"]["overall"]
    if stage4_lowrank_metrics is not None:
        comparison["tf_7d_train_83d_forecast_historical_analog_stage2_poly_stage3_overlay_stage4_lowrank_az"] = stage4_lowrank_metrics["metrics_azel_forecast_83d"]["overall"]
    (out_root / "summary.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
