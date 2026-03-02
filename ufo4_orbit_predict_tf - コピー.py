#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UFO4 GEO orbit predictor (TensorFlow): train AZ/EL, reconstruct Lat/Lon/Alt.

Workflow:
1) Train on 2017-2023 CSVs using AZ/EL only (minute-wise, time features + GEO template residual).
2) Predict AZ/EL for 2024 minute-wise.
3) Reconstruct Lat/Lon/Alt from predicted AZ/EL with GEO-shell geometry.
4) Output prediction CSV and evaluation artifacts:
   - overall metrics (AZ/EL + optional reconstructed Lat/Lon/Alt)
   - monthly metrics
   - monthly comparison plots (not yearly-compressed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# CSV output order
FULL_TARGET_COLS = ["Lat", "Lon", "Alt", "AZ", "EL"]
AZEL_COLS = ["AZ", "EL"]
ANGLE_COLS = {"Lon", "AZ"}

# Time constants
SECONDS_PER_DAY = 86400.0
SECONDS_PER_WEEK = 7.0 * SECONDS_PER_DAY
SECONDS_PER_TROPICAL_YEAR = 365.2422 * SECONDS_PER_DAY
SECONDS_PER_SIDEREAL_DAY = 86164.0905
SECONDS_PER_MONTH = 30.4375 * SECONDS_PER_DAY
JST_OFFSET_SEC = 9.0 * 3600.0

# GEO template granularity
MINUTES_PER_DAY = 1440
MAX_DOY = 366
SLOTS_PER_YEAR = MAX_DOY * MINUTES_PER_DAY

# WGS84
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Model target: AZ residual (deg), EL residual (deg)
MODEL_TARGET_DIM = 2


@dataclass
class FeatureConfig:
    daily_harmonics: int = 16
    sidereal_harmonics: int = 14
    yearly_harmonics: int = 8
    weekly_harmonics: int = 6
    monthly_harmonics: int = 4


@dataclass
class TemplatePack:
    slot_az_deg: np.ndarray
    slot_el_deg: np.ndarray
    slot_weight: np.ndarray
    minute_az_deg: np.ndarray
    minute_el_deg: np.ndarray
    minute_weight: np.ndarray


@dataclass
class BiasCalibration:
    alpha_az: float
    alpha_el: float


def log(msg: str) -> None:
    print(msg, flush=True)


def require_tensorflow() -> None:
    if tf is None:
        raise RuntimeError(
            "TensorFlow is required but not available. "
            "Install tensorflow first (e.g. pip install tensorflow)."
        )


def parse_hidden_units(s: str) -> List[int]:
    out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("--hidden-units must contain at least one integer.")
    return out


def parse_year_from_path(path: Path) -> int:
    stem = path.name[:4]
    if stem.isdigit():
        return int(stem)
    raise ValueError(f"Cannot infer year from filename: {path.name}")


def wrap180(deg: np.ndarray) -> np.ndarray:
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def wrap360(deg: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(deg, dtype=np.float64), 360.0)


def angle_diff_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return wrap180(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64))


def load_orbit_numeric_full(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load UNIX + [Lat, Lon, Alt, AZ, EL]."""
    arr = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 4, 5, 6),
        dtype=np.float64,
    )
    arr = np.atleast_2d(arr)
    unix = arr[:, 0].astype(np.float64)
    y = arr[:, 1:6].astype(np.float64)
    return unix, y


def load_orbit_numeric_azel(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load UNIX + [AZ, EL]."""
    arr = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 5, 6),
        dtype=np.float64,
    )
    arr = np.atleast_2d(arr)
    unix = arr[:, 0].astype(np.float64)
    azel = arr[:, 1:3].astype(np.float64)
    return unix, azel


def load_template_with_dates(path: Path) -> Tuple[List[str], np.ndarray]:
    dates: List[str] = []
    unix: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {path}")
        for row in reader:
            if not row:
                continue
            dates.append(row[0])
            unix.append(float(row[1]))
    return dates, np.asarray(unix, dtype=np.float64)


def load_training_data_azel(train_files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unix_all: List[np.ndarray] = []
    azel_all: List[np.ndarray] = []
    years_all: List[np.ndarray] = []

    for p in train_files:
        year = parse_year_from_path(p)
        unix, azel = load_orbit_numeric_azel(p)
        unix_all.append(unix)
        azel_all.append(azel)
        years_all.append(np.full(unix.shape[0], year, dtype=np.int16))
        log(f"Loaded {p.name}: rows={unix.shape[0]}")

    unix_cat = np.concatenate(unix_all, axis=0)
    azel_cat = np.concatenate(azel_all, axis=0)
    years_cat = np.concatenate(years_all, axis=0)

    order = np.argsort(unix_cat)
    unix_cat = unix_cat[order]
    azel_cat = azel_cat[order]
    years_cat = years_cat[order]
    return unix_cat, azel_cat, years_cat


def build_sample_weights(years: np.ndarray, gamma: float) -> np.ndarray:
    y_min = int(np.min(years))
    y_max = int(np.max(years))
    if y_max == y_min:
        return np.ones_like(years, dtype=np.float32)
    pos = (years.astype(np.float64) - y_min) / float(y_max - y_min)
    w = np.exp(float(gamma) * pos)
    w = w / np.mean(w)
    return w.astype(np.float32)


def split_train_val(unix: np.ndarray, years: np.ndarray, val_days: int) -> Tuple[np.ndarray, np.ndarray]:
    """Temporal split: last val_days of latest year for validation."""
    latest_year = int(np.max(years))
    latest_mask = years == latest_year
    if np.any(latest_mask):
        latest_unix_max = np.max(unix[latest_mask])
        cutoff = latest_unix_max - float(val_days) * SECONDS_PER_DAY
        val_mask = latest_mask & (unix >= cutoff)
    else:
        val_mask = np.zeros_like(unix, dtype=bool)

    if int(np.sum(val_mask)) < 20000:
        n = unix.shape[0]
        n_val = max(1, int(0.05 * n))
        val_mask = np.zeros((n,), dtype=bool)
        val_mask[-n_val:] = True

    train_mask = ~val_mask
    if not np.any(train_mask):
        raise RuntimeError("Train split is empty.")
    if not np.any(val_mask):
        raise RuntimeError("Validation split is empty.")
    return train_mask, val_mask


def _local_time_fields(unix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (local seconds, doy_index[0..365], minute_of_day[0..1439])."""
    sec_local = np.asarray(np.round(unix), dtype=np.int64) + int(JST_OFFSET_SEC)
    dt_s = sec_local.astype("datetime64[s]")
    dt_d = dt_s.astype("datetime64[D]")
    year_start = dt_s.astype("datetime64[Y]")
    doy = (dt_d - year_start).astype(np.int32)
    doy = np.clip(doy, 0, MAX_DOY - 1)
    mod_sec = np.mod(sec_local, int(SECONDS_PER_DAY))
    minute = (mod_sec // 60).astype(np.int32)
    minute = np.clip(minute, 0, MINUTES_PER_DAY - 1)
    return sec_local, doy, minute


def compute_slot_index(unix: np.ndarray) -> np.ndarray:
    _, doy, minute = _local_time_fields(unix)
    return doy * MINUTES_PER_DAY + minute


def compute_month_from_unix_local(unix: np.ndarray) -> np.ndarray:
    sec_local = np.asarray(np.round(unix), dtype=np.int64) + int(JST_OFFSET_SEC)
    dt_m = sec_local.astype("datetime64[s]").astype("datetime64[M]")
    month_zero_based = dt_m.astype(np.int64) % 12
    return (month_zero_based + 1).astype(np.int32)


def build_azel_template(unix: np.ndarray, azel: np.ndarray, weight: np.ndarray) -> TemplatePack:
    """Build weighted circular template by (day-of-year, minute) + minute fallback."""
    slot_idx = compute_slot_index(unix)
    _, _, minute_idx = _local_time_fields(unix)

    w = np.asarray(weight, dtype=np.float64)
    az_rad = np.deg2rad(azel[:, 0])
    el = azel[:, 1]

    slot_sin = np.zeros((SLOTS_PER_YEAR,), dtype=np.float64)
    slot_cos = np.zeros((SLOTS_PER_YEAR,), dtype=np.float64)
    slot_el_sum = np.zeros((SLOTS_PER_YEAR,), dtype=np.float64)
    slot_w = np.zeros((SLOTS_PER_YEAR,), dtype=np.float64)

    np.add.at(slot_sin, slot_idx, w * np.sin(az_rad))
    np.add.at(slot_cos, slot_idx, w * np.cos(az_rad))
    np.add.at(slot_el_sum, slot_idx, w * el)
    np.add.at(slot_w, slot_idx, w)

    minute_sin = np.zeros((MINUTES_PER_DAY,), dtype=np.float64)
    minute_cos = np.zeros((MINUTES_PER_DAY,), dtype=np.float64)
    minute_el_sum = np.zeros((MINUTES_PER_DAY,), dtype=np.float64)
    minute_w = np.zeros((MINUTES_PER_DAY,), dtype=np.float64)

    np.add.at(minute_sin, minute_idx, w * np.sin(az_rad))
    np.add.at(minute_cos, minute_idx, w * np.cos(az_rad))
    np.add.at(minute_el_sum, minute_idx, w * el)
    np.add.at(minute_w, minute_idx, w)

    slot_az = np.zeros_like(slot_sin)
    slot_el = np.zeros_like(slot_el_sum)
    minute_az = np.zeros_like(minute_sin)
    minute_el = np.zeros_like(minute_el_sum)

    slot_valid = slot_w > 0.0
    minute_valid = minute_w > 0.0

    slot_az[slot_valid] = wrap360(np.rad2deg(np.arctan2(slot_sin[slot_valid], slot_cos[slot_valid])))
    slot_el[slot_valid] = slot_el_sum[slot_valid] / slot_w[slot_valid]

    minute_az[minute_valid] = wrap360(np.rad2deg(np.arctan2(minute_sin[minute_valid], minute_cos[minute_valid])))
    minute_el[minute_valid] = minute_el_sum[minute_valid] / minute_w[minute_valid]

    # Rare fallback if a minute bin is empty (should not happen with minute-wise yearly data)
    if not np.all(minute_valid):
        fallback_az = wrap360(np.rad2deg(np.arctan2(np.sum(minute_sin), np.sum(minute_cos))))
        fallback_el = float(np.sum(minute_el_sum) / max(np.sum(minute_w), 1e-9))
        minute_az[~minute_valid] = fallback_az
        minute_el[~minute_valid] = fallback_el

    return TemplatePack(
        slot_az_deg=slot_az,
        slot_el_deg=slot_el,
        slot_weight=slot_w,
        minute_az_deg=minute_az,
        minute_el_deg=minute_el,
        minute_weight=minute_w,
    )


def lookup_template_azel(unix: np.ndarray, tpl: TemplatePack) -> Tuple[np.ndarray, np.ndarray]:
    """Return baseline [AZ, EL] and slot weight for each unix."""
    slot_idx = compute_slot_index(unix)
    _, _, minute_idx = _local_time_fields(unix)

    az = tpl.slot_az_deg[slot_idx].copy()
    el = tpl.slot_el_deg[slot_idx].copy()
    sw = tpl.slot_weight[slot_idx].copy()

    missing = sw <= 0.0
    if np.any(missing):
        az[missing] = tpl.minute_az_deg[minute_idx[missing]]
        el[missing] = tpl.minute_el_deg[minute_idx[missing]]
        sw[missing] = tpl.minute_weight[minute_idx[missing]]

    baseline = np.column_stack([wrap360(az), el]).astype(np.float64)
    return baseline, sw.astype(np.float64)


def build_residual_targets(true_azel: np.ndarray, base_azel: np.ndarray) -> np.ndarray:
    """Residual targets in degrees: [AZ_residual_wrapped, EL_residual]."""
    az_res = angle_diff_deg(true_azel[:, 0], base_azel[:, 0])
    el_res = true_azel[:, 1] - base_azel[:, 1]
    return np.column_stack([az_res, el_res]).astype(np.float64)


def scale_targets(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(y, axis=0).astype(np.float64)
    std = np.std(y, axis=0).astype(np.float64)
    std = np.where(std < 1e-9, 1.0, std)
    y_scaled = ((y - mean) / std).astype(np.float32)
    return y_scaled, mean, std


def decode_residual_predictions(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    y = y_scaled.astype(np.float64)
    return y * std.reshape(1, -1) + mean.reshape(1, -1)


def _add_harmonics_tf(features: List[tf.Tensor], phase: tf.Tensor, harmonics: int) -> None:
    for k in range(1, int(harmonics) + 1):
        arg = phase * float(k)
        features.append(tf.sin(arg))
        features.append(tf.cos(arg))


def build_time_features_tf(unix_batch: tf.Tensor, base_unix: float, cfg: FeatureConfig) -> tf.Tensor:
    u = tf.cast(unix_batch, tf.float32)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float32)
    base_unix_tf = tf.constant(float(base_unix), dtype=tf.float32)
    jst_offset_tf = tf.constant(JST_OFFSET_SEC, dtype=tf.float32)

    t_year = (u - base_unix_tf) / tf.constant(SECONDS_PER_TROPICAL_YEAR, dtype=tf.float32)
    t2 = 0.1 * tf.square(t_year)
    t3 = 0.01 * tf.pow(t_year, 3.0)

    u_local = u + jst_offset_tf
    phase_day = two_pi * u_local / tf.constant(SECONDS_PER_DAY, dtype=tf.float32)
    phase_sidereal = two_pi * u_local / tf.constant(SECONDS_PER_SIDEREAL_DAY, dtype=tf.float32)
    phase_year = two_pi * u_local / tf.constant(SECONDS_PER_TROPICAL_YEAR, dtype=tf.float32)
    phase_week = two_pi * u_local / tf.constant(SECONDS_PER_WEEK, dtype=tf.float32)
    phase_month = two_pi * u_local / tf.constant(SECONDS_PER_MONTH, dtype=tf.float32)

    feats: List[tf.Tensor] = [t_year, t2, t3]
    _add_harmonics_tf(feats, phase_day, cfg.daily_harmonics)
    _add_harmonics_tf(feats, phase_sidereal, cfg.sidereal_harmonics)
    _add_harmonics_tf(feats, phase_year, cfg.yearly_harmonics)
    _add_harmonics_tf(feats, phase_week, cfg.weekly_harmonics)
    _add_harmonics_tf(feats, phase_month, cfg.monthly_harmonics)

    # Drift interactions
    for k in range(1, min(5, cfg.daily_harmonics + 1)):
        arg = phase_day * float(k)
        feats.append(t_year * tf.sin(arg))
        feats.append(t_year * tf.cos(arg))
    for k in range(1, min(4, cfg.sidereal_harmonics + 1)):
        arg = phase_sidereal * float(k)
        feats.append(t_year * tf.sin(arg))
        feats.append(t_year * tf.cos(arg))
    for k in range(1, min(4, cfg.yearly_harmonics + 1)):
        arg = phase_year * float(k)
        feats.append(t_year * tf.sin(arg))
        feats.append(t_year * tf.cos(arg))

    return tf.stack(feats, axis=1)


def augment_with_template_features(
    time_feats: tf.Tensor,
    base_az_batch: tf.Tensor,
    base_el_batch: tf.Tensor,
    slot_weight_batch: tf.Tensor,
) -> tf.Tensor:
    az_rad = tf.cast(base_az_batch, tf.float32) * (math.pi / 180.0)
    az_sin = tf.sin(az_rad)
    az_cos = tf.cos(az_rad)
    el_norm = tf.cast(base_el_batch, tf.float32) / 90.0
    sw_log = tf.math.log1p(tf.cast(slot_weight_batch, tf.float32))

    extras = tf.stack([az_sin, az_cos, el_norm, sw_log], axis=1)
    return tf.concat([time_feats, extras], axis=1)


def make_dataset(
    unix: np.ndarray,
    base_az: np.ndarray,
    base_el: np.ndarray,
    slot_weight: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    base_unix: float,
    cfg: FeatureConfig,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (
            unix.astype(np.float32),
            base_az.astype(np.float32),
            base_el.astype(np.float32),
            slot_weight.astype(np.float32),
            y.astype(np.float32),
            sample_weight.astype(np.float32),
        )
    )
    if shuffle:
        buffer = min(unix.shape[0], 400_000)
        ds = ds.shuffle(buffer_size=int(buffer), seed=int(seed), reshuffle_each_iteration=True)

    ds = ds.batch(int(batch_size), drop_remainder=False)

    def _mapper(u, baz, bel, sw, yy, ww):
        tfeat = build_time_features_tf(u, base_unix=base_unix, cfg=cfg)
        feat = augment_with_template_features(tfeat, baz, bel, sw)
        return feat, yy, ww

    ds = ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def make_predict_dataset(
    unix: np.ndarray,
    base_az: np.ndarray,
    base_el: np.ndarray,
    slot_weight: np.ndarray,
    base_unix: float,
    cfg: FeatureConfig,
    batch_size: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (
            unix.astype(np.float32),
            base_az.astype(np.float32),
            base_el.astype(np.float32),
            slot_weight.astype(np.float32),
        )
    )
    ds = ds.batch(int(batch_size), drop_remainder=False)

    def _mapper(u, baz, bel, sw):
        tfeat = build_time_features_tf(u, base_unix=base_unix, cfg=cfg)
        feat = augment_with_template_features(tfeat, baz, bel, sw)
        return feat

    ds = ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(input_dim: int, hidden_units: Sequence[int], dropout: float, l2_reg: float) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(float(l2_reg))
    inp = tf.keras.Input(shape=(int(input_dim),), name="features")
    x = inp
    for units in hidden_units:
        h = tf.keras.layers.Dense(int(units), kernel_initializer="he_normal", kernel_regularizer=reg)(x)
        h = tf.keras.layers.Activation("swish")(h)
        h = tf.keras.layers.Dense(int(units), kernel_initializer="he_normal", kernel_regularizer=reg)(h)
        h = tf.keras.layers.Activation("swish")(h)
        if x.shape[-1] == units:
            x = tf.keras.layers.Add()([x, h])
        else:
            x = h
        if float(dropout) > 0.0:
            x = tf.keras.layers.Dropout(float(dropout))(x)
    out = tf.keras.layers.Dense(MODEL_TARGET_DIM, name="residual_azel")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="ufo4_geo_azel_residual_model")


def make_weighted_huber_loss(dim_weights: Sequence[float], delta: float):
    w = tf.constant(np.asarray(dim_weights, dtype=np.float32), dtype=tf.float32)
    delta_tf = tf.constant(float(delta), dtype=tf.float32)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = tf.abs(y_pred - y_true)
        quad = tf.minimum(err, delta_tf)
        lin = err - quad
        huber = 0.5 * tf.square(quad) + delta_tf * lin
        return tf.reduce_mean(huber * w, axis=-1)

    return loss_fn


def make_optimizer(learning_rate: float, weight_decay: float) -> tf.keras.optimizers.Optimizer:
    if hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(learning_rate=float(learning_rate), weight_decay=float(weight_decay))
    return tf.keras.optimizers.Adam(learning_rate=float(learning_rate))


def calibrate_alpha_per_target(
    true_azel: np.ndarray,
    base_azel: np.ndarray,
    pred_residual: np.ndarray,
    max_alpha: float = 1.2,
    step: float = 0.01,
) -> BiasCalibration:
    az_true = true_azel[:, 0]
    el_true = true_azel[:, 1]
    az_base = base_azel[:, 0]
    el_base = base_azel[:, 1]
    raz = pred_residual[:, 0]
    rel = pred_residual[:, 1]

    grid = np.arange(0.0, float(max_alpha) + 1e-12, float(step), dtype=np.float64)
    if grid.size == 0:
        return BiasCalibration(alpha_az=1.0, alpha_el=1.0)

    best_az = 1.0
    best_el = 1.0
    best_rmse_az = float("inf")
    best_rmse_el = float("inf")

    for a in grid:
        az_pred = wrap360(az_base + a * raz)
        el_pred = el_base + a * rel
        err_az = angle_diff_deg(az_pred, az_true)
        err_el = el_pred - el_true
        rmse_az = float(np.sqrt(np.mean(np.square(err_az))))
        rmse_el = float(np.sqrt(np.mean(np.square(err_el))))
        if rmse_az < best_rmse_az:
            best_rmse_az = rmse_az
            best_az = float(a)
        if rmse_el < best_rmse_el:
            best_rmse_el = rmse_el
            best_el = float(a)

    return BiasCalibration(alpha_az=best_az, alpha_el=best_el)


def apply_residual_prediction(
    base_azel: np.ndarray,
    residual_pred: np.ndarray,
    calib: BiasCalibration,
) -> np.ndarray:
    az = wrap360(base_azel[:, 0] + float(calib.alpha_az) * residual_pred[:, 0])
    el = base_azel[:, 1] + float(calib.alpha_el) * residual_pred[:, 1]
    el = np.clip(el, -90.0, 90.0)
    return np.column_stack([az, el]).astype(np.float64)


def geodetic_to_ecef_km(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    alt = np.asarray(alt_km, dtype=np.float64)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


def ecef_to_geodetic_km(x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    z = np.asarray(z_km, dtype=np.float64)

    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)

    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(7):
        sin_lat = np.sin(lat)
        n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.maximum(np.cos(lat), 1e-12) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / np.maximum(n + alt, 1e-12)))

    sin_lat = np.sin(lat)
    n = WGS84_A_KM / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.maximum(np.cos(lat), 1e-12) - n

    lat_deg = np.rad2deg(lat)
    lon_deg = (np.rad2deg(lon) + 180.0) % 360.0 - 180.0
    return lat_deg, lon_deg, alt


def azel_to_lla_geoshell(
    az_deg: np.ndarray,
    el_deg: np.ndarray,
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_m: float,
    geo_radius_km: float,
) -> np.ndarray:
    """Reconstruct satellite LLA from observer AZ/EL by ray and GEO-radius intersection.

    Assumption: satellite lies on a geocentric sphere with radius geo_radius_km.
    """
    az = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    el = np.deg2rad(np.asarray(el_deg, dtype=np.float64))

    obs_alt_km = float(observer_alt_m) / 1000.0
    ox, oy, oz = geodetic_to_ecef_km(
        np.asarray([observer_lat_deg]),
        np.asarray([observer_lon_deg]),
        np.asarray([obs_alt_km]),
    )
    o = np.array([ox[0], oy[0], oz[0]], dtype=np.float64)

    lat0 = math.radians(float(observer_lat_deg))
    lon0 = math.radians(float(observer_lon_deg))
    sl = math.sin(lat0)
    cl = math.cos(lat0)
    so = math.sin(lon0)
    co = math.cos(lon0)

    # ENU unit vector from az/el
    e = np.cos(el) * np.sin(az)
    n = np.cos(el) * np.cos(az)
    u = np.sin(el)

    # ENU -> ECEF
    dx = -so * e - sl * co * n + cl * co * u
    dy = co * e - sl * so * n + cl * so * u
    dz = cl * n + sl * u

    d = np.column_stack([dx, dy, dz])

    b = d @ o
    c = float(np.dot(o, o) - geo_radius_km * geo_radius_km)
    disc = b * b - c
    disc = np.where(disc < 0.0, 0.0, disc)
    sqrt_disc = np.sqrt(disc)

    s1 = -b + sqrt_disc
    s2 = -b - sqrt_disc
    s = np.where((s1 > 0.0) & ((s1 >= s2) | (s2 <= 0.0)), s1, s2)
    s = np.where(s > 0.0, s, np.nan)

    sat = o.reshape(1, 3) + s.reshape(-1, 1) * d
    lat, lon, alt = ecef_to_geodetic_km(sat[:, 0], sat[:, 1], sat[:, 2])
    return np.column_stack([lat, lon, alt]).astype(np.float64)


def write_prediction_csv(
    path: Path,
    dates: Sequence[str],
    unix: np.ndarray,
    pred_lla: np.ndarray,
    pred_azel: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "UNIX", "Lat", "Lon", "Alt", "AZ", "EL"])
        for d, u, lla, azel in zip(dates, unix, pred_lla, pred_azel):
            writer.writerow(
                [
                    d,
                    f"{float(u):.1f}",
                    f"{float(lla[0]):.12f}",
                    f"{float(lla[1]):.12f}",
                    f"{float(lla[2]):.12f}",
                    f"{float(azel[0]):.12f}",
                    f"{float(azel[1]):.12f}",
                ]
            )


def align_by_unix(
    pred_unix: np.ndarray,
    pred_full: np.ndarray,
    true_unix: np.ndarray,
    true_full: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pu = pred_unix.astype(np.int64)
    tu = true_unix.astype(np.int64)
    common, idx_p, idx_t = np.intersect1d(pu, tu, return_indices=True)
    if common.size == 0:
        raise RuntimeError("No common UNIX timestamps between prediction and truth.")
    return pred_full[idx_p], true_full[idx_t], pred_unix[idx_p]


def _metric_block(err: np.ndarray) -> Dict[str, float]:
    ae = np.abs(err)
    return {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "max_abs_error": float(np.max(ae)),
        "bias": float(np.mean(err)),
    }


def compute_metrics_azel(y_true_full: np.ndarray, y_pred_full: np.ndarray) -> Dict[str, Dict[str, float]]:
    az_err = angle_diff_deg(y_pred_full[:, 3], y_true_full[:, 3])
    el_err = y_pred_full[:, 4] - y_true_full[:, 4]

    out: Dict[str, Dict[str, float]] = {
        "AZ": _metric_block(az_err),
        "EL": _metric_block(el_err),
    }
    out["overall"] = {
        "mae_mean": float(np.mean([out["AZ"]["mae"], out["EL"]["mae"]])),
        "rmse_mean": float(np.mean([out["AZ"]["rmse"], out["EL"]["rmse"]])),
        "max_abs_error_max": float(max(out["AZ"]["max_abs_error"], out["EL"]["max_abs_error"])),
        "bias_mean": float(np.mean([out["AZ"]["bias"], out["EL"]["bias"]])),
    }
    return out


def compute_metrics_lla(y_true_full: np.ndarray, y_pred_full: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    lat_err = y_pred_full[:, 0] - y_true_full[:, 0]
    lon_err = angle_diff_deg(y_pred_full[:, 1], y_true_full[:, 1])
    alt_err = y_pred_full[:, 2] - y_true_full[:, 2]

    out["Lat"] = _metric_block(lat_err)
    out["Lon"] = _metric_block(lon_err)
    out["Alt"] = _metric_block(alt_err)
    out["overall"] = {
        "mae_mean": float(np.mean([out["Lat"]["mae"], out["Lon"]["mae"], out["Alt"]["mae"]])),
        "rmse_mean": float(np.mean([out["Lat"]["rmse"], out["Lon"]["rmse"], out["Alt"]["rmse"]])),
        "max_abs_error_max": float(
            max(out["Lat"]["max_abs_error"], out["Lon"]["max_abs_error"], out["Alt"]["max_abs_error"])
        ),
        "bias_mean": float(np.mean([out["Lat"]["bias"], out["Lon"]["bias"], out["Alt"]["bias"]])),
    }
    return out


def compute_monthly_azel_metrics(
    unix: np.ndarray,
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    month = compute_month_from_unix_local(unix)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for m in range(1, 13):
        mask = month == m
        if not np.any(mask):
            continue
        az_err = angle_diff_deg(y_pred_full[mask, 3], y_true_full[mask, 3])
        el_err = y_pred_full[mask, 4] - y_true_full[mask, 4]
        out[f"{m:02d}"] = {
            "AZ": _metric_block(az_err),
            "EL": _metric_block(el_err),
            "rows": {"count": int(np.sum(mask))},
        }
    return out


def write_metrics_json(
    path: Path,
    rows_evaluated: int,
    calib: BiasCalibration,
    overall_azel: Dict[str, Dict[str, float]],
    monthly_azel: Dict[str, Dict[str, Dict[str, float]]],
    overall_lla: Dict[str, Dict[str, float]] | None,
) -> None:
    payload = {
        "rows_evaluated": int(rows_evaluated),
        "alpha_calibration": asdict(calib),
        "metrics_azel": overall_azel,
        "monthly_metrics_azel": monthly_azel,
        "metrics_lla_reconstructed": overall_lla,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, metrics_azel: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "mae", "rmse", "max_abs_error", "bias"])
        for col in AZEL_COLS:
            m = metrics_azel[col]
            writer.writerow([col, m["mae"], m["rmse"], m["max_abs_error"], m["bias"]])
        o = metrics_azel["overall"]
        writer.writerow(["overall_mean", o["mae_mean"], o["rmse_mean"], o["max_abs_error_max"], o["bias_mean"]])


def write_monthly_metrics_csv(path: Path, monthly: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["month", "rows", "target", "mae", "rmse", "max_abs_error", "bias"])
        for month in sorted(monthly.keys()):
            rows = int(monthly[month].get("rows", {}).get("count", 0))
            for col in AZEL_COLS:
                m = monthly[month][col]
                writer.writerow([month, rows, col, m["mae"], m["rmse"], m["max_abs_error"], m["bias"]])


def write_lla_metrics_csv(path: Path, metrics_lla: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "mae", "rmse", "max_abs_error", "bias"])
        for col in ["Lat", "Lon", "Alt"]:
            m = metrics_lla[col]
            writer.writerow([col, m["mae"], m["rmse"], m["max_abs_error"], m["bias"]])
        o = metrics_lla["overall"]
        writer.writerow(["overall_mean", o["mae_mean"], o["rmse_mean"], o["max_abs_error_max"], o["bias_mean"]])


def make_time_axis_local(unix: np.ndarray) -> np.ndarray:
    return (np.asarray(np.round(unix), dtype=np.int64) + int(JST_OFFSET_SEC)).astype("datetime64[s]")


def plot_monthly_azel(
    unix: np.ndarray,
    y_true_full: np.ndarray,
    y_pred_full: np.ndarray,
    out_dir: Path,
    max_points_per_month: int = 12000,
) -> None:
    if plt is None:
        log("matplotlib is not available; skipping monthly plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    month = compute_month_from_unix_local(unix)
    t_all = make_time_axis_local(unix)

    for m in range(1, 13):
        mask = month == m
        if not np.any(mask):
            continue

        idx = np.where(mask)[0]
        stride = max(1, int(math.ceil(idx.size / float(max_points_per_month))))
        idx_plot = idx[::stride]

        t = t_all[idx_plot]
        az_true = y_true_full[idx_plot, 3]
        az_pred = y_pred_full[idx_plot, 3]
        el_true = y_true_full[idx_plot, 4]
        el_pred = y_pred_full[idx_plot, 4]
        az_err = angle_diff_deg(az_pred, az_true)
        el_err = el_pred - el_true

        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

        axes[0].plot(t, az_true, linewidth=0.9, label="True AZ")
        axes[0].plot(t, az_pred, linewidth=0.9, label="Pred AZ")
        axes[0].set_ylabel("AZ (deg)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(t, el_true, linewidth=0.9, label="True EL")
        axes[1].plot(t, el_pred, linewidth=0.9, label="Pred EL")
        axes[1].set_ylabel("EL (deg)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        axes[2].plot(t, az_err, linewidth=0.9, label="AZ error")
        axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axes[2].set_ylabel("AZ err")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right")

        axes[3].plot(t, el_err, linewidth=0.9, label="EL error")
        axes[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axes[3].set_ylabel("EL err")
        axes[3].set_xlabel("Local time (JST)")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper right")

        fig.suptitle(f"2024-{'%02d' % m} UFO4: AZ/EL prediction vs truth")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = out_dir / f"2024_{m:02d}_az_el_compare.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TensorFlow AZ/EL predictor on 2017-2023 and predict 2024."
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=[f"{y}_calc_az_el.csv" for y in range(2017, 2024)],
        help="Training CSVs (default: 2017..2023_calc_az_el.csv).",
    )
    parser.add_argument(
        "--predict-template",
        default="2024_calc_az_el.csv",
        help="Template CSV that defines 2024 timestamps/date strings.",
    )
    parser.add_argument(
        "--truth-file",
        default="2024_calc_az_el.csv",
        help="Truth CSV for evaluation metrics/plots. Set empty to skip evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="ufo4_2024_prediction_tf",
        help="Directory for model/prediction/metrics/plots.",
    )

    # Observer and GEO shell for AZ/EL -> LLA reconstruction
    parser.add_argument("--observer-lat", type=float, default=36.3022)
    parser.add_argument("--observer-lon", type=float, default=137.9031)
    parser.add_argument("--observer-alt-m", type=float, default=0.0)
    parser.add_argument("--geo-radius-km", type=float, default=42164.0)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-7)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--l2-reg", type=float, default=5e-7)
    parser.add_argument("--hidden-units", default="512,384,256,128")
    parser.add_argument("--huber-delta", type=float, default=0.4)
    parser.add_argument("--val-days", type=int, default=120)
    parser.add_argument("--recent-weight-gamma", type=float, default=3.0)
    parser.add_argument("--fine-tune-lr-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    # Harmonics
    parser.add_argument("--daily-harmonics", type=int, default=16)
    parser.add_argument("--sidereal-harmonics", type=int, default=14)
    parser.add_argument("--yearly-harmonics", type=int, default=8)
    parser.add_argument("--weekly-harmonics", type=int, default=6)
    parser.add_argument("--monthly-harmonics", type=int, default=4)

    # Plot
    parser.add_argument("--max-plot-points-per-month", type=int, default=12000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_tensorflow()

    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    train_files = [Path(p) for p in args.train_files]
    for p in train_files:
        if not p.exists():
            raise FileNotFoundError(f"Training file not found: {p}")

    template_path = Path(args.predict_template)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    truth_path = Path(args.truth_file) if str(args.truth_file).strip() else None
    if truth_path is not None and not truth_path.exists():
        log(f"Truth file not found: {truth_path}. Evaluation will be skipped.")
        truth_path = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_plot_dir = output_dir / "monthly_plots"

    model_path = output_dir / "ufo4_azel_model_2017_2023.keras"
    meta_path = output_dir / "ufo4_azel_model_meta.json"
    pred_csv_path = output_dir / "2024_predicted_calc_az_el.csv"
    metrics_json_path = output_dir / "2024_metrics.json"
    metrics_csv_path = output_dir / "2024_metrics_az_el.csv"
    metrics_monthly_csv_path = output_dir / "2024_metrics_monthly_az_el.csv"
    metrics_lla_csv_path = output_dir / "2024_metrics_lla_reconstructed.csv"

    feat_cfg = FeatureConfig(
        daily_harmonics=int(args.daily_harmonics),
        sidereal_harmonics=int(args.sidereal_harmonics),
        yearly_harmonics=int(args.yearly_harmonics),
        weekly_harmonics=int(args.weekly_harmonics),
        monthly_harmonics=int(args.monthly_harmonics),
    )
    hidden_units = parse_hidden_units(args.hidden_units)

    log("Loading training data...")
    unix_all, azel_all, years_all = load_training_data_azel(train_files)
    log(f"Total training rows: {unix_all.shape[0]}")

    base_unix = float(np.min(unix_all))
    sample_weights_all = build_sample_weights(years_all, gamma=float(args.recent_weight_gamma))

    # GEO template from all training data (weighted, recent emphasized)
    template = build_azel_template(unix_all, azel_all, sample_weights_all)
    base_azel_all, slot_weight_all = lookup_template_azel(unix_all, template)

    y_residual_all = build_residual_targets(true_azel=azel_all, base_azel=base_azel_all)
    y_scaled_all, y_mean, y_std = scale_targets(y_residual_all)

    train_mask, val_mask = split_train_val(unix_all, years_all, val_days=int(args.val_days))
    log(f"Train rows: {int(np.sum(train_mask))}, Val rows: {int(np.sum(val_mask))}")

    ds_train = make_dataset(
        unix=unix_all[train_mask],
        base_az=base_azel_all[train_mask, 0],
        base_el=base_azel_all[train_mask, 1],
        slot_weight=slot_weight_all[train_mask],
        y=y_scaled_all[train_mask],
        sample_weight=sample_weights_all[train_mask],
        base_unix=base_unix,
        cfg=feat_cfg,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        shuffle=True,
    )
    ds_val = make_dataset(
        unix=unix_all[val_mask],
        base_az=base_azel_all[val_mask, 0],
        base_el=base_azel_all[val_mask, 1],
        slot_weight=slot_weight_all[val_mask],
        y=y_scaled_all[val_mask],
        sample_weight=sample_weights_all[val_mask],
        base_unix=base_unix,
        cfg=feat_cfg,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        shuffle=False,
    )

    sample_feat = augment_with_template_features(
        build_time_features_tf(tf.constant(unix_all[:4], dtype=tf.float32), base_unix=base_unix, cfg=feat_cfg),
        tf.constant(base_azel_all[:4, 0], dtype=tf.float32),
        tf.constant(base_azel_all[:4, 1], dtype=tf.float32),
        tf.constant(slot_weight_all[:4], dtype=tf.float32),
    )
    input_dim = int(sample_feat.shape[-1])
    log(f"Feature dimension: {input_dim}")

    model = build_model(
        input_dim=input_dim,
        hidden_units=hidden_units,
        dropout=float(args.dropout),
        l2_reg=float(args.l2_reg),
    )

    loss_fn = make_weighted_huber_loss(dim_weights=[2.0, 2.2], delta=float(args.huber_delta))
    optimizer = make_optimizer(learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay))

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
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
            min_lr=1e-6,
        ),
    ]

    log("Stage 1 training (with validation)...")
    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=int(args.epochs),
        verbose=1,
        callbacks=callbacks,
    )

    if int(args.fine_tune_epochs) > 0:
        log("Stage 2 fine-tuning on all training rows...")
        if hasattr(model.optimizer, "learning_rate"):
            new_lr = float(args.learning_rate) * float(args.fine_tune_lr_ratio)
            model.optimizer.learning_rate.assign(new_lr)
        ds_all = make_dataset(
            unix=unix_all,
            base_az=base_azel_all[:, 0],
            base_el=base_azel_all[:, 1],
            slot_weight=slot_weight_all,
            y=y_scaled_all,
            sample_weight=sample_weights_all,
            base_unix=base_unix,
            cfg=feat_cfg,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            shuffle=True,
        )
        model.fit(ds_all, epochs=int(args.fine_tune_epochs), verbose=1)

    # Calibrate residual strength on validation split
    ds_val_pred = make_predict_dataset(
        unix=unix_all[val_mask],
        base_az=base_azel_all[val_mask, 0],
        base_el=base_azel_all[val_mask, 1],
        slot_weight=slot_weight_all[val_mask],
        base_unix=base_unix,
        cfg=feat_cfg,
        batch_size=int(args.batch_size),
    )
    val_res_pred_scaled = model.predict(ds_val_pred, verbose=1)
    val_res_pred = decode_residual_predictions(val_res_pred_scaled, mean=y_mean, std=y_std)

    calib = calibrate_alpha_per_target(
        true_azel=azel_all[val_mask],
        base_azel=base_azel_all[val_mask],
        pred_residual=val_res_pred,
    )
    log(f"Calibrated alpha: AZ={calib.alpha_az:.3f}, EL={calib.alpha_el:.3f}")

    model.save(model_path)
    log(f"Saved model: {model_path}")

    model_meta = {
        "target_mode": "AZ_EL_only",
        "model_target_layout": ["AZ_residual_deg", "EL_residual_deg"],
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "alpha_calibration": asdict(calib),
        "feature_config": asdict(feat_cfg),
        "hidden_units": hidden_units,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "l2_reg": float(args.l2_reg),
        "recent_weight_gamma": float(args.recent_weight_gamma),
        "train_files": [str(p) for p in train_files],
        "base_unix": base_unix,
        "observer_lat": float(args.observer_lat),
        "observer_lon": float(args.observer_lon),
        "observer_alt_m": float(args.observer_alt_m),
        "geo_radius_km": float(args.geo_radius_km),
    }
    meta_path.write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log("Loading 2024 template and running prediction...")
    pred_dates, pred_unix = load_template_with_dates(template_path)
    pred_base_azel, pred_slot_weight = lookup_template_azel(pred_unix, template)

    ds_pred = make_predict_dataset(
        unix=pred_unix,
        base_az=pred_base_azel[:, 0],
        base_el=pred_base_azel[:, 1],
        slot_weight=pred_slot_weight,
        base_unix=base_unix,
        cfg=feat_cfg,
        batch_size=int(args.batch_size),
    )
    pred_res_scaled = model.predict(ds_pred, verbose=1)
    pred_res = decode_residual_predictions(pred_res_scaled, mean=y_mean, std=y_std)
    pred_azel = apply_residual_prediction(base_azel=pred_base_azel, residual_pred=pred_res, calib=calib)

    pred_lla = azel_to_lla_geoshell(
        az_deg=pred_azel[:, 0],
        el_deg=pred_azel[:, 1],
        observer_lat_deg=float(args.observer_lat),
        observer_lon_deg=float(args.observer_lon),
        observer_alt_m=float(args.observer_alt_m),
        geo_radius_km=float(args.geo_radius_km),
    )

    pred_full = np.column_stack([pred_lla, pred_azel]).astype(np.float64)
    write_prediction_csv(pred_csv_path, pred_dates, pred_unix, pred_lla=pred_lla, pred_azel=pred_azel)
    log(f"Saved predicted CSV: {pred_csv_path}")

    if truth_path is None:
        log("Truth file is not available. Skipping evaluation.")
        return

    log("Evaluating against 2024 truth and creating monthly plots...")
    true_unix, true_full = load_orbit_numeric_full(truth_path)
    pred_aligned, true_aligned, unix_aligned = align_by_unix(
        pred_unix=pred_unix,
        pred_full=pred_full,
        true_unix=true_unix,
        true_full=true_full,
    )

    metrics_azel = compute_metrics_azel(y_true_full=true_aligned, y_pred_full=pred_aligned)
    monthly_azel = compute_monthly_azel_metrics(unix=unix_aligned, y_true_full=true_aligned, y_pred_full=pred_aligned)

    # Reconstructed LLA quality is auxiliary diagnostic (depends on GEO-shell assumption)
    metrics_lla = compute_metrics_lla(y_true_full=true_aligned, y_pred_full=pred_aligned)

    write_metrics_json(
        metrics_json_path,
        rows_evaluated=true_aligned.shape[0],
        calib=calib,
        overall_azel=metrics_azel,
        monthly_azel=monthly_azel,
        overall_lla=metrics_lla,
    )
    write_metrics_csv(metrics_csv_path, metrics_azel)
    write_monthly_metrics_csv(metrics_monthly_csv_path, monthly_azel)
    write_lla_metrics_csv(metrics_lla_csv_path, metrics_lla)

    plot_monthly_azel(
        unix=unix_aligned,
        y_true_full=true_aligned,
        y_pred_full=pred_aligned,
        out_dir=monthly_plot_dir,
        max_points_per_month=int(args.max_plot_points_per_month),
    )

    log(f"Saved metrics JSON           : {metrics_json_path}")
    log(f"Saved metrics CSV (AZ/EL)    : {metrics_csv_path}")
    log(f"Saved monthly metrics CSV    : {metrics_monthly_csv_path}")
    log(f"Saved reconstructed LLA CSV  : {metrics_lla_csv_path}")
    log(f"Saved monthly plots directory: {monthly_plot_dir}")
    log("Done.")


if __name__ == "__main__":
    main()
