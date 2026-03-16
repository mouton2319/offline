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


def parse_int_csv(text: str) -> list[int]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
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


def load_truth_cache(years: Sequence[int], suffix: str = "") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in years:
        p = ROOT / f"{int(year)}_calc_az_el{suffix}.csv"
        if p.exists():
            cache[int(year)] = core.load_orbit_numeric_full(p)
    return cache


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
    warmup_weight: float,
    forecast_weight: float,
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
        *extra_periodic,
        d_az.astype(np.float32).reshape(-1, 1),
        d_el.astype(np.float32).reshape(-1, 1),
        t_rel.astype(np.float32).reshape(-1, 1),
        forecast_flag,
        static.astype(np.float32),
    ]
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
    out = tf.keras.layers.Add(name="trig_plus_delta")([base_trig, delta])
    out = tf.keras.layers.Lambda(normalize_trig4_seq_tf, name="norm_trig4")(out)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss=tf.keras.losses.Huber(delta=0.03),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def predict_full_trig4(
    model,
    x_norm: np.ndarray,
    seq_len: int,
    seq_stride: int,
    predict_batch_size: int,
) -> np.ndarray:
    n = int(x_norm.shape[0])
    L = int(seq_len)
    S = int(max(1, seq_stride))
    if n < L:
        pad = np.zeros((L, x_norm.shape[1]), dtype=np.float32)
        pad[:n] = np.asarray(x_norm, dtype=np.float32)
        pred = np.asarray(model.predict(pad[None, ...], verbose=0)[0], dtype=np.float64)
        return hm.normalize_trig4_np(pred[:n]).astype(np.float32)
    starts = list(range(0, n - L + 1, S))
    tail = n - L
    if starts[-1] != tail:
        starts.append(tail)
    acc = np.zeros((n, 4), dtype=np.float64)
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
    return hm.normalize_trig4_np(out).astype(np.float32)


def compute_forecast_max_abs_error(true_azel: np.ndarray, pred_azel: np.ndarray, forecast_mask: np.ndarray) -> float:
    t = np.zeros((int(np.sum(forecast_mask)), 5), dtype=np.float64)
    p = np.zeros((int(np.sum(forecast_mask)), 5), dtype=np.float64)
    t[:, 3:5] = np.asarray(true_azel, dtype=np.float64)[np.asarray(forecast_mask, dtype=bool)]
    p[:, 3:5] = np.asarray(pred_azel, dtype=np.float64)[np.asarray(forecast_mask, dtype=bool)]
    return float(core.compute_metrics_azel(t, p)["overall"]["max_abs_error_max"])


def blend_azel_unitvec(base_azel: np.ndarray, alt_azel: np.ndarray, alpha: float) -> np.ndarray:
    ua = exp.build_unit_vector_targets_from_azel(np.asarray(base_azel, dtype=np.float64))
    ub = exp.build_unit_vector_targets_from_azel(np.asarray(alt_azel, dtype=np.float64))
    return exp.decode_unit_vector_targets_to_azel((1.0 - float(alpha)) * ua + float(alpha) * ub)


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
    t.add_argument("--add-baseline-periodic-harmonics", type=int, default=0)
    t.add_argument("--warmup-weight", type=float, default=0.20)
    t.add_argument("--forecast-weight", type=float, default=1.00)
    t.add_argument("--lstm-units-1", type=int, default=96)
    t.add_argument("--lstm-units-2", type=int, default=48)
    t.add_argument("--dense-units", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.10)
    t.add_argument("--delta-scale", type=float, default=0.35)
    t.add_argument("--lr", type=float, default=1.0e-3)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=64)
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
    e.add_argument("--tle-dir", default="pred_tle")
    e.add_argument("--max-files", type=int, default=0)
    e.add_argument("--specific-files", default="")
    e.add_argument("--output-json", default=".tmp/eval_single_tle_lstm_geo_model.json")

    s = sub.add_parser("predict", help="Predict one TLE file")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--tle-file", required=True)
    s.add_argument("--sat-name", default="23467")
    s.add_argument("--observer-lat", type=float, default=36.3022)
    s.add_argument("--observer-lon", type=float, default=137.9031)
    s.add_argument("--days", type=int, default=90)
    s.add_argument("--train-days", type=int, default=7)
    s.add_argument("--step-minutes", type=int, default=10)
    s.add_argument("--predict-batch-size", type=int, default=128)
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
                warmup_weight=float(args.warmup_weight),
                forecast_weight=float(args.forecast_weight),
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
                warmup_weight=float(args.warmup_weight),
                forecast_weight=float(args.forecast_weight),
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
        y=y_train,
        sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
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
        "add_baseline_periodic_harmonics": int(args.add_baseline_periodic_harmonics),
        "time_feat_dim": int(time_feat_dim),
        "trig_offset": int(trig_offset),
        "warmup_weight": float(args.warmup_weight),
        "forecast_weight": float(args.forecast_weight),
        "lstm_units_1": int(args.lstm_units_1),
        "lstm_units_2": int(args.lstm_units_2),
        "dense_units": int(args.dense_units),
        "dropout": float(args.dropout),
        "delta_scale": float(args.delta_scale),
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

    val_rows = []
    val_cache = []
    for ep, (x, _, _) in zip(val_eps, val_steps):
        xn = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
        pred_trig4 = predict_full_trig4(
            model=model,
            x_norm=xn,
            seq_len=int(args.seq_len),
            seq_stride=int(args.seq_stride),
            predict_batch_size=int(args.predict_batch_size),
        )
        pred_azel = hm.trig4_to_azel(pred_trig4)
        err_raw = compute_forecast_max_abs_error(ep.true_azel, pred_azel, ep.forecast_mask)
        base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
        val_cache.append((ep, pred_azel, float(base_err), float(err_raw)))
        val_rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "lstm_raw_max_abs_error_max": float(err_raw),
            }
        )

    alpha_grid = [i / 20.0 for i in range(21)]
    alpha_scores = []
    for alpha in alpha_grid:
        errs = []
        for ep, pred_azel, _, _ in val_cache:
            blended = blend_azel_unitvec(ep.baseline_azel, pred_azel, alpha=float(alpha))
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

    val_rows = []
    for ep, pred_azel, base_err, err_raw in val_cache:
        blended = blend_azel_unitvec(ep.baseline_azel, pred_azel, alpha=best_alpha)
        err_blended = compute_forecast_max_abs_error(ep.true_azel, blended, ep.forecast_mask)
        val_rows.append(
            {
                "tle_name": ep.tle_name,
                "baseline_max_abs_error_max": float(base_err),
                "lstm_raw_max_abs_error_max": float(err_raw),
                "lstm_blended_max_abs_error_max": float(err_blended),
            }
        )
    val_rows.sort(key=lambda r: r["lstm_blended_max_abs_error_max"], reverse=True)
    val_summary = {
        "baseline_summary": summarize([float(r["baseline_max_abs_error_max"]) for r in val_rows]),
        "lstm_raw_summary": summarize([float(r["lstm_raw_max_abs_error_max"]) for r in val_rows]),
        "lstm_blended_summary": summarize([float(r["lstm_blended_max_abs_error_max"]) for r in val_rows]),
        "selected_alpha": float(best_alpha),
        "alpha_search_top5": alpha_scores[:5],
        "rows": val_rows,
    }
    (out_dir / "validation_rows.json").write_text(json.dumps(val_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["post_blend_alpha"] = float(best_alpha)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(val_summary["lstm_blended_summary"], ensure_ascii=False, indent=2))


def load_model_and_meta(model_dir: Path):
    core.require_tensorflow()
    tf = core.tf
    model = tf.keras.models.load_model(
        model_dir / "model.keras",
        compile=False,
        custom_objects={"normalize_trig4_seq_tf": normalize_trig4_seq_tf},
    )
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    x_mean = np.asarray(meta["x_mean"], dtype=np.float32)
    x_std = np.asarray(meta["x_std"], dtype=np.float32)
    return model, meta, x_mean, x_std


def predict_episode_with_model(
    model,
    meta: dict,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    ep: Episode,
    warmup_weight: float,
    forecast_weight: float,
) -> tuple[np.ndarray, float, float]:
    x, _, _ = build_step_arrays(
        ep=ep,
        days=int(meta["days"]),
        time_yearly_harmonics=int(meta["time_yearly_harmonics"]),
        time_feature_mode=str(meta.get("time_feature_mode", "periodic")),
        add_baseline_periodic_harmonics=int(meta.get("add_baseline_periodic_harmonics", 0)),
        warmup_weight=float(warmup_weight),
        forecast_weight=float(forecast_weight),
    )
    xn = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    pred_trig4 = predict_full_trig4(
        model=model,
        x_norm=xn,
        seq_len=int(meta["seq_len"]),
        seq_stride=int(meta["seq_stride"]),
        predict_batch_size=int(meta.get("predict_batch_size", 128)),
    )
    pred_azel_raw = hm.trig4_to_azel(pred_trig4)
    alpha = float(meta.get("post_blend_alpha", 1.0))
    pred_azel = blend_azel_unitvec(ep.baseline_azel, pred_azel_raw, alpha=alpha)
    lstm_err = compute_forecast_max_abs_error(ep.true_azel, pred_azel, ep.forecast_mask)
    base_err = compute_forecast_max_abs_error(ep.true_azel, ep.baseline_azel, ep.forecast_mask)
    return pred_azel, float(base_err), float(lstm_err)


def run_eval_dir(args: argparse.Namespace) -> None:
    model_dir = ROOT / str(args.model_dir)
    model, meta, x_mean, x_std = load_model_and_meta(model_dir)

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


def run_predict(args: argparse.Namespace) -> None:
    model_dir = ROOT / str(args.model_dir)
    model, meta, x_mean, x_std = load_model_and_meta(model_dir)
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
    raw_azel, _ = exp.propagate_tle_azel_lla_at_unix(
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
        add_baseline_periodic_harmonics=int(meta.get("add_baseline_periodic_harmonics", 0)),
        warmup_weight=float(meta["warmup_weight"]),
        forecast_weight=float(meta["forecast_weight"]),
    )
    xn = ((x - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
    pred_trig4 = predict_full_trig4(
        model=model,
        x_norm=xn,
        seq_len=int(meta["seq_len"]),
        seq_stride=int(meta["seq_stride"]),
        predict_batch_size=int(args.predict_batch_size),
    )
    pred_azel = hm.trig4_to_azel(pred_trig4)

    out = ROOT / str(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["unix,az_deg,el_deg"]
    for u, a in zip(window.unix.tolist(), pred_azel.tolist()):
        lines.append(f"{int(round(float(u)))},{float(a[0]):.10f},{float(a[1]):.10f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if truth_cache:
        truth_unix, truth_full = pick_truth_window(
            cache=truth_cache,
            start_unix=window.start_unix,
            end_unix=window.end_unix,
        )
        if truth_unix is not None and truth_full is not None:
            pred_full = np.zeros((window.unix.shape[0], 5), dtype=np.float64)
            pred_full[:, 3:5] = pred_azel
            pred_aligned, true_aligned, unix_aligned = core.align_by_unix(window.unix, pred_full, truth_unix, truth_full)
            forecast_mask = np.asarray(unix_aligned >= float(window.train_end_unix), dtype=bool)
            err = compute_forecast_max_abs_error(true_aligned[:, 3:5], pred_aligned[:, 3:5], forecast_mask)
            print(json.dumps({"tle_file": tle_path.name, "forecast_max_abs_error_max": float(err)}, ensure_ascii=False, indent=2))
            return
    print(json.dumps({"tle_file": tle_path.name, "output_csv": str(out)}, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "eval-dir":
        run_eval_dir(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
