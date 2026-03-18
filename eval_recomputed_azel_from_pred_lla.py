#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import ufo4_orbit_predict_tf as core


def _pick_key(fieldnames: list[str], candidates: list[str]) -> str | None:
    lut = {str(k).strip().lower(): str(k) for k in fieldnames}
    for c in candidates:
        k = lut.get(str(c).strip().lower())
        if k is not None:
            return k
    return None


def _as_float(v: str) -> float:
    return float(str(v).strip())


def load_pred_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError(f"No header in prediction csv: {path}")
        fields = [str(x) for x in r.fieldnames]
        k_unix = _pick_key(fields, ["unix", "UNIX"])
        k_lat = _pick_key(fields, ["lat_deg", "Lat", "lat"])
        k_lon = _pick_key(fields, ["lon_deg", "Lon", "lon"])
        k_alt = _pick_key(fields, ["alt_km", "Alt", "alt"])
        k_az = _pick_key(fields, ["az_deg", "AZ", "az"])
        k_el = _pick_key(fields, ["el_deg", "EL", "el"])
        if k_unix is None or k_lat is None or k_lon is None or k_alt is None:
            raise RuntimeError(
                "Prediction csv must contain unix + lat/lon/alt columns. "
                f"found={fields}"
            )

        unix_vals: list[float] = []
        lla_vals: list[list[float]] = []
        azel_vals: list[list[float]] = []
        has_azel = k_az is not None and k_el is not None
        for row in r:
            try:
                u = _as_float(row[k_unix])
                lat = _as_float(row[k_lat])
                lon = _as_float(row[k_lon])
                alt = _as_float(row[k_alt])
            except Exception:
                continue
            unix_vals.append(u)
            lla_vals.append([lat, lon, alt])
            if has_azel:
                try:
                    az = _as_float(row[k_az])
                    el = _as_float(row[k_el])
                except Exception:
                    az = float("nan")
                    el = float("nan")
                azel_vals.append([az, el])

    if not unix_vals:
        raise RuntimeError(f"No numeric rows in prediction csv: {path}")

    unix = np.asarray(unix_vals, dtype=np.float64)
    lla = np.asarray(lla_vals, dtype=np.float64)
    azel = np.asarray(azel_vals, dtype=np.float64) if has_azel else None

    order = np.argsort(unix, kind="mergesort")
    unix = unix[order]
    lla = lla[order]
    if azel is not None:
        azel = azel[order]

    uniq_unix, uniq_idx = np.unique(unix, return_index=True)
    unix = np.asarray(uniq_unix, dtype=np.float64)
    lla = np.asarray(lla[uniq_idx], dtype=np.float64)
    if azel is not None:
        azel = np.asarray(azel[uniq_idx], dtype=np.float64)
    return unix, lla, azel


def recompute_azel_from_lla(
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


def make_time_axis_local(unix: np.ndarray) -> np.ndarray:
    return (np.asarray(np.round(unix), dtype=np.int64) + int(core.JST_OFFSET_SEC)).astype("datetime64[s]")


def plot_compare_and_error(
    unix: np.ndarray,
    true_azel: np.ndarray,
    rec_azel: np.ndarray,
    raw_azel: np.ndarray | None,
    out_compare: Path,
    out_error: Path,
    max_points: int,
) -> list[Path]:
    if plt is None:
        return []
    n = int(unix.shape[0])
    if n <= 0:
        return []
    stride = int(max(1, math.ceil(float(n) / float(max(1, int(max_points))))))
    idx = np.arange(0, n, stride, dtype=np.int64)
    t = make_time_axis_local(unix[idx])
    truth = np.asarray(true_azel[idx], dtype=np.float64)
    rec = np.asarray(rec_azel[idx], dtype=np.float64)
    raw = np.asarray(raw_azel[idx], dtype=np.float64) if raw_azel is not None else None

    rec_az_err = np.asarray(core.angle_diff_deg(rec[:, 0], truth[:, 0]), dtype=np.float64)
    rec_el_err = np.asarray(rec[:, 1] - truth[:, 1], dtype=np.float64)
    if raw is not None:
        raw_az_err = np.asarray(core.angle_diff_deg(raw[:, 0], truth[:, 0]), dtype=np.float64)
        raw_el_err = np.asarray(raw[:, 1] - truth[:, 1], dtype=np.float64)
    else:
        raw_az_err = None
        raw_el_err = None

    fig_c, axes_c = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes_c[0].plot(t, truth[:, 0], linewidth=0.9, label="True AZ")
    axes_c[0].plot(t, rec[:, 0], linewidth=0.9, label="Recomputed AZ")
    if raw is not None:
        axes_c[0].plot(t, raw[:, 0], linewidth=0.9, label="Pred AZ(raw)")
    axes_c[0].set_ylabel("AZ (deg)")
    axes_c[0].grid(True, alpha=0.3)
    axes_c[0].legend(loc="upper right")

    axes_c[1].plot(t, truth[:, 1], linewidth=0.9, label="True EL")
    axes_c[1].plot(t, rec[:, 1], linewidth=0.9, label="Recomputed EL")
    if raw is not None:
        axes_c[1].plot(t, raw[:, 1], linewidth=0.9, label="Pred EL(raw)")
    axes_c[1].set_ylabel("EL (deg)")
    axes_c[1].set_xlabel("Local time (JST)")
    axes_c[1].grid(True, alpha=0.3)
    axes_c[1].legend(loc="upper right")

    fig_c.suptitle("AZ/EL: Truth vs Recomputed-from-LLA")
    fig_c.tight_layout(rect=[0, 0, 1, 0.98])
    out_compare.parent.mkdir(parents=True, exist_ok=True)
    fig_c.savefig(out_compare, dpi=160)
    plt.close(fig_c)

    fig_e, axes_e = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes_e[0].plot(t, rec_az_err, linewidth=0.9, label="AZ err (recomputed)")
    if raw_az_err is not None:
        axes_e[0].plot(t, raw_az_err, linewidth=0.9, label="AZ err (raw)")
    axes_e[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes_e[0].set_ylabel("AZ err (deg)")
    axes_e[0].grid(True, alpha=0.3)
    axes_e[0].legend(loc="upper right")

    axes_e[1].plot(t, rec_el_err, linewidth=0.9, label="EL err (recomputed)")
    if raw_el_err is not None:
        axes_e[1].plot(t, raw_el_err, linewidth=0.9, label="EL err (raw)")
    axes_e[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes_e[1].set_ylabel("EL err (deg)")
    axes_e[1].set_xlabel("Local time (JST)")
    axes_e[1].grid(True, alpha=0.3)
    axes_e[1].legend(loc="upper right")

    fig_e.suptitle("AZ/EL Error vs Truth")
    fig_e.tight_layout(rect=[0, 0, 1, 0.98])
    out_error.parent.mkdir(parents=True, exist_ok=True)
    fig_e.savefig(out_error, dpi=160)
    plt.close(fig_e)
    return [out_compare, out_error]


def write_aligned_csv(
    path: Path,
    unix: np.ndarray,
    true_azel: np.ndarray,
    rec_azel: np.ndarray,
    raw_azel: np.ndarray | None,
) -> None:
    rows = []
    for i in range(int(unix.shape[0])):
        u = int(round(float(unix[i])))
        az_t = float(true_azel[i, 0])
        el_t = float(true_azel[i, 1])
        az_r = float(rec_azel[i, 0])
        el_r = float(rec_azel[i, 1])
        az_err_r = float(core.angle_diff_deg(np.asarray([az_r]), np.asarray([az_t]))[0])
        el_err_r = float(el_r - el_t)
        row = [
            str(u),
            f"{az_t:.10f}",
            f"{el_t:.10f}",
            f"{az_r:.10f}",
            f"{el_r:.10f}",
            f"{az_err_r:.10f}",
            f"{el_err_r:.10f}",
        ]
        if raw_azel is not None:
            az_o = float(raw_azel[i, 0])
            el_o = float(raw_azel[i, 1])
            az_err_o = float(core.angle_diff_deg(np.asarray([az_o]), np.asarray([az_t]))[0])
            el_err_o = float(el_o - el_t)
            row += [f"{az_o:.10f}", f"{el_o:.10f}", f"{az_err_o:.10f}", f"{el_err_o:.10f}"]
        rows.append(row)

    header = [
        "unix",
        "true_az_deg",
        "true_el_deg",
        "recomputed_az_deg",
        "recomputed_el_deg",
        "recomputed_az_err_deg",
        "recomputed_el_err_deg",
    ]
    if raw_azel is not None:
        header += ["raw_pred_az_deg", "raw_pred_el_deg", "raw_pred_az_err_deg", "raw_pred_el_err_deg"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute AZ/EL from predicted LLA and evaluate against truth.")
    p.add_argument("--pred-csv", required=True, help="Prediction csv with unix + lat/lon/alt columns.")
    p.add_argument("--truth-file", default="2024_calc_az_el.csv")
    p.add_argument("--observer-lat", type=float, default=36.3022)
    p.add_argument("--observer-lon", type=float, default=137.9031)
    p.add_argument("--observer-alt-m", type=float, default=0.0)
    p.add_argument("--train-days", type=float, default=7.0, help="For forecast-only metrics from start unix.")
    p.add_argument("--max-plot-points", type=int, default=12000)
    p.add_argument("--output-dir", default=".tmp/recomputed_azel_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_csv = Path(args.pred_csv)
    truth_file = Path(args.truth_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_unix, pred_lla, pred_azel_raw = load_pred_csv(pred_csv)
    rec_azel = recompute_azel_from_lla(
        sat_lla=pred_lla,
        observer_lat=float(args.observer_lat),
        observer_lon=float(args.observer_lon),
        observer_alt_m=float(args.observer_alt_m),
    )
    pred_full_rec = np.column_stack([pred_lla, rec_azel]).astype(np.float64)

    truth_unix, truth_full = core.load_orbit_numeric_full(truth_file)
    rec_aligned, true_aligned, unix_aligned = core.align_by_unix(pred_unix, pred_full_rec, truth_unix, truth_full)
    rec_metrics_all = core.compute_metrics_azel(true_aligned, rec_aligned)

    raw_metrics_all = None
    raw_aligned = None
    if pred_azel_raw is not None:
        pred_full_raw = np.column_stack([pred_lla, pred_azel_raw]).astype(np.float64)
        raw_aligned, true2, unix2 = core.align_by_unix(pred_unix, pred_full_raw, truth_unix, truth_full)
        if int(unix2.shape[0]) == int(unix_aligned.shape[0]) and np.all(np.asarray(unix2 == unix_aligned)):
            raw_metrics_all = core.compute_metrics_azel(true2, raw_aligned)

    train_end_unix = float(pred_unix[0]) + float(args.train_days) * 86400.0
    fc_mask = np.asarray(unix_aligned >= train_end_unix, dtype=bool)
    if int(np.sum(fc_mask)) <= 0:
        fc_mask = np.ones_like(unix_aligned, dtype=bool)
    rec_metrics_fc = core.compute_metrics_azel(true_aligned[fc_mask], rec_aligned[fc_mask])

    raw_metrics_fc = None
    if raw_aligned is not None:
        raw_metrics_fc = core.compute_metrics_azel(true_aligned[fc_mask], raw_aligned[fc_mask])

    aligned_csv = out_dir / "aligned_recomputed_azel_vs_truth.csv"
    write_aligned_csv(
        path=aligned_csv,
        unix=np.asarray(unix_aligned, dtype=np.float64),
        true_azel=np.asarray(true_aligned[:, 3:5], dtype=np.float64),
        rec_azel=np.asarray(rec_aligned[:, 3:5], dtype=np.float64),
        raw_azel=None if raw_aligned is None else np.asarray(raw_aligned[:, 3:5], dtype=np.float64),
    )

    compare_png = out_dir / "recomputed_vs_truth_compare.png"
    error_png = out_dir / "recomputed_vs_truth_error.png"
    plot_paths = plot_compare_and_error(
        unix=np.asarray(unix_aligned, dtype=np.float64),
        true_azel=np.asarray(true_aligned[:, 3:5], dtype=np.float64),
        rec_azel=np.asarray(rec_aligned[:, 3:5], dtype=np.float64),
        raw_azel=None if raw_aligned is None else np.asarray(raw_aligned[:, 3:5], dtype=np.float64),
        out_compare=compare_png,
        out_error=error_png,
        max_points=int(args.max_plot_points),
    )

    payload = {
        "pred_csv": str(pred_csv),
        "truth_file": str(truth_file),
        "rows_aligned": int(unix_aligned.shape[0]),
        "rows_forecast": int(np.sum(fc_mask)),
        "metrics_all_recomputed_azel": rec_metrics_all,
        "metrics_forecast_recomputed_azel": rec_metrics_fc,
        "metrics_all_raw_pred_azel": raw_metrics_all,
        "metrics_forecast_raw_pred_azel": raw_metrics_fc,
        "aligned_csv": str(aligned_csv),
        "plots": [str(p) for p in plot_paths],
    }
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
