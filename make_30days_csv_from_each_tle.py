# -*- coding: utf-8 -*-
"""
28628_2023 ディレクトリにある各TLE（*.txt）を「1ファイル=1つのTLE」として扱い、
そのTLEだけを使って「ファイル名の時刻(JST想定)」から先30日分の
UNIX / Lat / Lon / Alt / AZ / EL を計算し、CSVを 28628_2023_csv に出力します。

- 入力例: 28628_2023/2023-01-04-20-23-18.txt
- 出力例: 28628_2023_csv/2023-01-04-20-23-18.csv

CSV形式は添付の 2023_calc_az_el.csv に合わせて:
date,UNIX,Lat,Lon,Alt,AZ,EL

注意:
- date は JST(Asia/Tokyo) のローカル時刻を文字列で出力
- UNIX はそのJST時刻に対応する Unix epoch (UTC) 秒
- 伝播計算は UTC で行います (JSTから常に -9時間)

依存:
    pip install skyfield numpy

（Skyfield が IERS データを初回に取得する場合があります）
"""

from __future__ import annotations

import argparse
import calendar
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84


# --- ここは日本(東京)固定：DST無しなので +9h 固定で十分 ---
JST_OFFSET = timedelta(hours=9)

TLE_LINE1_RE = re.compile(r"^1\s+(\d{5})")
TLE_LINE2_RE = re.compile(r"^2\s+(\d{5})")
FILENAME_DT_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")


@dataclass(frozen=True)
class TLEPair:
    line1: str
    line2: str
    satnum: str  # 5桁


def parse_jst_datetime_from_filename(filename: str) -> datetime:
    """
    例:
        2023-01-08-20-49-54.txt
        2023-01-08-20-49-54_1.txt
    などから先頭の日時部分だけを取り出して JST の naive datetime として返す。
    """
    m = FILENAME_DT_RE.search(filename)
    if not m:
        raise ValueError(f"ファイル名から日時を抽出できません: {filename}")
    dt_str = m.group(1)
    return datetime.strptime(dt_str, "%Y-%m-%d-%H-%M-%S")


def read_tle_pair(tle_path: Path, sat_name: Optional[str] = None) -> TLEPair:
    """
    TLEファイルから (line1, line2) を抽出する。
    - sat_name が指定されていれば、その衛星番号(5桁)に一致するTLEを探す
    - そうでなければ、最初に見つかった「1行目→2行目」のペアを採用
    """
    raw = tle_path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln.strip() for ln in raw if ln.strip()]

    want_satnum = None
    if sat_name:
        # "28628" のような番号文字列を想定。先頭5桁を使う
        m = re.search(r"(\d{5})", sat_name)
        if m:
            want_satnum = m.group(1)

    # まずは隣接する 1行目→2行目 をスキャン
    for i in range(len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if not l1.startswith("1 ") or not l2.startswith("2 "):
            continue
        m1 = TLE_LINE1_RE.match(l1)
        m2 = TLE_LINE2_RE.match(l2)
        if not m1 or not m2:
            continue
        satnum1, satnum2 = m1.group(1), m2.group(1)
        if satnum1 != satnum2:
            continue
        if want_satnum and satnum1 != want_satnum:
            continue
        return TLEPair(line1=l1, line2=l2, satnum=satnum1)

    # 次に、離れている 1行目 と 2行目 を探す（念のため）
    if want_satnum:
        idx1 = None
        for i, ln in enumerate(lines):
            if ln.startswith("1 "):
                m1 = TLE_LINE1_RE.match(ln)
                if m1 and m1.group(1) == want_satnum:
                    idx1 = i
                    break
        if idx1 is not None:
            for j in range(idx1 + 1, len(lines)):
                ln2 = lines[j]
                if ln2.startswith("2 "):
                    m2 = TLE_LINE2_RE.match(ln2)
                    if m2 and m2.group(1) == want_satnum:
                        return TLEPair(line1=lines[idx1], line2=ln2, satnum=want_satnum)

    raise ValueError(f"TLE(1行目/2行目)のペアを抽出できません: {tle_path}")


def jst_naive_to_unix_seconds(dt_jst_naive: datetime) -> float:
    """
    JST の naive datetime を「JST(UTC+9)として解釈」して Unix 秒に変換する。
    （OSのローカルタイムゾーンに依存しないよう、+9h固定でUTCに戻してから timegm する）
    """
    dt_utc = dt_jst_naive - JST_OFFSET
    # dt_utc は UTC の naive datetime とみなす
    return calendar.timegm(dt_utc.timetuple()) + dt_utc.microsecond / 1e6


def build_utc_time_arrays(start_dt_jst: datetime, count: int, step: timedelta) -> Tuple[list[int], list[int], list[int], list[int], list[int], list[float]]:
    """
    Skyfield の ts.utc(year, month, day, hour, minute, second) に渡す配列を作る。
    start_dt_jst は JST naive として受け取り、UTCに変換してから生成する。
    """
    start_utc = start_dt_jst - JST_OFFSET
    years: list[int] = []
    months: list[int] = []
    days: list[int] = []
    hours: list[int] = []
    minutes: list[int] = []
    seconds: list[float] = []

    dt = start_utc
    for _ in range(count):
        years.append(dt.year)
        months.append(dt.month)
        days.append(dt.day)
        hours.append(dt.hour)
        minutes.append(dt.minute)
        seconds.append(dt.second + dt.microsecond / 1e6)
        dt += step

    return years, months, days, hours, minutes, seconds


def generate_csv_for_one_tle(
    tle_path: Path,
    out_csv_path: Path,
    sat_name: str,
    observer_lat: float,
    observer_lon: float,
    days: int,
    step_minutes: int,
    include_end: bool,
    ts,
) -> None:
    start_dt_jst = parse_jst_datetime_from_filename(tle_path.name)
    tle = read_tle_pair(tle_path, sat_name=sat_name)

    step = timedelta(minutes=step_minutes)
    total_minutes = days * 24 * 60
    if total_minutes % step_minutes != 0:
        raise ValueError(f"days*24*60 が step_minutes で割り切れません: days={days}, step_minutes={step_minutes}")

    count = total_minutes // step_minutes
    if include_end:
        # 例: 30日ちょうど先の時刻も含める
        count += 1

    # 時刻配列（UTC成分）
    years, months, days_arr, hours, minutes, seconds = build_utc_time_arrays(start_dt_jst, count, step)
    t = ts.utc(years, months, days_arr, hours, minutes, seconds)

    # 衛星・観測点
    satellite = EarthSatellite(tle.line1, tle.line2, sat_name, ts)
    observer = wgs84.latlon(observer_lat, observer_lon)

    # 衛星の地上座標（lat/lon/alt）
    geocentric = satellite.at(t)
    lat, lon = wgs84.latlon_of(geocentric)
    alt_km = wgs84.height_of(geocentric).km

    # 方位・仰角（AZ/EL）
    topocentric = (satellite - observer).at(t)
    alt, az, _distance = topocentric.altaz()
    az_deg = az.degrees
    el_deg = alt.degrees

    # date(JST) と UNIX
    # JST表示用の dt は、start_dt_jst から step で増やす
    dt_jst_list = [start_dt_jst + step * i for i in range(count)]
    unix0 = jst_naive_to_unix_seconds(start_dt_jst)
    unix_list = (unix0 + (np.arange(count) * step.total_seconds())).astype(float)

    # 出力
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "UNIX", "Lat", "Lon", "Alt", "AZ", "EL"])
        for dt_jst, unix_ts, la, lo, al, azv, elv in zip(
            dt_jst_list, unix_list, lat.degrees, lon.degrees, alt_km, az_deg, el_deg
        ):
            w.writerow([
                dt_jst.strftime("%Y-%m-%d %H:%M:%S"),
                float(unix_ts),
                float(la),
                float(lo),
                float(al),
                float(azv),
                float(elv),
            ])


def main() -> None:
    p = argparse.ArgumentParser(
        description="28628_2023 の各TLEファイルから、各々30日分の予報(UNIX/Lat/Lon/Alt/AZ/EL)をCSVに出力します。"
    )
    p.add_argument("--tle-dir", default="28628_2023", help="入力TLEディレクトリ（*.txt）")
    p.add_argument("--out-dir", default="28628_2023_csv", help="出力CSVディレクトリ")
    p.add_argument("--sat-name", default="28628", help="対象衛星番号（例: 28628）")
    p.add_argument("--observer-lat", type=float, default=36.3022, help="観測点 緯度（既定: 36.3022）")
    p.add_argument("--observer-lon", type=float, default=137.9031, help="観測点 経度（既定: 137.9031）")
    p.add_argument("--days", type=int, default=30, help="何日先まで出すか（既定: 30）")
    p.add_argument("--step-minutes", type=int, default=1, help="時間ステップ（分）（既定: 1）")
    p.add_argument("--include-end", action="store_true", help="30日ちょうど先の時刻も含めて出力する")
    p.add_argument("--overwrite", action="store_true", help="既存CSVがあっても上書きする")
    p.add_argument("--max-files", type=int, default=0, help="デバッグ用: 処理する最大ファイル数（0なら全件）")

    args = p.parse_args()

    tle_dir = Path(args.tle_dir)
    out_dir = Path(args.out_dir)

    if not tle_dir.exists() or not tle_dir.is_dir():
        raise FileNotFoundError(f"TLEディレクトリが見つかりません: {tle_dir.resolve()}")

    tle_files = sorted(tle_dir.glob("*.txt"))
    if args.max_files and args.max_files > 0:
        tle_files = tle_files[: args.max_files]

    print(f"[INFO] tle_dir   : {tle_dir.resolve()}")
    print(f"[INFO] out_dir   : {out_dir.resolve()}")
    print(f"[INFO] sat_name  : {args.sat_name}")
    print(f"[INFO] observer  : lat={args.observer_lat}, lon={args.observer_lon}")
    print(f"[INFO] horizon   : {args.days} days, step={args.step_minutes} minutes, include_end={args.include_end}")
    print(f"[INFO] files     : {len(tle_files)}")

    # Skyfield Timescale（初回にIERSデータを取得/更新する場合があります）
    ts = load.timescale()

    ok = 0
    ng = 0

    for i, tle_path in enumerate(tle_files, 1):
        out_csv = out_dir / (tle_path.stem + ".csv")
        if out_csv.exists() and not args.overwrite:
            print(f"[SKIP] ({i}/{len(tle_files)}) exists: {out_csv.name}")
            continue

        try:
            print(f"[DO  ] ({i}/{len(tle_files)}) {tle_path.name} -> {out_csv.name}")
            generate_csv_for_one_tle(
                tle_path=tle_path,
                out_csv_path=out_csv,
                sat_name=args.sat_name,
                observer_lat=args.observer_lat,
                observer_lon=args.observer_lon,
                days=args.days,
                step_minutes=args.step_minutes,
                include_end=args.include_end,
                ts=ts,
            )
            ok += 1
        except Exception as e:
            ng += 1
            print(f"[ERR ] {tle_path.name}: {e}")

    print(f"[DONE] ok={ok}, ng={ng}")


if __name__ == "__main__":
    main()
