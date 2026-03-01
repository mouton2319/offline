#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ff_conv_encode_1overN.py

フィードフォワード型（feedforward）畳み込み符号（符号化率 1/N, k=1）で
指定ファイルをビット列として符号化して出力します。

教科書の式(10.8):
    x_i = A (m_i, m_{i-1}, ..., m_{i-S})^T   (mod 2)
ここで
    - m_i ∈ {0,1} は入力（情報）ビット（1ステップ1ビット）
    - x_i ∈ {0,1}^N は出力ベクトル（1ステップ N ビット）
    - A は 2元 N×(S+1) 行列（生成行列）

このスクリプトでは A を以下のどちらかで指定できます:

(A) --A-rows で行ベクトルを0/1文字列として指定（推奨: 式(10.8)とそのまま対応）
    例: 図10.9相当 (N=2, S=2, A=[[1 1 1],[1 0 1]])
        --N 2 --A-rows 111 101

(B) --generators-octal で生成多項式を8進表記で指定（通信工学でよく使う表現）
    例: (7,5)_8 は上の A と等価
        --N 2 --generators-octal 7 5

注意:
- 生成行列 A が未指定の場合、N=2 なら図10.9の A をデフォルトにします。
  N≠2 の場合は必ず --A-rows か --generators-octal を指定してください。
- 入力ファイルは「1バイトのMSB→LSB」の順でビット化します。
- 出力は、各入力ビットごとに x[0], x[1], ..., x[N-1] の順で N ビットを出し、
  全体を MSB-first でバイトにパックします。
- 既知終端が必要なら --terminate（デフォルトON）で S 個の 0 を末尾に追加します
  （S=メモリ長。状態を 0 に戻すための tail bits）。

使い方（例）:
  # 図10.9の符号（A=[[111],[101]]）で符号化（デフォルト）
  python ff_conv_encode_1overN.py input.bin encoded.bin --N 2 --meta meta.json

  # 1/3 符号（例: generators 7,5,7 で S=2）※例としての指定
  python ff_conv_encode_1overN.py input.bin encoded.bin --N 3 --generators-octal 7 5 7

  # 終端無し（tail bitsを付けない）
  python ff_conv_encode_1overN.py input.bin encoded.bin --N 2 --no-terminate

"""

from __future__ import annotations

import argparse
import json
import os
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple


def iter_bits_msb_first(f: BinaryIO, chunk_size: int = 1024 * 1024) -> Iterable[int]:
    """ファイルを MSB-first でビット列として取り出すジェネレータ。"""
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            return
        for b in chunk:
            for shift in range(7, -1, -1):
                yield (b >> shift) & 1


class BitWriter:
    """MSB-first でビットを書き、8ビットごとに出力する。"""

    def __init__(self, f: BinaryIO):
        self._f = f
        self._buf = 0
        self._nbits = 0
        self.total_bits_written = 0  # パディング込み

    def write_bit(self, bit: int) -> None:
        self._buf = ((self._buf << 1) | (bit & 1)) & 0xFF
        self._nbits += 1
        self.total_bits_written += 1
        if self._nbits == 8:
            self._f.write(bytes([self._buf]))
            self._buf = 0
            self._nbits = 0

    def write_bits(self, bits: Iterable[int]) -> None:
        for b in bits:
            self.write_bit(int(b) & 1)

    def flush(self) -> int:
        """残りビットを 0 で埋めて 1バイトとして書き出す。埋めたビット数を返す。"""
        if self._nbits == 0:
            return 0
        pad = 8 - self._nbits
        self._buf = (self._buf << pad) & 0xFF
        self._f.write(bytes([self._buf]))
        self._buf = 0
        self._nbits = 0
        self.total_bits_written += pad
        return pad


def _parse_A_rows(A_rows: List[str]) -> Tuple[List[int], int]:
    """A の各行を '1011' のような文字列で受け取り、(mask_list, memory_S) を返す。

    row[0] が m_i（現在）, row[1] が m_{i-1}, ..., row[S] が m_{i-S} の係数。
    mask の bit k が row[k] に対応する（bit0 が m_i）。
    """
    if not A_rows:
        raise ValueError("A_rows is empty")

    row_len = len(A_rows[0])
    if row_len < 1:
        raise ValueError("A row length must be >= 1")

    for r in A_rows:
        if len(r) != row_len:
            raise ValueError("All A-rows must have the same length (=S+1)")
        if any(ch not in "01" for ch in r):
            raise ValueError(f"A-row must be 0/1 string. got={r!r}")

    S = row_len - 1
    masks: List[int] = []
    for r in A_rows:
        mask = 0
        for k, ch in enumerate(r):
            if ch == "1":
                mask |= (1 << k)
        if mask == 0:
            raise ValueError("Each generator row must not be all-zero (it would output always 0).")
        masks.append(mask)

    return masks, S


def _parse_octal_list(gen_octal: List[str]) -> Tuple[List[int], int]:
    """8進表記の生成多項式リストを受け取り、(mask_list, memory_S) を返す。"""
    if not gen_octal:
        raise ValueError("generators-octal is empty")

    masks: List[int] = []
    max_bits = 0
    for s in gen_octal:
        ss = s.strip().lower()
        if ss.startswith("0o"):
            ss = ss[2:]
        if not ss or any(ch not in "01234567" for ch in ss):
            raise ValueError(f"Invalid octal generator: {s!r}")
        mask = int(ss, 8)
        if mask == 0:
            raise ValueError("Generator must not be 0.")
        masks.append(mask)
        max_bits = max(max_bits, mask.bit_length())

    S = max_bits - 1
    if S < 0:
        S = 0
    return masks, S


def _masks_to_A_rows(masks: List[int], S: int) -> List[str]:
    """mask_list (bit0..bitS) を '101...' の A-rows 表現に戻す（メタ用）。"""
    rows: List[str] = []
    for mask in masks:
        row = []
        for k in range(S + 1):
            row.append("1" if ((mask >> k) & 1) else "0")
        rows.append("".join(row))
    return rows


def _parity(x: int) -> int:
    """x の1の個数の偶奇（偶数→0, 奇数→1）。"""
    return x.bit_count() & 1


def encode_file(
    in_path: str,
    out_path: str,
    N: int,
    masks: List[int],
    S: int,
    terminate: bool,
    meta_path: Optional[str],
) -> None:
    if N <= 0:
        raise ValueError("N must be >= 1")
    if len(masks) != N:
        raise ValueError(f"N={N} but number of generators is {len(masks)}")
    if S < 0:
        raise ValueError("S must be >= 0")

    orig_size = os.path.getsize(in_path)
    orig_bits = orig_size * 8

    # 終端する場合は tail bits = S（0をS個入力）
    tail_bits = S if terminate else 0

    # ストリーミング符号化
    reg = 0  # 過去Sビットを保持（bit0が1遅延）
    encoded_bits_nominal = N * (orig_bits + tail_bits)

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        bw = BitWriter(fout)

        def step(m: int) -> None:
            nonlocal reg
            # window bits: bit0=m_i, bit1=m_{i-1}, ..., bitS=m_{i-S}
            window = (reg << 1) | (m & 1)
            # N個出力
            for g in masks:
                bw.write_bit(_parity(window & g))
            # 次のレジスタ（過去Sビットだけ保持）
            if S > 0:
                reg = window & ((1 << S) - 1)
            else:
                reg = 0

        for bit in iter_bits_msb_first(fin):
            step(bit)

        if terminate and S > 0:
            for _ in range(S):
                step(0)

        pad_bits = bw.flush()

    out_size = os.path.getsize(out_path)

    if meta_path:
        meta: Dict = {
            "scheme": "feedforward_convolutional",
            "rate": {"k": 1, "n": N, "value": 1.0 / float(N)},
            "memory": S,
            "constraint_length": S + 1,
            "generators_octal": [format(m, "o") for m in masks],
            "generator_matrix_A_rows": _masks_to_A_rows(masks, S),
            "termination": {"enabled": bool(terminate), "tail_bits": tail_bits},
            "original": {"size_bytes": orig_size, "bits": orig_bits},
            "encoded": {
                "bits": int(encoded_bits_nominal),
                "bytes": int(out_size),
                "pad_bits_at_end": int(pad_bits),
                "bits_in_file_including_pad": int(out_size * 8),
            },
            "io": {
                "input_bit_order": "MSB-first per byte",
                "output_bit_order": "x[0]..x[N-1] per input bit, packed MSB-first per byte",
            },
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="フィードフォワード型畳み込み符号（符号化率1/N）でファイルを符号化します。")
    ap.add_argument("input", help="入力ファイル（バイナリ）")
    ap.add_argument("output", help="出力ファイル（符号化ビット列、ヘッダ無し）")
    ap.add_argument("--N", type=int, required=True, help="出力ビット数 N（符号化率 1/N）")

    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--A-rows",
        nargs="+",
        default=None,
        help="生成行列Aの各行を 0/1 文字列で指定（例: --A-rows 111 101）",
    )
    group.add_argument(
        "--generators-octal",
        nargs="+",
        default=None,
        help="生成多項式を8進表記で N 個指定（例: --generators-octal 7 5）",
    )

    ap.add_argument(
        "--memory",
        type=int,
        default=None,
        help="メモリ長 S を明示指定（省略時は A-rows/生成多項式から推定）。",
    )
    ap.add_argument("--meta", default=None, help="メタデータJSONの出力先（任意）。")
    ap.add_argument("--no-terminate", action="store_true", help="終端（tail bits の 0 追加）を行わない。")
    args = ap.parse_args()

    N = int(args.N)

    # 生成行列の決定
    masks: List[int]
    S: int
    from_A: bool

    if args.A_rows is not None:
        masks, S = _parse_A_rows(list(args.A_rows))
        from_A = True
    elif args.generators_octal is not None:
        masks, S = _parse_octal_list(list(args.generators_octal))
        from_A = False
    else:
        # 未指定なら、N=2 のときだけ図10.9のデフォルト（A=[[111],[101]]）を使う
        if N != 2:
            raise SystemExit("N≠2 の場合は --A-rows か --generators-octal を指定してください。")
        masks, S = _parse_A_rows(["111", "101"])
        from_A = True

    if len(masks) != N:
        raise SystemExit(f"指定N={N} と、指定した生成行列の行数/生成多項式個数={len(masks)} が一致しません。")

    # memory（メモリ長 S）の扱い
    if args.memory is not None:
        Sm = int(args.memory)
        if Sm < 0:
            raise SystemExit("--memory は 0 以上で指定してください。")
        if from_A:
            # A-rows で列数 (=S+1) が決まるため一致必須
            if Sm != S:
                raise SystemExit(f"--memory {Sm} は A-rows から決まる S={S} と一致させてください。")
        else:
            # generators-octal から推定される最小Sより小さくはできない
            if Sm < S:
                raise SystemExit(f"--memory {Sm} は生成多項式から推定される最小S={S} 以上にしてください。")
            S = Sm

    terminate = not bool(args.no_terminate)

    encode_file(args.input, args.output, N, masks, S, terminate, args.meta)


if __name__ == "__main__":
    main()
