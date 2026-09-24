#!/usr/bin/env python3
"""Run one HealDA global analysis on Apple Silicon (CPU or MPS).

Everything upstream assumes CUDA; this script supplies the two missing pieces:

  * the pure-torch `earth2grid` shim (added to sys.path before any import)
  * a local model package, so no network call to HuggingFace is needed

Usage
-----
    # smoke test, conventional obs only, on CPU
    .venv/bin/python healda_mac/02_run_analysis.py --sensors conv --device cpu

    # full run, all obs, on the M5 GPU
    .venv/bin/python healda_mac/02_run_analysis.py --sensors both --device mps
"""

import argparse
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "shim"))  # earth2grid shim must come first

import numpy as np
import torch

from earth2studio.data import UFSObsConv, UFSObsSat, fetch_dataframe
from earth2studio.models.auto import Package
from earth2studio.models.da import HealDA

CKPT = ROOT / "healda_ckpt"


def rss_gb() -> float:
    """Resident set size in GB (macOS: ru_maxrss is in bytes)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-01-01T00:00:00")
    ap.add_argument("--sensors", choices=["conv", "sat", "both"], default="conv")
    ap.add_argument(
        "--conv-vars",
        nargs="+",
        default=None,
        metavar="VAR",
        help="只同化指定的常规观测类型子集（默认全取）。"
             "可选: gps gps_t gps_q pres q t u v。"
             "例: --conv-vars gps gps_t gps_q  → 只同化 GNSSRO",
    )
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--limit-obs", type=int, default=0, help="subsample N obs per stream (0 = all)")
    ap.add_argument("--no-autocast", action="store_true",
                    help="关掉 bf16 autocast，全程 fp32（MPS 上时可能需要）")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--region",
        nargs=4,
        type=float,
        default=None,
        metavar=("LON0", "LON1", "LAT0", "LAT1"),
        help="report stats for this box (the run itself is always global)",
    )
    args = ap.parse_args()

    t_start = time.time()
    print(f"[env] torch {torch.__version__} | MPS built={torch.backends.mps.is_built()} "
          f"available={torch.backends.mps.is_available()}")

    if args.no_autocast:
        # earth2studio 的 _forward 里写死了 torch.autocast(device, bfloat16)。
        # 在 MPS 上这条路径会触发 Metal 断言，这里整体降级成 fp32。
        import contextlib

        torch.autocast = lambda *a, **k: contextlib.nullcontext()
        print("[env] autocast disabled -> 全程 fp32")

    # ---------------------------------------------------------------- model
    t0 = time.time()
    # 注意：先保持在 CPU 上。观测解码用 ProcessPoolExecutor，若此时模型已在 MPS 上，
    # fork 之后子进程里 Metal device 变 nil，会触发
    #   MPSNDArray ... failed assertion 'device may not be nil'
    # 所以等观测取完再 .to(device)。
    model = HealDA.load_model(Package(str(CKPT)), lat_lon=False)
    model.eval()
    print(f"[model] loaded on cpu  ({time.time() - t0:.1f}s)  rss={rss_gb():.2f} GB")

    conv_schema, sat_schema = model.input_coords()
    print(f"[schema] conv fields : {list(conv_schema.keys())}")
    print(f"[schema] sat  fields : {list(sat_schema.keys())}")

    # ------------------------------------------------------------------ obs
    analysis_time = np.array([np.datetime64(args.date)])
    time_tolerance = (np.timedelta64(-21, "h"), np.timedelta64(3, "h"))
    want_conv = args.sensors in ("conv", "both")
    want_sat = args.sensors in ("sat", "both")

    conv_df = sat_df = None
    if want_conv:
        t0 = time.time()
        conv_vars = np.array(conv_schema["variable"])
        if args.conv_vars:
            known = set(conv_schema["variable"])
            bad = [v for v in args.conv_vars if v not in known]
            if bad:
                raise SystemExit(f"未知常规观测类型 {bad}，可选: {sorted(known)}")
            conv_vars = np.array([v for v in conv_schema["variable"] if v in args.conv_vars])
            print(f"[obs] conventional 子集: {list(conv_vars)}")
        conv_df = fetch_dataframe(
            UFSObsConv(time_tolerance=time_tolerance),
            time=analysis_time,
            variable=conv_vars,
            fields=np.array(list(conv_schema.keys())),
        )
        print(f"[obs] conventional: {len(conv_df):,} rows  ({time.time() - t0:.1f}s)  rss={rss_gb():.2f} GB")
    if want_sat:
        t0 = time.time()
        sat_df = fetch_dataframe(
            UFSObsSat(time_tolerance=time_tolerance),
            time=analysis_time,
            variable=np.array(sat_schema["variable"]),
            fields=np.array(list(sat_schema.keys())),
        )
        print(f"[obs] satellite   : {len(sat_df):,} rows  ({time.time() - t0:.1f}s)  rss={rss_gb():.2f} GB")

    if args.limit_obs:
        if conv_df is not None:
            conv_df = conv_df.iloc[: args.limit_obs]
        if sat_df is not None:
            sat_df = sat_df.iloc[: args.limit_obs]
        print(f"[obs] subsampled to <= {args.limit_obs:,} rows per stream")

    if conv_df is None and sat_df is None:
        raise SystemExit("nothing to assimilate")

    # --------------------------------------------------------------- forward
    t0 = time.time()
    model = model.to(args.device)
    print(f"[model] moved to {model.device}  ({time.time() - t0:.1f}s)  rss={rss_gb():.2f} GB")

    if args.device == "mps":
        # HealDA 的 adaLN-Zero 调制层是 nn.Linear(0, 6144)（无条件模型，
        # class_labels 维度为 0）。这类零输入 Linear 在 Metal 上做前向会直接
        # SIGABRT：MPSNDArray ... 'device may not be nil'。
        # 因为 in_features=0，输出恒等于 bias，改到 CPU 上算数值完全不变。
        sys.path.insert(0, str(HERE))
        from mps_patch import patch_build_input_on_cpu, patch_for_mps

        patch_for_mps(model, verbose=False)
        print(f"[mps] patched zero-input Linear layers  rss={rss_gb():.2f} GB")

        # 第二个坑：build_input() 把 11M 行观测逐列 .to(mps, non_blocking=True)，
        # 其中 target_time(int64) 会被后续拷贝覆盖成观测时间的纳秒值，
        # 导致 *_meta 里的相对时间特征变垃圾（分析场与 ERA5 相关性 0.95→0.45）。
        # 改到 CPU 上构造再整体搬回，规避 Metal 的缓冲区复用问题。
        if patch_build_input_on_cpu(model):
            print("[mps] build_input() 已改到 CPU 构造")

    t0 = time.time()
    with torch.inference_mode():
        out = model(conv_obs=conv_df, sat_obs=sat_df)
    dt = time.time() - t0
    print(f"[run] analysis done  ({dt:.1f}s)  peak rss={rss_gb():.2f} GB")
    print(f"[out] dims={out.dims}  shape={out.shape}  dtype={out.dtype}")

    # 原生输出只有 npix 维，没有经纬度。用 shim 的 HEALPIX_PAD_XY 网格补上，
    # 这样后面才能按经纬度框选区域。
    if "npix" in out.dims and "lat" not in out.coords:
        grid = model._grid
        out = out.assign_coords(
            lat=("npix", np.asarray(grid.lat)),
            lon=("npix", np.asarray(grid.lon)),
        )
        print(f"[out] attached lat/lon: {len(out.coords['lat'])} pixels")

    # ---------------------------------------------------------------- report
    print("\n=== 通道统计（全球） ===")
    for i, v in enumerate(out.coords["variable"].values[:8]):
        col = out.isel(variable=i).values
        print(f"  {str(v):10s} min={np.nanmin(col):11.4f}  max={np.nanmax(col):11.4f}  "
              f"mean={np.nanmean(col):11.4f}")

    if args.region:
        lon0, lon1, lat0, lat1 = args.region
        grid_lat = np.asarray(out.coords["lat"].values) if "lat" in out.coords else None
        if grid_lat is None:
            print("\n[warn] 输出没有 lat/lon 坐标，跳过区域统计")
        else:
            grid_lon = np.asarray(out.coords["lon"].values)
            sel = (grid_lon >= lon0) & (grid_lon <= lon1) & (grid_lat >= lat0) & (grid_lat <= lat1)
            print(f"\n=== 区域 {lon0}..{lon1}E, {lat0}..{lat1}N : {int(sel.sum())} 个像素 ===")
            sub = out.isel(npix=sel) if "npix" in out.dims else out
            for i, v in enumerate(out.coords["variable"].values[:8]):
                col = sub.isel(variable=i).values
                print(f"  {str(v):10s} mean={np.nanmean(col):11.4f}")

    # ------------------------------------------------------------------ save
    out_path = args.out or f"outputs/healda_{args.date.replace(':', '')}_{args.sensors}.nc"
    os.makedirs(pathlib.Path(out_path).parent, exist_ok=True)
    out.to_netcdf(out_path)
    print(f"\n[saved] {out_path}")
    print(f"[total] {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
