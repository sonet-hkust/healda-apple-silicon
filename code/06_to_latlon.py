#!/usr/bin/env python3
"""把 HealDA 的 HEALPix (nside=64) 输出重网格为规则经纬网格的 CF 兼容 NetCDF，
可直接用 ncview / Panoply / Python 查看。

输入:  healda_*.nc   dims = (time, variable, npix), 通道名形如 t850 / u300 / tcwv
输出:  *_latlon.nc   dims = (time, level, lat, lon)  +  (time, lat, lon)

用法:
    python 06_to_latlon.py outputs/healda_20240601_mps.nc
    python 06_to_latlon.py outputs/healda_20240601_mps.nc --res 0.25 \
        --region 70 140 15 55 --out outputs/eastasia_0p25.nc
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import time

import numpy as np
import xarray as xr

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

# ---------------------------------------------------------------- 通道定义
# HealDA 输出的 74 个通道: 5 个三维量 x 13 层 + 9 个二维量
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
VAR3D = {
    "t": ("air_temperature", "K", "Air temperature"),
    "u": ("eastward_wind", "m s-1", "Eastward wind component"),
    "v": ("northward_wind", "m s-1", "Northward wind component"),
    "z": ("geopotential", "m2 s-2", "Geopotential"),
    "q": ("specific_humidity", "kg kg-1", "Specific humidity"),
}
# 由 z 派生, 看图更直观
G0 = 9.80665
GH_META = ("geopotential_height", "gpm", "Geopotential height (z / g0)")

VAR2D = {
    "tcwv": ("atmosphere_mass_content_of_water_vapor", "kg m-2", "Total column water vapour"),
    "t2m": ("air_temperature", "K", "2 metre temperature"),
    "u10m": ("eastward_wind", "m s-1", "10 metre U wind component"),
    "v10m": ("northward_wind", "m s-1", "10 metre V wind component"),
    "u100m": ("eastward_wind", "m s-1", "100 metre U wind component"),
    "v100m": ("northward_wind", "m s-1", "100 metre V wind component"),
    "msl": ("air_pressure_at_mean_sea_level", "Pa", "Mean sea level pressure"),
    "sst": ("sea_surface_temperature", "K", "Sea surface temperature"),
    "sic": ("sea_ice_area_fraction", "1", "Sea ice area fraction"),
}


# ---------------------------------------------------------------- 网格工具
def hpx_latlon(npix: int) -> tuple[np.ndarray, np.ndarray]:
    """没有 lat/lon 坐标时, 用 earth2grid shim 现算 HEALPix PAD_XY 的像元中心。"""
    sys.path.insert(0, str(HERE / "shim"))
    from earth2grid import healpix  # noqa: PLC0415

    nside = int(round(np.sqrt(npix / 12)))
    g = healpix.Grid(int(np.log2(nside)), pixel_order=healpix.HEALPIX_PAD_XY)
    return np.asarray(g.lat, dtype=np.float64), np.asarray(g.lon, dtype=np.float64)


def make_target_grid(res: float, region=None):
    """生成规则经纬网格中心点。返回 lat1d, lon1d。"""
    if region is None:
        lon0, lon1, lat0, lat1 = 0.0, 360.0, -90.0, 90.0
    else:
        lon0, lon1, lat0, lat1 = region
    nlat = int(round((lat1 - lat0) / res)) + 1
    nlon = int(round((lon1 - lon0) / res)) + 1
    lat = lat0 + res * np.arange(nlat)
    lon = lon0 + res * np.arange(nlon - 1)  # 不含右端点, 避免 0/360 重复
    return np.ascontiguousarray(lat, dtype=np.float64), np.ascontiguousarray(
        lon, dtype=np.float64
    )


def xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """经纬度 -> 单位球笛卡尔坐标 (用于球面最近邻, 天然处理经度周期)。"""
    la = np.deg2rad(lat_deg)
    lo = np.deg2rad(lon_deg)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def build_weights(src_lat, src_lon, tgt_lat, tgt_lon, k=6, smooth_deg=0.55):
    """返回 (idx, w): 目标点到源点的 k 近邻索引与高斯权重(已归一化)。"""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    tree = cKDTree(xyz(src_lat, src_lon).reshape(-1, 3))
    tgt = xyz(tgt_lat, tgt_lon).reshape(-1, 3)
    dist, idx = tree.query(tgt, k=k, workers=-1)
    L = np.deg2rad(smooth_deg)
    w = np.exp(-((dist / L) ** 2))
    w /= w.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), w.astype(np.float32)  # (N,k) (N,k)


def remap(field2d: np.ndarray, idx: np.ndarray, w: np.ndarray, shape) -> np.ndarray:
    """field2d: (npix, nchan) -> (nlat, nlon, nchan); 再转置为 (nchan, nlat, nlon)。"""
    # gather: (N,k,nchan) 逐通道加权
    g = field2d[idx]                     # (N, k, nchan) float32
    out = np.einsum("nk,nkc->nc", w, g, optimize=True)
    nlat, nlon = shape
    return out.reshape(nlat, nlon, -1).transpose(2, 0, 1)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(
        description="HealDA HEALPix 输出 -> 规则经纬网格 NetCDF (ncview 友好)"
    )
    ap.add_argument("nc", help="HealDA 原始输出 .nc")
    ap.add_argument("--out", default=None, help="输出文件名 (默认在输入旁生成 *_latlon.nc)")
    ap.add_argument("--res", type=float, default=0.5, help="网格分辨率(度), 默认 0.5")
    ap.add_argument("--region", nargs=4, type=float, default=None,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"), help="只输出该区域")
    ap.add_argument("--method", choices=["idw", "nearest"], default="idw",
                    help="插值方式: idw=高斯加权平滑(默认), nearest=最近邻")
    ap.add_argument("--smooth-deg", type=float, default=0.55,
                    help="IDW 高斯核尺度(度)。小=更贴近原像元, 大=更平滑")
    ap.add_argument("--no-gh", action="store_true", help="不输出派生变量 gh (位势高度 gpm)")
    ap.add_argument("--vars", nargs="+", default=None,
                    help="只输出指定变量, 如 t u v z gh msl t2m")
    ap.add_argument("--format", default="NETCDF4_CLASSIC",
                    choices=["NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_CLASSIC",
                             "NETCDF3_64BIT"],
                    help="NetCDF 格式。ncview 若读不了 NETCDF4, 改用 NETCDF3_CLASSIC")
    ap.add_argument("--time-units", default=None,
                    help="时间坐标基准, 默认 'hours since <该时刻>'")
    args = ap.parse_args()

    src = pathlib.Path(args.nc)
    out = pathlib.Path(args.out) if args.out else src.with_name(src.stem + "_latlon.nc")
    out.parent.mkdir(parents=True, exist_ok=True)

    t_all = time.time()

    # ---------- 读源场 ----------
    ds = xr.open_dataset(src)
    key = list(ds.data_vars)[0]
    da = ds[key]
    dims = da.dims
    if "npix" not in dims or "variable" not in dims:
        print(f"[err] 输入维度不是 (time, variable, npix): {dims}", file=sys.stderr)
        return 1
    da = da.transpose("time", "variable", "npix")
    F = np.asarray(da.values, dtype=np.float32)          # (nt, nch, npix)
    ch_names = [str(v) for v in ds["variable"].values]
    ch_idx = {n: i for i, n in enumerate(ch_names)}
    nt = F.shape[0]

    if "lat" in ds.coords and "lon" in ds.coords:
        src_lat = np.asarray(ds["lat"].values, dtype=np.float64)
        src_lon = np.asarray(ds["lon"].values, dtype=np.float64)
    else:
        src_lat, src_lon = hpx_latlon(F.shape[-1])
        print(f"[grid] 从 shim 生成 HEALPix 中心: nside={int(round(np.sqrt(F.shape[-1]/12)))}")

    # 时间
    tvals = np.asarray(ds["time"].values)
    if tvals.dtype.kind == "M":
        times = tvals.astype("datetime64[s]").astype(dt.datetime)
    else:
        times = [dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(v)) for v in tvals]

    print(f"[in ] {src.name}  dims={dict(ds.sizes)}  time={times[0]}  dtype=float32")

    # ---------- 目标网格 ----------
    lat1d, lon1d = make_target_grid(args.res, args.region)
    nlat, nlon = lat1d.size, lon1d.size
    print(f"[grid] 目标网格 {nlat} x {nlon} @ {args.res}deg"
          + (f"  区域 lon[{lon1d[0]},{lon1d[-1]}] lat[{lat1d[0]},{lat1d[-1]}]"
             if args.region else "  全球"))

    TLAT, TLON = np.meshgrid(lat1d, lon1d, indexing="ij")
    t0 = time.time()
    if args.method == "nearest":
        from scipy.spatial import cKDTree  # noqa: PLC0415
        tree = cKDTree(xyz(src_lat, src_lon).reshape(-1, 3))
        _, idx = tree.query(xyz(TLAT, TLON).reshape(-1, 3), k=1, workers=-1)
        idx = idx.reshape(-1, 1)
        w = np.ones_like(idx, dtype=np.float32)
    else:
        idx, w = build_weights(src_lat, src_lon, TLAT, TLON,
                               k=6, smooth_deg=args.smooth_deg)
    print(f"[grid] 邻接权重构建完成 ({time.time() - t0:.1f}s, k={idx.shape[1]}, "
          f"method={args.method})")

    # ---------- 选变量 ----------
    want3 = list(VAR3D.keys()) + ([] if args.no_gh else ["gh"])
    if args.vars:
        vset = set(args.vars)
        want3 = [v for v in want3 if v in vset]
        want2 = [v for v in VAR2D if v in vset]
    else:
        want2 = list(VAR2D.keys())
    missing = [v for v in want3 if v != "gh" and f"{v}850" not in ch_idx]
    if missing:
        print(f"[warn] 源文件中缺少变量: {missing}", file=sys.stderr)
        want3 = [v for v in want3 if v not in missing]

    print(f"[out] 3D: {want3}   2D: {want2}")

    # ---------- 重网格 ----------
    t0 = time.time()
    data_vars = {}
    for v in want3:
        src_v = "z" if v == "gh" else v
        cols = [ch_idx[f"{src_v}{lv}"] for lv in LEVELS]
        cube = np.empty((nt, len(LEVELS), nlat, nlon), dtype=np.float32)
        for it in range(nt):
            blk = F[it][cols, :].T.copy()               # (npix, nlev) 连续内存
            cube[it] = remap(blk, idx, w, (nlat, nlon))
        if v == "gh":
            cube /= G0
        std_name, units, long_name = (GH_META if v == "gh" else VAR3D[v])
        data_vars[v] = xr.DataArray(
            cube,
            dims=("time", "level", "lat", "lon"),
            attrs=dict(standard_name=std_name, long_name=long_name, units=units,
                       _FillValue=np.float32(np.nan)),
        )
        print(f"   {v:3s} -> {tuple(cube.shape)}  "
              f"min={np.nanmin(cube):11.4f} max={np.nanmax(cube):11.4f}")

    for v in want2:
        if v not in ch_idx:
            print(f"   [skip] {v} 不在源文件")
            continue
        col = ch_idx[v]
        arr = np.empty((nt, nlat, nlon), dtype=np.float32)
        for it in range(nt):
            blk = F[it][col:col + 1, :].T.copy()        # (npix, 1)
            arr[it] = remap(blk, idx, w, (nlat, nlon))[0]   # (nchan=1, nlat, nlon) -> (nlat, nlon)
        std_name, units, long_name = VAR2D[v]
        data_vars[v] = xr.DataArray(
            arr, dims=("time", "lat", "lon"),
            attrs=dict(standard_name=std_name, long_name=long_name, units=units,
                       _FillValue=np.float32(np.nan)),
        )
        print(f"   {v:5s} -> {tuple(arr.shape)}  "
              f"min={np.nanmin(arr):11.4f} max={np.nanmax(arr):11.4f}")

    print(f"[grid] 重网格完成 ({time.time() - t0:.1f}s)")

    # ---------- 组装 ----------
    # 时间坐标统一写成数值型 + CF units (基准取第一帧), ncview/ncdump/Python 都能解
    tbase = args.time_units or f"hours since {times[0]:%Y-%m-%dT%H:%M:%S}"
    base = times[0] if args.time_units is None else None
    tnum = np.zeros(nt, dtype=np.float64)
    if base is not None:
        for i, tv in enumerate(times):
            tnum[i] = (tv - base).total_seconds() / 3600.0
    else:
        tnum = np.asarray(
            [(tv - times[0]).total_seconds() / 3600.0 for tv in times]
        )  # 自定义基准时退化为相对第一帧(需用户自行校准)

    coords = {
        "time": ("time", tnum,
                 dict(units=tbase, calendar="standard",
                      standard_name="time", long_name="valid time")),
        "level": ("level", np.asarray(LEVELS, dtype=np.float32),
                  dict(units="hPa", positive="down", axis="Z",
                       standard_name="air_pressure", long_name="pressure level")),
        "lat": ("lat", lat1d.astype(np.float32),
                dict(units="degrees_north", standard_name="latitude",
                     long_name="latitude", axis="Y")),
        "lon": ("lon", lon1d.astype(np.float32),
                dict(units="degrees_east", standard_name="longitude",
                     long_name="longitude", axis="X")),
    }
    out_ds = xr.Dataset(data_vars, coords=coords)
    out_ds.attrs.update(
        dict(
            Conventions="CF-1.8",
            title="HealDA analysis (regional/global, regridded to lat-lon)",
            source="NVIDIA HealDA (healda_ufs_era5.mdlus) via earth2studio",
            model="HealDA: Observation Encoder + HPX ViT, 330M params, nside=64 (HPX64)",
            observations="NOAA UFS GEFSv13 replay conventional obs (prepbufr), "
                         "24h window [-21h, +3h]",
            regrid=f"HEALPix nside=64 ({F.shape[-1]} pixels) -> {args.res} deg regular "
                   f"lat-lon, {args.method}"
                   + (f" (smooth={args.smooth_deg} deg)" if args.method == "idw" else ""),
            history=f"{dt.datetime.now():%Y-%m-%dT%H:%M:%S} "
                    f"healda_mac/06_to_latlon.py {src.name}",
            note="z is geopotential (m2 s-2); gh = z/9.80665 in gpm. "
                 "sst/sic over land are not masked (model output everywhere).",
            original_resolution="HEALPix nside=64 ~ 1 deg effective",
        )
    )

    enc = {v: dict(zlib=True, complevel=4, dtype="float32") for v in out_ds.data_vars}
    if args.format.startswith("NETCDF3"):
        enc = {v: dict(dtype="float32") for v in out_ds.data_vars}

    t0 = time.time()
    out_ds.to_netcdf(out, format=args.format, encoding=enc)
    size_mb = out.stat().st_size / 1e6
    print(f"\n[done] {out}  ({size_mb:.1f} MB, 总耗时 {time.time() - t_all:.1f}s)")
    print(f"       查看:  ncview {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
