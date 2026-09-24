#!/usr/bin/env python3
"""把任意多组 HealDA 分析场与 ERA5 真值做多维度比较。

维度
----
1. 垂直维度  fig_profile.png  —— 逐层 corr / RMSE / std 比 / 相对基线的改善率
2. 水平维度  fig_spatial.png  —— 关键层各实验的绝对误差 + 相对基线的改善
3. 剖面维度  fig_section.png  —— 纬度-高度剖面 + 整层汇总

第一组实验被当作基线（baseline），其余各组都与它比较。

用法
----
    .venv/bin/python healda_mac/07_compare_multi.py \
        --exp conv=outputs/healda_20240601_conv.nc \
        --exp convsat=outputs/healda_20240601_convsat.nc \
        --era5 /Users/hxf/Desktop/era5_2406_00.nc \
        --outdir outputs
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "shim"))
from earth2grid import healpix  # noqa: E402

LEVELS_3D = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
VARS_3D = ["t", "u", "v", "z", "q"]
VAR_UNIT = {"t": "K", "u": "m/s", "v": "m/s", "z": "m2/s2", "q": "kg/kg"}
VAR_LABEL = {"t": "temperature", "u": "zonal wind", "v": "meridional wind",
             "z": "geopotential", "q": "specific humidity"}
MAP_FIELDS = [("t", 850), ("q", 850), ("u", 500), ("v", 500), ("z", 500)]

# 各实验配色（第一组=基线，用蓝）
PALETTE = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#BA7517", "#D4537E"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


# ------------------------------------------------------------------ I/O
def load_healda(path: str):
    """返回 (field[74,npix], channel_names, lat[npix], lon[npix])，lon 归一到 [0,360)。"""
    ds = xr.open_dataset(path)
    key = list(ds.data_vars)[0]
    field = np.asarray(ds[key])[0]
    names = [str(v) for v in ds["variable"].values]
    grid = healpix.Grid(6, pixel_order=healpix.HEALPIX_PAD_XY)
    lat = np.asarray(grid.lat, dtype=np.float64)
    lon = np.asarray(grid.lon, dtype=np.float64) % 360.0
    return field, names, lat, lon


def load_era5(path: str, when=None):
    ds = xr.open_dataset(path)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64) % 360.0
    lev = np.asarray(ds["pressure_level"].values, dtype=np.float64)
    tsel = 0 if when is None else int(np.argmin(
        np.abs(ds["valid_time"].values - np.datetime64(when))))
    out = {}
    for v in VARS_3D:
        out[v] = np.asarray(ds[v].isel(valid_time=tsel).values, dtype=np.float64)
    return out, lat, lon, lev


def interp_hpx(vals, hlat, hlon, glat2d, glon2d):
    pts = np.column_stack([hlon, hlat])
    out = griddata(pts, vals, (glon2d, glat2d), method="linear")
    bad = np.isnan(out)
    if bad.any():
        out[bad] = griddata(pts, vals, (glon2d[bad], glat2d[bad]), method="nearest")
    return out


def stats(a, e):
    a, e = np.asarray(a).ravel(), np.asarray(e).ravel()
    m = np.isfinite(a) & np.isfinite(e)
    a, e = a[m], e[m]
    d = a - e
    return dict(bias=float(d.mean()),
                rmse=float(np.sqrt((d ** 2).mean())),
                corr=float(np.corrcoef(a, e)[0, 1]),
                stdratio=float(a.std() / (e.std() + 1e-30)))


# ------------------------------------------------------------------ 计算
def build(exps: dict[str, str], era5_nc, when=None):
    """exps: {实验名: nc 路径}，第一个为基线。返回 data, glat, glon。"""
    era5, glat, glon, glev = load_era5(era5_nc, when)
    glon2d, glat2d = np.meshgrid(glon, glat)

    loaded = {}
    for name, path in exps.items():
        f, names, hlat, hlon = load_healda(path)
        loaded[name] = (f, {n: i for i, n in enumerate(names)})

    data: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for v in VARS_3D:
        data[v] = {}
        for lev in LEVELS_3D:
            ch = f"{v}{lev}"
            if any(ch not in idx for _, idx in loaded.values()):
                continue
            li = int(np.argmin(np.abs(glev - lev)))
            if abs(glev[li] - lev) > 1.0:
                continue
            entry = {
                name: interp_hpx(f[idx[ch]], hlat, hlon, glat2d, glon2d)
                for name, (f, idx) in loaded.items()
            }
            entry["era5"] = era5[v][li]
            data[v][lev] = entry
    return data, glat, glon


def style(ax, levs_all, log=True, yticks=(1000, 500, 200, 100, 50)):
    if log:
        ax.set_yscale("log")
        ax.set_ylim(max(levs_all) * 1.08, min(levs_all) * 0.92)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


# ------------------------------------------------------------------ 图 1
def fig_profile(data, names, out, title=""):
    base = names[0]
    rows, cols = len(VARS_3D), 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.0 * rows),
                             sharex="col")
    levs_all = sorted({lv for v in data for lv in data[v]})

    for r, v in enumerate(VARS_3D):
        lv = sorted(data[v], reverse=True)
        if not lv:
            for c in range(cols):
                axes[r][c].axis("off")
            continue
        st = {n: [stats(data[v][l][n], data[v][l]["era5"]) for l in lv] for n in names}
        r0 = np.array([s["rmse"] for s in st[base]])

        specs = [
            ("corr with ERA5", lambda s: s["corr"], None),
            (f"RMSE ({VAR_UNIT[v]})", lambda s: s["rmse"], None),
            ("std ratio (analysis / ERA5)", lambda s: s["stdratio"], 1.0),
        ]
        for c, (tt, get, ref) in enumerate(specs):
            ax = axes[r][c]
            for k, n in enumerate(names):
                ax.plot([get(s) for s in st[n]], lv, marker=MARKERS[k % 6],
                        color=PALETTE[k % 6], lw=1.6, ms=4, label=n)
            if ref is not None:
                ax.axvline(ref, color="#888780", lw=0.8, ls="--")
            style(ax, levs_all)
            if r == 0:
                ax.set_title(tt, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{v}  ({VAR_LABEL[v]})\nhPa", fontsize=9)

        # 第 4 列：相对基线的 RMSE 改善率
        ax = axes[r][3]
        xmax = 1.0
        for k, n in enumerate(names[1:], start=1):
            imp = (r0 - np.array([s["rmse"] for s in st[n]])) / np.maximum(r0, 1e-30) * 100
            ax.plot(imp, lv, marker=MARKERS[k % 6], color=PALETTE[k % 6],
                    lw=1.6, ms=4, label=n)
            xmax = max(xmax, float(np.nanmax(np.abs(imp))) * 1.15)
        ax.axvline(0, color="#888780", lw=0.8)
        style(ax, levs_all)
        ax.set_xlim(-max(12.0, xmax * 0.35), xmax)
        if r == 0:
            ax.set_title(f"RMSE reduced vs {base} (%)", fontsize=10)

    handles = [plt.Line2D([], [], color=PALETTE[k % 6], marker=MARKERS[k % 6],
                          label=n, lw=1.6) for k, n in enumerate(names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), fontsize=10,
               frameon=False)
    fig.suptitle(title or "Vertical structure against ERA5 (2024-06-01 00Z)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out, dpi=135)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------ 图 2
def fig_spatial(data, names, glat, glon, out, title=""):
    base = names[0]
    rows = len(MAP_FIELDS)
    cols = len(names) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.9 * rows),
                             squeeze=False)
    ext = [glon.min(), glon.max(), glat.min(), glat.max()]

    for r, (v, lev) in enumerate(MAP_FIELDS):
        if lev not in data.get(v, {}):
            for c in range(cols):
                axes[r][c].axis("off")
            continue
        d = data[v][lev]
        err = {n: np.abs(d[n] - d["era5"]) for n in names}
        vmax = max(e.max() for e in err.values()) or 1e-12
        panels = [(f"|{n} - ERA5|  {v}{lev}", err[n], 0, vmax, "Reds")
                  for n in names]
        imp = err[base] - err[names[-1]]
        lim = max(np.abs(imp).max(), 1e-12)
        panels.append((f"error reduced: {names[-1]} vs {base}  {v}{lev}",
                       imp, -lim, lim, "coolwarm"))
        for c, (tt, f_, vmin, vmax_, cmap) in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(f_, origin="lower", extent=ext, aspect="auto",
                           cmap=cmap, vmin=vmin, vmax=vmax_)
            ax.set_title(tt, fontsize=9.5)
            ax.tick_params(labelsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            if r == rows - 1:
                ax.set_xlabel("lon", fontsize=9)
            if c == 0:
                ax.set_ylabel("lat", fontsize=9)

    fig.suptitle(title or "Horizontal absolute error vs ERA5", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=135)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------ 图 3
def fig_section(data, names, glat, glon, out, title=""):
    base = names[0]
    others = names[1:]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    # (0,0) 整层平均：相对基线的 RMSE 降幅（分组柱状图）
    ax = axes[0][0]
    labels = [f"{v}\n({VAR_UNIT[v]})" for v in VARS_3D]
    x = np.arange(len(VARS_3D))
    w = 0.8 / max(len(others), 1)
    for k, n in enumerate(others):
        gains = []
        for v in VARS_3D:
            lv = sorted(data.get(v, {}), reverse=True)
            if not lv:
                gains.append(np.nan)
                continue
            a = np.mean([stats(data[v][l][base], data[v][l]["era5"])["rmse"] for l in lv])
            b = np.mean([stats(data[v][l][n], data[v][l]["era5"])["rmse"] for l in lv])
            gains.append((a - b) / a * 100.0)
        ax.bar(x + k * w - 0.4 + w / 2, gains, width=w * 0.92,
               color=PALETTE[(k + 1) % 6], label=n)
    ax.axhline(0, color="#5F5E5A", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE reduction (%)", fontsize=9)
    ax.set_title(f"Layer-mean gain vs {base}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # (0,1) 逐层 RMSE 比值（相对基线）
    ax = axes[0][1]
    for v, col, mk in (("t", "#D85A30", "o"), ("z", "#534AB7", "s"), ("q", "#0F6E56", "^")):
        lv = sorted(data.get(v, {}), reverse=True)
        if not lv:
            continue
        rc = np.array([stats(data[v][l][base], data[v][l]["era5"])["rmse"] for l in lv])
        for k, n in enumerate(others):
            rs = np.array([stats(data[v][l][n], data[v][l]["era5"])["rmse"] for l in lv])
            ax.plot((rs / np.maximum(rc, 1e-30)) * 100.0, lv, marker=mk, ms=3.5,
                    color=col, ls=["-", "--", ":"][k % 3],
                    label=f"{v} ({n})" if len(others) > 1 else v)
    ax.axvline(100, color="#5F5E5A", lw=0.9, ls="--")
    style(ax, LEVELS_3D)
    ax.set_xlabel(f"RMSE / RMSE({base})  (%)", fontsize=9)
    ax.set_ylabel("hPa", fontsize=9)
    ax.set_title("Ratio < 100% means improvement", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(labelsize=8)

    # (0,2) z500 纬向平均
    ax = axes[0][2]
    if 500 in data.get("z", {}):
        d = data["z"][500]
        ax.plot(glat, d["era5"].mean(axis=1) / 9.80665, color="#2C2C2A",
                lw=2.0, label="ERA5")
        for k, n in enumerate(names):
            ax.plot(glat, d[n].mean(axis=1) / 9.80665, color=PALETTE[k % 6],
                    lw=1.4, ls="--" if k else "-", marker=MARKERS[k % 6], ms=3, label=n)
        ax.set_ylabel("z500 geopotential height (gpm)", fontsize=9)
        ax.legend(fontsize=8)
    ax.set_xlabel("lat", fontsize=9)
    ax.set_title("Zonal mean z500", fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)

    # (1,*) 纬度-高度剖面：最后一组相对基线的绝对误差削减
    ref = names[-1] if len(names) > 1 else base
    for k, v in enumerate(["t", "u", "q"]):
        ax = axes[1][k]
        lv = sorted(data.get(v, {}), reverse=True)
        if not lv:
            ax.axis("off")
            continue
        grid = np.array([np.abs(data[v][l][base] - data[v][l]["era5"]).mean(axis=1)
                         - np.abs(data[v][l][ref] - data[v][l]["era5"]).mean(axis=1)
                         for l in lv])
        vmax = max(np.abs(grid).max(), 1e-12)
        im = ax.contourf(glat, lv, grid, levels=21, cmap="coolwarm",
                         vmin=-vmax, vmax=vmax)
        style(ax, LEVELS_3D)
        ax.set_xlabel("lat", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)
        ax.set_title(f"{v}: zonal-mean |err| drop, {ref} vs {base} ({VAR_UNIT[v]})",
                     fontsize=9.5)
        ax.tick_params(labelsize=8)

    fig.suptitle(title or "Sections and summary", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=135)
    plt.close(fig)
    print(f"[saved] {out}")


# ------------------------------------------------------------------ 表
def report(data, names):
    base = names[0]
    head = f"{'var':>4s} {'hPa':>6s}"
    for n in names:
        head += f" {('c_' + n)[:9]:>9s}"
    for n in names[1:]:
        head += f" {('r_' + n)[:9]:>9s}"
    for n in names[1:]:
        head += f" {('g%' + n)[:8]:>8s}"
    print("\n" + "=" * len(head))
    print(head)
    print("=" * len(head))
    agg = {n: {} for n in names}
    for v in VARS_3D:
        for lev in sorted(data.get(v, {}), reverse=True):
            s = {n: stats(data[v][lev][n], data[v][lev]["era5"]) for n in names}
            line = f"{v:>4s} {lev:6d}"
            for n in names:
                line += f" {s[n]['corr']:9.4f}"
            for n in names[1:]:
                line += f" {s[n]['rmse']:9.4g}"
            for n in names[1:]:
                g = (s[base]["rmse"] - s[n]["rmse"]) / max(s[base]["rmse"], 1e-30) * 100
                line += f" {g:+8.2f}"
                agg[n].setdefault(v, []).append((s[base], s[n], g))
            print(line)
    print("-" * len(head))
    for n in names[1:]:
        for v, rows in agg[n].items():
            mc = np.mean([r[0]["corr"] for r in rows])
            ms = np.mean([r[1]["corr"] for r in rows])
            rc = np.mean([r[0]["rmse"] for r in rows])
            rs = np.mean([r[1]["rmse"] for r in rows])
            g = (rc - rs) / max(rc, 1e-30) * 100.0
            print(f"{v:>4s} {'mean':>6s} corr {mc:.4f} -> {ms:.4f}   "
                  f"rmse {rc:.4g} -> {rs:.4g}  ({VAR_UNIT[v]})   gain {g:+7.2f}%  [{n}]")
    print("=" * len(head))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True, metavar="NAME=PATH",
                    help="实验，可重复；第一个作为基线。例: --exp conv=a.nc --exp ro=b.nc")
    ap.add_argument("--era5", default="/Users/hxf/Desktop/era5_2406_00.nc")
    ap.add_argument("--time", default=None)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖上一次的图")
    ap.add_argument("--title", default="", help="图标题")
    args = ap.parse_args()

    exps = {}
    for e in args.exp:
        if "=" not in e:
            raise SystemExit(f"--exp 需写成 NAME=PATH，收到: {e}")
        k, v = e.split("=", 1)
        exps[k] = v
    names = list(exps)
    print(f"[exp] baseline = {names[0]}   比较组 = {names[1:]}")

    data, glat, glon = build(exps, args.era5, args.time)
    print(f"[grid] ERA5 domain: lat {glat.min():.2f}..{glat.max():.2f} "
          f"({len(glat)}), lon {glon.min():.2f}..{glon.max():.2f} ({len(glon)})")
    print(f"[grid] levels: {sorted({l for v in data for l in data[v]}, reverse=True)}")

    report(data, names)
    od = pathlib.Path(args.outdir)
    tag = f"_{args.tag}" if args.tag else ""
    fig_profile(data, names, od / f"fig_profile{tag}.png", args.title)
    fig_spatial(data, names, glat, glon, od / f"fig_spatial{tag}.png", args.title)
    fig_section(data, names, glat, glon, od / f"fig_section{tag}.png", args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
