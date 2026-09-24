#!/usr/bin/env python
"""Regenerate all publication figures + statistics tables for the HealDA paper.

Inputs  : native HEALPix analysis files in ../outputs/
Truth   : ERA5 pressure-level reanalysis (user supplied)
Outputs : figures/*.pdf  and  stats.csv / stats.json

All labels are English.  Figures are written as PDF with embedded TrueType
fonts (pdf.fonttype = 42) so that they merge cleanly into the LaTeX build.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "healda_mac"))
sys.path.insert(0, os.path.join(ROOT, "healda_mac", "shim"))

from compare_helper import VARS_3D, LEVELS_3D, load_era5, load_healda, interp_hpx, stats  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

ERA5 = "/Users/hxf/Desktop/era5_2406_00.nc"
WHEN = np.datetime64("2024-06-01T00:00:00")

# ----------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 200,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
})

VLAB = {"t": "Temperature $T$", "u": "Zonal wind $u$",
        "v": "Meridional wind $v$", "z": "Geopotential $z$",
        "q": "Specific humidity $q$"}
VUNIT = {"t": "K", "u": "m s$^{-1}$", "v": "m s$^{-1}$",
         "z": "m$^2$ s$^{-2}$", "q": "kg kg$^{-1}$"}
VSCALE = {"t": 1.0, "u": 1.0, "v": 1.0, "z": 1.0, "q": 1e6}   # q -> mg/kg
VSUNIT = {"t": "K", "u": "m s$^{-1}$", "v": "m s$^{-1}$",
          "z": "m$^2$ s$^{-2}$", "q": "mg kg$^{-1}$"}

EXP = {
    "conv":      "outputs/healda_20240601_conv.nc",
    "convsat":   "outputs/healda_20240601_convsat.nc",
    "noRO":      "outputs/healda_20240601_conv_noro.nc",
    "RO3ch":     "outputs/healda_20240601_ro3ch.nc",
    "RObangle":  "outputs/healda_20240601_ro_bangle.nc",
    "cpu":       "outputs/healda_20240601_cpu.nc",
    "mpsbroken": "/tmp/healda_trash/_corrupt_mps_pre_fix/healda_20240601_mps.nc",
}

NAME = {
    "cpu": "CPU reference",
    "mpsbroken": "MPS (before fix)",
    "conv": "MPS (after fix)",
    "convsat": "Conv + Sat",
    "noRO": "Conv, no RO",
    "RO3ch": "RO 3-channel",
    "RObangle": "RO bending angle only",
}


# --------------------------------------------------------------- data set
def build(names):
    """Return {var: {lev: {exp: 2D field}}} on the ERA5 grid (truth included)."""
    era5, glat, glon, glev = load_era5(ERA5)
    glon2d, glat2d = np.meshgrid(glon, glat)

    loaded = {}
    for n in names:
        f, idx_names, hlat, hlon = load_healda(os.path.join(ROOT, EXP[n]))
        loaded[n] = (f, {s: i for i, s in enumerate(idx_names)}, hlat, hlon)

    data = {}
    for v in VARS_3D:
        data[v] = {}
        for lev in LEVELS_3D:
            ch = f"{v}{lev}"
            if any(ch not in idx for _, idx, _, _ in loaded.values()):
                continue
            li = int(np.argmin(np.abs(glev - lev)))
            if abs(glev[li] - lev) > 1.0:
                continue
            entry = {}
            for n, (f, idx, hlat, hlon) in loaded.items():
                entry[n] = interp_hpx(f[idx[ch]], hlat, hlon, glat2d, glon2d)
            entry["era5"] = era5[v][li]
            data[v][lev] = entry
    return data, glat, glon


# ------------------------------------------------------------- figure 1
def fig_mps(data, glat, glon):
    """MPS failure and repair: 850 hPa temperature, 4 panels."""
    lev = 850
    v = "t"
    panels = [("era5", "ERA5 (truth)"), ("cpu", "CPU reference"),
              ("mpsbroken", "MPS, before fix"), ("conv", "MPS, after fix")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4),
                             constrained_layout=True)
    vmin, vmax = 280.0, 305.0
    for ax, (key, title) in zip(axes.ravel(), panels):
        fld = data[v][lev][key]
        p = ax.pcolormesh(glon, glat, fld, cmap="RdBu_r", vmin=vmin,
                          vmax=vmax, shading="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Longitude ($^\\circ$E)")
        ax.set_ylabel("Latitude ($^\\circ$N)")
        ax.tick_params(direction="in")
        if key != "era5":
            s = stats(fld, data[v][lev]["era5"])
            ax.text(0.03, 0.05,
                    f"RMSE = {s['rmse']:.2f} K\nr = {s['corr']:.3f}",
                    transform=ax.transAxes, fontsize=7.5,
                    bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.5))
    cb = fig.colorbar(p, ax=axes, orientation="horizontal", shrink=0.55,
                      aspect=40, pad=0.03, extend="both")
    cb.set_label(f"{VLAB[v]} at {lev} hPa (K)")
    fig.savefig(os.path.join(OUT, "fig1_mps_failure.pdf"))
    plt.close(fig)


def fig_mps_scatter(data):
    """CPU vs MPS(broken) and CPU vs MPS(fixed) scatter of 500 hPa geopotential."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    for ax, key, ttl in [(axes[0], "mpsbroken", "MPS before fix vs CPU"),
                         (axes[1], "conv", "MPS after fix vs CPU")]:
        a = np.asarray(data["z"][500][key]).ravel()
        b = np.asarray(data["z"][500]["cpu"]).ravel()
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        ax.plot(b, a, ".", ms=0.6, color="#2A6F97", alpha=0.35, rasterized=True)
        lo, hi = b.min(), b.max()
        ax.plot([lo, hi], [lo, hi], "-", color="0.3", lw=0.8)
        s = stats(a, b)
        ax.set_title(f"{ttl}\nr = {s['corr']:.4f}, RMSE = {s['rmse']:.1f} "
                     f"m$^2$ s$^{-2}$", fontsize=8.5)
        ax.set_xlabel("CPU: $z_{500}$ (m$^2$ s$^{-2}$)")
        ax.set_ylabel("MPS: $z_{500}$ (m$^2$ s$^{-2}$)")
        ax.tick_params(direction="in")
    fig.savefig(os.path.join(OUT, "fig2_mps_scatter.pdf"))
    plt.close(fig)


# ------------------------------------------------------- profile helpers
def _profile_axes(fig, ncol, ylog=True):
    axes = np.atleast_2d(np.array(fig.subplots(2, ncol, squeeze=False)))
    return axes


def profile_figure(data, exps, colors, fname, title_extra=""):
    """Two rows: RMSE (top) and anomaly correlation (bottom), one col per var."""
    fig, axes = plt.subplots(2, len(VARS_3D), figsize=(7.4, 4.6),
                             constrained_layout=True, sharey=True)
    levs = sorted(data["t"].keys(), reverse=True)
    for j, v in enumerate(VARS_3D):
        axr, axc = axes[0, j], axes[1, j]
        for e in exps:
            rm, co = [], []
            for lev in levs:
                s = stats(data[v][lev][e], data[v][lev]["era5"])
                rm.append(s["rmse"] * VSCALE[v])
                co.append(s["corr"])
            axr.plot(rm, levs, "-o", ms=2.6, color=colors[e], label=NAME[e])
            axc.plot(co, levs, "-o", ms=2.6, color=colors[e], label=NAME[e])
        axr.set_title(VLAB[v], fontsize=9)
        axc.set_xlabel("Correlation")
        axr.set_xlabel(f"RMSE ({VSUNIT[v]})")
        for ax in (axr, axc):
            ax.set_yscale("log")
            ax.set_ylim(max(levs) * 1.08, min(levs) * 0.92)
            ax.set_yticks([1000, 500, 200, 100, 50])
            ax.set_yticklabels(["1000", "500", "200", "100", "50"])
            ax.tick_params(direction="in")
            ax.grid(alpha=0.25, lw=0.4)
    axes[0, 0].set_ylabel("Pressure (hPa)")
    axes[1, 0].set_ylabel("Pressure (hPa)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(exps),
               bbox_to_anchor=(0.5, -0.055), frameon=False)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


# ------------------------------------------------------------ spatial map
def spatial_figure(data, glat, glon, cases, fname, levels=(850, 500),
                   var=("t", "z")):
    """Rows: configurations; columns: (var@level) error maps."""
    nrow, ncol = len(cases), 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.55 * nrow + 1.2),
                             constrained_layout=True, squeeze=False)
    for i, (e, ttl) in enumerate(cases):
        for j, (v, lev) in enumerate(zip(var, levels)):
            ax = axes[i, j]
            d = np.asarray(data[v][lev][e]) - np.asarray(data[v][lev]["era5"])
            d = d * VSCALE[v]
            lim = np.nanpercentile(np.abs(d), 99)
            lim = max(lim, 1e-9)
            p = ax.pcolormesh(glon, glat, d, cmap="RdBu_r",
                              vmin=-lim, vmax=lim, shading="auto")
            ax.tick_params(direction="in")
            if i == 0:
                ax.set_title(f"{VLAB[v]} @ {lev} hPa", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ttl}\nLatitude ($^\\circ$N)", fontsize=8)
            else:
                ax.set_ylabel("")
            if i == nrow - 1:
                ax.set_xlabel("Longitude ($^\\circ$E)")
            cb = fig.colorbar(p, ax=ax, shrink=0.85, aspect=18, pad=0.015,
                              extend="both")
            cb.set_label(VSUNIT[v], fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
            s = stats(data[v][lev][e], data[v][lev]["era5"])
            ax.text(0.02, 0.04, f"RMSE {s['rmse'] * VSCALE[v]:.2f}",
                    transform=ax.transAxes, fontsize=6.8,
                    bbox=dict(fc="white", ec="0.6", lw=0.4, pad=1.8))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def improvement_figure(data, glat, glon, base, new, fname):
    """RMSE difference maps: base -> new, for t@850 and z@500."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), constrained_layout=True)
    for ax, (v, lev) in zip(axes, [("t", 850), ("z", 500)]):
        db = np.abs(np.asarray(data[v][lev][base]) - np.asarray(data[v][lev]["era5"]))
        dn = np.abs(np.asarray(data[v][lev][new]) - np.asarray(data[v][lev]["era5"]))
        imp = (db - dn) * VSCALE[v]
        lim = np.nanpercentile(np.abs(imp), 98)
        p = ax.pcolormesh(glon, glat, imp, cmap="RdBu",
                          vmin=-lim, vmax=lim, shading="auto")
        ax.set_title(f"{VLAB[v]} @ {lev} hPa: error reduction\n"
                     f"RMSE({NAME[base]}) $-$ RMSE({NAME[new]})", fontsize=8.5)
        ax.set_xlabel("Longitude ($^\\circ$E)")
        ax.set_ylabel("Latitude ($^\\circ$N)")
        ax.tick_params(direction="in")
        cb = fig.colorbar(p, ax=ax, shrink=0.9, aspect=18, pad=0.02,
                          extend="both")
        cb.set_label(f"RMSE reduction ({VSUNIT[v]})", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


# ------------------------------------------------------------- zonal mean
def zonal_figure(data, glat, exps, colors, fname, lev=500, var="z"):
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4), constrained_layout=True)
    tr = np.nanmean(np.asarray(data[var][lev]["era5"]), axis=1)
    ax.plot(glat, tr / 9.80665, "-", color="k", lw=1.4, label="ERA5")
    for e in exps:
        f = np.nanmean(np.asarray(data[var][lev][e]), axis=1)
        ax.plot(glat, f / 9.80665, "-", color=colors[e], lw=1.1, label=NAME[e])
    ax.set_xlabel("Latitude ($^\\circ$N)")
    ax.set_ylabel(f"Zonal-mean $z_{{{lev}}}$ (gpm)")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(f"Zonal mean, {VLAB[var]} at {lev} hPa", fontsize=9)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def zonal_bias_figure(data, glat, exps, colors, fname, lev=500, var="z"):
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4), constrained_layout=True)
    tr = np.nanmean(np.asarray(data[var][lev]["era5"]), axis=1)
    ax.axhline(0, color="k", lw=0.8)
    for e in exps:
        f = np.nanmean(np.asarray(data[var][lev][e]), axis=1)
        ax.plot(glat, (f - tr) / 9.80665, "-", color=colors[e], lw=1.2,
                label=NAME[e])
    ax.set_xlabel("Latitude ($^\\circ$N)")
    ax.set_ylabel(f"Zonal-mean bias, ${var}_{{{lev}}}$ (gpm)")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(direction="in")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(f"Bias vs ERA5, {VLAB[var]} at {lev} hPa", fontsize=9)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


# ------------------------------------------------------------ stats table
def dump_stats(data, path_csv, path_json):
    rows = []
    store = {}
    for v in VARS_3D:
        for lev in sorted(data[v].keys(), reverse=True):
            for e in data[v][lev]:
                if e == "era5":
                    continue
                s = stats(data[v][lev][e], data[v][lev]["era5"])
                rows.append(dict(variable=v, level=lev, exp=e,
                                 rmse=s["rmse"] * VSCALE[v],
                                 corr=s["corr"], bias=s["bias"] * VSCALE[v],
                                 stdratio=s["stdratio"]))
                store[f"{v}{lev}|{e}"] = s
    import csv
    with open(path_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(path_json, "w") as fh:
        json.dump(store, fh, indent=1)

    # layer-mean summary
    print("\n=== layer-mean (13 levels) ===")
    print(f"{'exp':10s} " + " ".join(f"{v:>18s}" for v in VARS_3D))
    for e in ["cpu", "mpsbroken", "conv", "convsat", "noRO", "RO3ch", "RObangle"]:
        cells = []
        for v in VARS_3D:
            rm = np.mean([store[f"{v}{l}|{e}"]["rmse"] * VSCALE[v]
                          for l in data[v] if f"{v}{l}|{e}" in store])
            co = np.mean([store[f"{v}{l}|{e}"]["corr"]
                          for l in data[v] if f"{v}{l}|{e}" in store])
            cells.append(f"{rm:9.3f}/{co:6.3f}")
        print(f"{e:10s} " + " ".join(f"{c:>18s}" for c in cells))


# -------------------------------------------------------------------- main
def set_names(**kw):
    """Per-figure display-name overrides (mutates the module-level NAME dict)."""
    NAME.update(kw)


def main():
    names = [n for n in EXP if os.path.exists(os.path.join(ROOT, EXP[n]))]
    print("experiments:", names)
    data, glat, glon = build(names)

    C = {"cpu": "#444444", "mpsbroken": "#C0392B", "conv": "#2A6F97",
         "convsat": "#1B7F4B", "noRO": "#E08A1E", "RO3ch": "#8E44AD",
         "RObangle": "#7F8C8D"}

    fig_mps(data, glat, glon)
    fig_mps_scatter(data)

    # ---- satellite increment block
    set_names(conv="Conv only", convsat="Conv + Sat")
    profile_figure(data, ["conv", "convsat"], C, "fig3_profile_sat.pdf")
    spatial_figure(data, glat, glon,
                   [("conv", "Conv only"), ("convsat", "Conv + Sat")],
                   "fig4_spatial_sat.pdf")
    improvement_figure(data, glat, glon, "conv", "convsat",
                       "fig5_improvement_sat.pdf")
    zonal_figure(data, glat, ["conv", "convsat"], C, "fig8_zonal_z500.pdf")

    # ---- RO ablation block
    set_names(conv="Conv (all)", noRO="Conv w/o RO",
              RO3ch="RO 3-channel", RObangle="RO bangle only")
    profile_figure(data, ["conv", "noRO", "RO3ch", "RObangle"], C,
                   "fig6_profile_ro.pdf")
    spatial_figure(data, glat, glon,
                   [("conv", "Conv (all)"), ("noRO", "Conv w/o RO"),
                    ("RO3ch", "RO 3-channel"), ("RObangle", "RO bangle only")],
                   "fig7_spatial_ro.pdf", levels=(300, 850), var=("t", "t"))
    zonal_bias_figure(data, glat, ["conv", "convsat", "noRO", "RO3ch"], C,
                      "fig9_zonal_bias_z500.pdf")

    dump_stats(data, os.path.join(HERE, "stats.csv"),
               os.path.join(HERE, "stats.json"))
    print("\nfigures ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()
