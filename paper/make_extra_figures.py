#!/usr/bin/env python
"""Three supplementary figures requested in revision:

fig10_obs_composition.pdf  -- observation inventory bar chart (Table 1 data)
fig11_ro_cross_section.pdf -- latitude-height temperature-bias cross-sections
                              at 110 E for the four RO-ablation experiments
fig12_increment_map.pdf    -- conv+sat minus conv analysis increment at 500 hPa

Counts in fig10 are the exact fetch-time counts of the 2024-06-01 00 UTC
24-h window (see Table 1 of the manuscript).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

EA = os.path.join(ROOT, "outputs")
EXP = {
    "conv":     ("Conv (all)",        os.path.join(EA, "ea_conv.nc")),
    "noRO":     ("Conv w/o RO",       os.path.join(EA, "ea_conv_noro.nc")),
    "RO3ch":    ("RO 3-channel",      os.path.join(EA, "ea_ro3ch.nc")),
    "RObangle": ("RO bangle only",    os.path.join(EA, "ea_ro_bangle.nc")),
    "convsat":  ("Conv + Sat",        os.path.join(EA, "ea_convsat.nc")),
}
ERA5 = "/Users/hxf/Desktop/era5_2406_00.nc"


# ------------------------------------------------------- fig10: composition
def fig10():
    groups = [
        ("RO bending angle (gps)", 2680664, "#8E44AD"),
        ("RO background T (gps_t)", 2680664, "#9B59B6"),
        ("RO background q (gps_q)", 2680664, "#B07CC6"),
        ("Wind u", 1080489, "#2A6F97"),
        ("Wind v", 1080489, "#2A6F97"),
        ("Temperature", 617321, "#5B9BD5"),
        ("Surface pressure", 428354, "#5B9BD5"),
        ("Specific humidity", 91356, "#5B9BD5"),
        ("AMSU-A", 2136601, "#1B7F4B"),
        ("ATMS", 985798, "#1B7F4B"),
        ("MHS", 528756, "#1B7F4B"),
    ]
    labels = [g[0] for g in groups][::-1]
    vals = [g[1] for g in groups][::-1]
    cols = [g[2] for g in groups][::-1]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols, edgecolor="none", height=0.72)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(3e4, 9e6)
    ax.set_xlabel("report count (24-h window, log scale)")
    for yi, v in zip(y, vals):
        ax.text(v * 1.15, yi, f"{v:,}", va="center", fontsize=6.2, color="#333")
    ax.legend(handles=[
        Patch(color="#8E44AD", label="GNSS RO channels"),
        Patch(color="#2A6F97", label="in-situ / profiles"),
        Patch(color="#1B7F4B", label="microwave radiances")],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3,
        fontsize=6.6, frameon=False)
    ax.set_title("Observation composition, 2024-06-01 00 UTC")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(OUT, "fig10_obs_composition.pdf"))
    plt.close(fig)
    print("fig10 done")


# ------------------------------------------------- fig11: RO cross-sections
def fig11():
    e = xr.open_dataset(ERA5)
    elat = np.asarray(e.latitude.values, float)          # 43 -> 16 (descending)
    elon = np.asarray(e.longitude.values, float) % 360.0
    elev = np.asarray(e.pressure_level.values, float)
    j = int(np.argmin(np.abs(elon - 110.0)))
    era_t = np.asarray(e["t"].isel(valid_time=0).values, float)  # (lev, lat, lon)
    # reorder era lat to ascending to match analysis grid (16 -> 43)
    order = np.argsort(elat)
    era_t = era_t[:, order, :]
    lat_e = elat[order]

    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.15), sharey=True)
    lev = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200,
                    150, 100, 50], float)
    vmax = 4.0
    for ax, key in zip(axes, ["conv", "noRO", "RO3ch", "RObangle"]):
        d = xr.open_dataset(EXP[key][1])
        alat = np.asarray(d.lat.values, float)
        alon = np.asarray(d.lon.values, float) % 360.0
        jj = int(np.argmin(np.abs(alon - 110.0)))
        ana = np.asarray(d["t"].isel(time=0).values, float)      # (lev, lat, lon)
        # per-level bias at the fixed longitude -> (nlev, nlat)
        bias = np.full((len(lev), ana.shape[1]), np.nan)
        for il, p in enumerate(lev):
            ie = int(np.argmin(np.abs(elev - p)))
            bias[il] = ana[il, :, jj] - era_t[ie, :, jj]
        # interpolate to a smooth lat axis for contourf
        fine_lat = np.linspace(alat.min(), alat.max(), 160)
        fine_p = np.geomspace(1000, 50, 120)
        from scipy.interpolate import RegularGridInterpolator
        itp = RegularGridInterpolator((lev, alat), bias, bounds_error=False,
                                      fill_value=np.nan)
        LA, PE = np.meshgrid(fine_lat, fine_p)
        Z = itp(np.column_stack([PE.ravel(), LA.ravel()])).reshape(PE.shape)
        pc = ax.contourf(LA, PE, Z, levels=np.linspace(-vmax, vmax, 17),
                         cmap="RdBu_r", extend="both")
        ax.set_yscale("log")
        ax.set_ylim(1000, 50)
        ax.set_yticks([1000, 850, 700, 500, 300, 200, 150, 100, 50],
                      ["1000", "850", "700", "500", "300", "200", "150",
                       "100", "50"], fontsize=6.5)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{EXP[key][0]}", fontsize=8)
        ax.set_xlabel("latitude ($^\\circ$N)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("pressure (hPa)", fontsize=7.5)
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("$T$ analysis $-$ ERA5 (K)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.suptitle("Temperature bias cross-section at 110$^\\circ$E",
                 fontsize=9, y=1.04)
    fig.savefig(os.path.join(OUT, "fig11_ro_cross_section.pdf"))
    plt.close(fig)
    print("fig11 done")


# ------------------------------------------------ fig12: satellite increment
def fig12():
    a = xr.open_dataset(EXP["conv"][1])
    b = xr.open_dataset(EXP["convsat"][1])
    lat = np.asarray(a.lat.values, float)
    lon = np.asarray(a.lon.values, float)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35),
                             constrained_layout=False)
    specs = [("t", 500, "Temperature $t_{500}$", "K", 1.2),
             ("z", 500, "Geopotential $z_{500}$", "m$^2$ s$^{-2}$", 120)]
    for ax, (v, p, title, unit, scale) in zip(axes, specs):
        ia = int(np.argmin(np.abs(np.asarray(a.level.values) - p)))
        inc = (np.asarray(b[v].isel(time=0, level=ia).values, float)
               - np.asarray(a[v].isel(time=0, level=ia).values, float))
        vmax = np.percentile(np.abs(inc), 98)
        pc = ax.pcolormesh(LO, LA, inc, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto",
                           rasterized=True)
        ax.set_title(f"increment: {title} ({unit})", fontsize=8)
        ax.set_xlabel("longitude ($^\\circ$E)", fontsize=7.5)
        ax.set_ylabel("latitude ($^\\circ$N)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.015)
        cb.ax.tick_params(labelsize=6.5)
    fig.suptitle("Analysis increment of adding satellite radiances "
                 "(conv+sat $-$ conv, 500 hPa)", fontsize=9, y=1.03)
    fig.savefig(os.path.join(OUT, "fig12_increment_map.pdf"))
    plt.close(fig)
    print("fig12 done")


if __name__ == "__main__":
    fig10()
    fig11()
    fig12()
