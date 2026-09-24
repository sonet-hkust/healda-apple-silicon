#!/usr/bin/env python
"""Shared I/O and verification metrics used by the figure scripts.

Self-contained (depends only on xarray / numpy / scipy / earth2grid shim) so
that the paper directory can be version-controlled independently.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "healda_mac"), os.path.join(ROOT, "healda_mac", "shim")):
    if p not in sys.path:
        sys.path.insert(0, p)

from earth2grid import healpix  # noqa: E402

LEVELS_3D = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
VARS_3D = ["t", "u", "v", "z", "q"]


def load_healda(path: str):
    """Return (field[74, npix], channel_names, lat[npix], lon[npix])."""
    ds = xr.open_dataset(path)
    key = list(ds.data_vars)[0]
    field = np.asarray(ds[key])[0]
    names = [str(v) for v in ds["variable"].values]
    grid = healpix.Grid(6, pixel_order=healpix.HEALPIX_PAD_XY)
    lat = np.asarray(grid.lat, dtype=np.float64)
    lon = np.asarray(grid.lon, dtype=np.float64) % 360.0
    return field, names, lat, lon


def load_era5(path: str, when=None):
    """Return {var: 2D[lat, lon]}, lat, lon, levels (lat kept descending)."""
    ds = xr.open_dataset(path)
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["longitude"].values, dtype=np.float64) % 360.0
    lev = np.asarray(ds["pressure_level"].values, dtype=np.float64)
    tsel = 0 if when is None else int(
        np.argmin(np.abs(ds["valid_time"].values - np.datetime64(when))))
    out = {}
    for v in VARS_3D:
        out[v] = np.asarray(ds[v].isel(valid_time=tsel).values, dtype=np.float64)
    return out, lat, lon, lev


def interp_hpx(vals, hlat, hlon, glat2d, glon2d):
    """Scatter-interpolate a HEALPix field onto a regular lat-lon grid."""
    pts = np.column_stack([hlon, hlat])
    out = griddata(pts, vals, (glon2d, glat2d), method="linear")
    bad = np.isnan(out)
    if bad.any():
        out[bad] = griddata(pts, vals, (glon2d[bad], glat2d[bad]),
                            method="nearest")
    return out


def stats(a, e):
    """Bias, RMSE, anomaly correlation and std ratio of `a` against truth `e`."""
    a, e = np.asarray(a).ravel(), np.asarray(e).ravel()
    m = np.isfinite(a) & np.isfinite(e)
    a, e = a[m], e[m]
    d = a - e
    return dict(bias=float(d.mean()),
                rmse=float(np.sqrt((d ** 2).mean())),
                corr=float(np.corrcoef(a, e)[0, 1]),
                stdratio=float(a.std() / (e.std() + 1e-30)))
