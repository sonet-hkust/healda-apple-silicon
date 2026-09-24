#!/usr/bin/env python3
"""Build a minimal pure-torch `earth2grid` shim for Apple Silicon.

Why this exists
---------------
earth2studio's HealDA wrapper needs `earth2grid` for exactly two things:

  1. earth2grid.healpix.Grid(6, pixel_order=HEALPIX_PAD_XY).ang2pix(lon, lat)
     -> maps every observation to an HPX64 pixel index (REQUIRED)
  2. earth2grid.get_regridder(...) / latlon grid
     -> only used when lat_lon=True, which we never set

The official earth2grid wheels are CUDA12/x86_64/Linux only and building from
source needs nvcc, so it cannot be installed on macOS.  However the code path
we actually need (the XY pixel ordering) is pure torch + pure python:

    ang2pix -> coordinates.angular_to_global / global_to_face   (pure torch)
            -> _xyf_to_nest                                     (pure python)
            -> _nest2me -> nest2xy + xy2xy                      (torch + _bit_ops)

`healpix_bare` (the compiled extension) is only touched by the RING ordering,
`corners()` (visualisation) and `get_interp_weights()` (regridder) - none of
which we call.  So we vendor the four files we need and stub the two modules
that core.py merely imports at module level.

Source: https://github.com/NVlabs/earth2grid  (Apache-2.0)
"""

import base64
import json
import pathlib
import sys
import urllib.request

REPO = "NVlabs/earth2grid"
REV = "main"

HERE = pathlib.Path(__file__).parent
SHIM = HERE / "shim"
PKG = SHIM / "earth2grid"

# Files copied verbatim from upstream (Apache-2.0).
VENDOR = {
    "earth2grid/base.py": PKG / "base.py",
    "earth2grid/_bit_ops.py": PKG / "_bit_ops.py",
    "earth2grid/healpix/core.py": PKG / "healpix" / "core.py",
    "earth2grid/healpix/coordinates.py": PKG / "healpix" / "coordinates.py",
    "earth2grid/healpix/_padding/pure_python.py": PKG / "healpix" / "_padding" / "pure_python.py",
}

# physicsnemo probes for earth2grid with `importlib.metadata` (it never imports
# it to check the version), so the shim has to ship distribution metadata too,
# otherwise `check_version_spec("earth2grid", "0.1.0")` reports it as missing
# and the HPX tokenizer refuses to build.
DIST_INFO = PKG.parent / "earth2grid-0.1.0.dist-info"
METADATA = """Metadata-Version: 2.1
Name: earth2grid
Version: 0.1.0
Summary: Minimal pure-torch shim of NVlabs/earth2grid for Apple Silicon
License-File: LICENSE.txt
"""

INIT_PADDING = '''"""Minimal earth2grid.healpix._padding shim.

Upstream this package picks between a CUDA kernel (`cuda.py`) and a pure
torch/python indexing backend (`pure_python.py`).  Only the pure backend is
vendored here; on CPU/MPS upstream would have selected the indexing backend
anyway (see `_padding.__init__`: cuda is only the default when a CUDA device
is present).
"""

from earth2grid.healpix._padding.pure_python import pad, pad_with_dim  # noqa: F401

__all__ = ["pad", "pad_with_dim"]
'''

INIT_PKG = '''"""Minimal earth2grid shim (pure torch) - see build_shim.py."""

from earth2grid import base, _bit_ops, healpix_bare, _regrid  # noqa: F401
from earth2grid import healpix  # noqa: F401
from earth2grid.base import Grid  # noqa: F401

__all__ = ["base", "healpix", "Grid"]
'''

INIT_HEALPIX = '''"""Minimal earth2grid.healpix shim - only what HealDA needs."""

from earth2grid.healpix import coordinates  # noqa: F401
from earth2grid.healpix.core import (  # noqa: F401
    HEALPIX_PAD_XY,
    XY,
    Compass,
    Grid,
    PixelOrder,
)

__all__ = ["coordinates", "Grid", "HEALPIX_PAD_XY", "XY", "PixelOrder", "Compass"]
'''

STUB_HEALPIX_BARE = '''"""Stub for earth2grid.healpix_bare (compiled CUDA/C++ extension).

Upstream this wraps `_earth2grid_C`.  The XY pixel-ordering path used by
HealDA never calls into it, so an empty module is enough to satisfy the
module-level `from earth2grid import _bit_ops, healpix_bare` in core.py.
"""

def __getattr__(name):  # pragma: no cover - fail loudly if actually needed
    raise AttributeError(
        f"earth2grid.healpix_bare.{name} is not available: this is a stub for "
        "the compiled extension, which cannot be built on macOS."
    )
'''

STUB_REGRID = '''"""Stub for earth2grid._regrid (needs netCDF4/scipy + CUDA kernels).

Only `Regridder` is imported at module level by healpix/core.py.  HealDA runs
with lat_lon=False and therefore never builds a regridder.
"""


class Regridder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "earth2grid regridding is unavailable in this macOS shim; "
            "run HealDA with lat_lon=False."
        )


class BilinearInterpolator(Regridder):
    pass


class KNNS2Interpolator(Regridder):
    pass
'''


def fetch(path: str) -> str:
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={REV}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        meta = json.load(fh)
    return base64.b64decode(meta["content"]).decode("utf-8")


def main() -> int:
    (PKG / "healpix" / "_padding").mkdir(parents=True, exist_ok=True)

    for src, dst in VENDOR.items():
        text = fetch(src)
        dst.write_text(text, encoding="utf-8")
        print(f"vendored  {src:42s} -> {dst.relative_to(HERE)}  ({len(text.splitlines())} lines)")

    (PKG / "__init__.py").write_text(INIT_PKG, encoding="utf-8")
    (PKG / "healpix" / "__init__.py").write_text(INIT_HEALPIX, encoding="utf-8")
    (PKG / "healpix_bare.py").write_text(STUB_HEALPIX_BARE, encoding="utf-8")
    (PKG / "_regrid.py").write_text(STUB_REGRID, encoding="utf-8")
    (PKG / "healpix" / "_padding" / "__init__.py").write_text(INIT_PADDING, encoding="utf-8")
    print("\nwrote __init__.py, healpix/__init__.py, _padding/__init__.py,")
    print("      healpix_bare.py (stub), _regrid.py (stub)")

    DIST_INFO.mkdir(parents=True, exist_ok=True)
    (DIST_INFO / "METADATA").write_text(METADATA, encoding="utf-8")
    (DIST_INFO / "INSTALLER").write_text("shim\n", encoding="utf-8")
    (DIST_INFO / "RECORD").write_text("", encoding="utf-8")
    print(f"wrote distribution metadata -> {DIST_INFO.name}")
    print(f"\nshim ready at: {SHIM}")
    print("usage:  sys.path.insert(0, str(<healda_mac>/shim))  before importing earth2studio.models.da")
    return 0


if __name__ == "__main__":
    sys.exit(main())
