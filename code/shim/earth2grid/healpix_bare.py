"""Stub for earth2grid.healpix_bare (compiled CUDA/C++ extension).

Upstream this wraps `_earth2grid_C`.  The XY pixel-ordering path used by
HealDA never calls into it, so an empty module is enough to satisfy the
module-level `from earth2grid import _bit_ops, healpix_bare` in core.py.
"""

def __getattr__(name):  # pragma: no cover - fail loudly if actually needed
    raise AttributeError(
        f"earth2grid.healpix_bare.{name} is not available: this is a stub for "
        "the compiled extension, which cannot be built on macOS."
    )
