"""Minimal earth2grid.healpix._padding shim.

Upstream this package picks between a CUDA kernel (`cuda.py`) and a pure
torch/python indexing backend (`pure_python.py`).  Only the pure backend is
vendored here; on CPU/MPS upstream would have selected the indexing backend
anyway (see `_padding.__init__`: cuda is only the default when a CUDA device
is present).
"""

from earth2grid.healpix._padding.pure_python import pad, pad_with_dim  # noqa: F401

__all__ = ["pad", "pad_with_dim"]
