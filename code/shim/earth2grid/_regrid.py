"""Stub for earth2grid._regrid (needs netCDF4/scipy + CUDA kernels).

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
