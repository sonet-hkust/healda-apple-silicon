#!/usr/bin/env python3
"""Workarounds for running HealDA (physicsnemo DiT) on Apple MPS.

Problem
-------
HealDA is an *unconditional* regression model: earth2studio builds
``class_labels = torch.empty([B, 0])`` and the DiT's conditioning embedder is a
``ZeroConditioningEmbedder``.  Consequently the adaLN-Zero modulation layers are
``nn.Linear(in_features=0, out_features=6144)`` — 24 blocks + 1 detokenizer,
25 layers in total.

On Apple's Metal backend, running a *forward* through a Linear whose weight has
``numel() == 0`` aborts the whole process inside
``MPSNDArray initWithDevice:descriptor:isTextureBacked:`` with
``Error: device may not be nil``  (SIGABRT, no Python traceback).
Moving such a weight to MPS is harmless; only the forward pass crashes.

Why the patch is numerically exact
----------------------------------
With ``in_features == 0`` the matmul contributes nothing, so

    y = x @ W.T + b  ==  b        (broadcast over the batch dim)

i.e. the layer is a *constant* function of its input.  Evaluating it anywhere
(CPU, GPU, fp32, fp64) gives bit-identical results.  We therefore route only
these degenerate layers through CPU and leave everything else on Metal.

Usage
-----
    from mps_patch import patch_for_mps
    model = model.to("mps")
    n = patch_for_mps(model)      # returns number of layers patched
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _CPUConstantLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear(0, out)`` that never touches MPS.

    The weight/bias are pinned to CPU: ``_apply`` is overridden so that a
    parent ``.to("mps")`` cannot drag them onto Metal.
    """

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        self.in_features = int(lin.in_features)
        self.out_features = int(lin.out_features)
        self.register_buffer("_w", lin.weight.detach().to("cpu").clone())
        if lin.bias is not None:
            self.register_buffer("_b", lin.bias.detach().to("cpu").clone())
        else:
            self._b = None  # type: ignore[assignment]

    # keep the parameters on CPU no matter what the parent module does
    def _apply(self, fn, recurse: bool = True):  # noqa: ANN001, ANN202, D102
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = torch.nn.functional.linear(x.to("cpu"), self._w, self._b)
        return out.to(x.device)

    def extra_repr(self) -> str:  # noqa: D102
        return f"in_features=0, out_features={self.out_features} (cpu-pinned)"


def _iter_named_modules(root: nn.Module):
    """Yield (parent_module, attribute_name, child_module) triples."""
    yield None, "", root  # type: ignore[misc]
    for name, mod in root.named_modules():
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            yield root.get_submodule(parent_name), attr, mod
        elif name:
            yield root, name, mod


def patch_for_mps(model: nn.Module, verbose: bool = True) -> int:
    """Replace every ``nn.Linear`` with ``in_features == 0`` by a CPU-pinned stub.

    Call this *after* ``model.to(device)``.
    """
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent, attr, mod in _iter_named_modules(model):
        if parent is None:
            continue
        if isinstance(mod, nn.Linear) and mod.in_features == 0:
            targets.append((parent, attr, mod))

    for parent, attr, lin in targets:
        setattr(parent, attr, _CPUConstantLinear(lin))

    if verbose:
        for parent, attr, lin in targets:
            print(f"    [mps-patch] {attr}: Linear(0, {lin.out_features}) -> cpu-pinned stub")
        print(f"    [mps-patch] patched {len(targets)} zero-input Linear layers")
    return len(targets)


def patch_build_input_on_cpu(model) -> bool:
    """让 HealDA.build_input() 在 CPU 上构造观测输入, 再同步搬回设备。

    为什么需要
    ----------
    在 Apple Metal 上, build_input 把 11M 行的观测列逐列
    ``.to(device, non_blocking=True)`` 搬上去, 其中 ``target_time``
    (epoch 秒, int64) 会被后续拷贝污染 —— 实测 MPS 上它的值变成了观测时间
    的纳秒值, 于是 ``target_time * 1e9`` 在 int64 里溢出,
    ``float_metadata`` 里的相对时间特征变成 1e8 量级的垃圾,
    整个观测编码器喂进去的是错的时间信息。

    表现: 分析场与 ERA5 的相关性从 0.95 掉到 0.45, 风场 v 甚至到 0。
    且只在观测规模大时才暴露(两万条时完全正常), 极易被误判成精度问题。

    做法: 临时把 ``model.device_buffer`` 换成 CPU 上的空张量, 使
    ``model.device`` 返回 cpu, 构造完再同步搬回真实设备。
    """
    cls = type(model)
    orig = cls.build_input
    if getattr(orig, "_cpu_patched", False):
        return False

    def build_input(self, obs_dict, request_time):  # noqa: ANN001, ANN202
        real_buf = self.device_buffer
        real_dev = real_buf.device
        if real_dev.type == "cpu":
            return orig(self, obs_dict, request_time)
        self.device_buffer = torch.empty(0, device="cpu")
        try:
            out = orig(self, obs_dict, request_time)
        finally:
            self.device_buffer = real_buf
        dev = real_dev
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in out.items()}

    build_input._cpu_patched = True
    cls.build_input = build_input
    return True


def assert_no_zero_input_linear(model: nn.Module) -> None:
    """Sanity check: no real ``nn.Linear(0, ...)`` may remain on-device."""
    bad = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.in_features == 0 and not isinstance(m, _CPUConstantLinear)
    ]
    if bad:
        raise RuntimeError(f"unpatched zero-input Linear layers remain: {bad}")


if __name__ == "__main__":  # pragma: no cover - micro reproducer
    lin = nn.Linear(0, 8)
    print("plain Linear(0,8) on mps:", end=" ", flush=True)
    try:
        y = lin.to("mps")(torch.empty(1, 0, device="mps"))
        print("OK", tuple(y.shape))
    except BaseException as exc:  # noqa: BLE001
        print("CRASHED ->", type(exc).__name__)

    stub = _CPUConstantLinear(lin).eval()
    print("patched stub on mps:    ", end=" ", flush=True)
    with torch.inference_mode():
        y = stub(torch.empty(1, 0, device="mps"))
    print("OK", tuple(y.shape), "device=", y.device)
