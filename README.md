# HealDA on Apple Silicon: port, numerical trust audit, and GNSS-RO ablation

Companion repository for the manuscript
**"Running an End-to-End Machine-Learning Data Assimilation System on a
Laptop: Porting HealDA to Apple Silicon, Auditing Its Numerical
Trustworthiness, and Ablating GNSS Radio Occultation Observations"**
([`paper/main.pdf`](paper/main.pdf); method background:
[HealDA, arXiv:2601.17636](https://arxiv.org/abs/2601.17636)).

HealDA (Gupta et al. 2026) is a 327M-parameter observation-to-state
transformer that maps a 24-hour window of observations to a global 1°
analysis on the HEALPix grid in one forward pass. This repository makes
it run — trustworthy — on an Apple M5 laptop with the PyTorch MPS
backend, and uses that pipeline to run controlled observation-subset
experiments for 2024-06-01 00 UTC verified against ERA5 over East Asia.

## What's here

```
code/
  02_run_analysis.py   Run one HealDA global analysis (CPU or MPS).
                       --sensors {conv,sat,both}, --conv-vars VAR... for
                       conventional-subset (ablation) experiments,
                       --device {cpu,mps}. Applies the MPS patches
                       automatically.
  mps_patch.py         The two Metal patches (zero-input Linear fix +
                       build_input-on-CPU fix). See below.
  build_shim.py        Fetches and assembles the pure-torch earth2grid
                       shim (the official package is CUDA-12/x86_64 only).
  shim/                The vendored earth2grid shim (Apache-2.0, NVIDIA).
  06_to_latlon.py      Regrid the native HEALPix output to regular
                       lat-lon CF-1.8 NetCDF (ncview-ready).
  07_compare_multi.py  Multi-experiment verification vs ERA5.

paper/
  main.tex / main.pdf  The manuscript (compile: tectonic -X compile main.tex).
  make_figures.py      Recomputes every figure and table from the NetCDF files.
  compare_helper.py    Shared verification utilities.
  stats.csv/.json      Per-level verification statistics
                       (7 experiments x 5 variables x 13 levels).
  figures/             Publication figures (PDF).

analysis_outputs/      East-Asia 0.25° lat-lon analyses (CF-1.8 NetCDF)
                       for every experiment in the paper.
figures/               PNG previews of the key figures.
```

## The two MPS patches

1. **Zero-input `nn.Linear` crash.** HealDA's adaLN-Zero conditioning
   layers are `nn.Linear(0, 6144)` (unconditional model). On MPS a
   forward through these aborts with `MPSNDArray ... device may not be
   nil` (SIGABRT). Since `y = xW^T + b = b` for zero inputs, the layers
   are replaced by CPU-pinned constant stubs — bit-exact.

2. **Silent corruption of large `int64` copies.** `build_input()` moves
   ~11M-row observation columns to the GPU with
   `.to("mps", non_blocking=True)`; under Metal, later copies overwrite
   earlier buffers, so `target_time` (epoch seconds) arrives corrupted
   (found containing observation times in nanoseconds). The subsequent
   `target_time * 1e9` overflows int64 and the encoder is fed garbage
   time features — analyses that look plausible but verify terribly
   (meridional wind r = −0.01 vs ERA5). The fix constructs the input
   tensors on CPU and transfers synchronously. After the fix, CPU and
   MPS analyses agree to bf16 noise (z500 RMSE 3.9 m² s⁻², r = 1.0000).

**Important:** the corruption only manifests at full observation scale
(~10⁷ values); at 2×10⁴ observations encoder outputs are bit-consistent
across devices. Parity tests for accelerator porting must run at
production scale.

## Results summary (East Asia, layer-mean vs ERA5, 13 levels)

| Experiment | T RMSE (K) | u | v | z (m² s⁻²) | q (mg/kg) |
|---|---|---|---|---|---|
| CPU reference | 2.50 | 3.03 | 2.42 | 611 | 592 |
| MPS before fix | 5.00 | 8.76 | 7.48 | 1431 | 2155 |
| MPS after fix (conv) | 2.50 | 3.04 | 2.41 | 612 | 590 |
| Conv + satellite | **0.80** | **1.96** | **1.92** | **135** | **378** |

GNSS-RO ablation (layer-mean RMSE):

| Experiment | T | u | v | z | q |
|---|---|---|---|---|---|
| Conv (all, 71% RO by count) | 2.50 | 3.04 | 2.41 | 612 | 590 |
| Conv w/o RO | 2.78 | 3.20 | 2.55 | 697 | 695 |
| RO 3-channel only | 3.95 | 5.03 | 5.40 | 1144 | 1119 |
| RO bending-angle only | 5.56 | 9.90 | 6.37 | 1528 | 1950 |

Notable: RO-only matches the full conventional stream in the upper
stratosphere (150 hPa T: r = 0.979 vs 0.974) but degrades below
500 hPa; removing the `gps_t`/`gps_q` *background-value* channels
collapses RO-only meridional-wind skill from r = 0.44 to r = 0.05 —
the network learns a joint (observation, background) → analysis map,
not a retrieval.

## Reproducing

```bash
# environment
uv sync          # python 3.12, torch 2.10, earth2studio 0.18.0

# single analyses (see paper Appendix A for the full set)
python code/02_run_analysis.py --date 2024-06-01T00:00:00 \
    --sensors both --device mps --out convsat.nc

# verification + figures (needs ERA5 pressure-level extract)
python paper/make_figures.py
```

The HealDA checkpoint is fetched automatically by `earth2studio`.
ERA5 data: Copernicus Climate Data Store.

## License

- `code/`: MIT (vendored `earth2grid` shim remains Apache-2.0, NVIDIA).
- `paper/` and `analysis_outputs/`: CC BY 4.0.

## Citation

If you use this work, please cite HealDA:

> Gupta, A., et al., 2026: HealDA: Highlighting the importance of
> initial errors in end-to-end AI weather forecasts.
> arXiv:2601.17636.
