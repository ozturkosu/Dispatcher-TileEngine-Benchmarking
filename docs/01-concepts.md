# 1. Concepts

## What is the "bridge"?

CK-Tile has two ways to get from a kernel description to a runnable GEMM:

- **TileEngine (TE)** — the older path. A codegen front end that emits and builds one benchmark
  executable per kernel instance.
- **Dispatcher** — the newer path. A generic kernel-selection and launch layer.

The **TileEngine→Dispatcher bridge** lets a TileEngine-style operator description be executed through the
Dispatcher instead of through TE's own build-and-run machinery. Benchmarking the bridge means answering
two questions:

1. **Correctness** — does the bridge produce numerically correct output for real production shapes?
2. **Performance** — is it at parity with the old TE path on the same kernel?

This repository covers the **correctness/coverage** sweep on gfx1250, driven by real GEMM shapes.

## Scope

| Dimension | Covered |
|---|---|
| Operator | `gemm_universal` only (plain `D = A × B`) |
| Dtypes | fp16, bf16, fp8, bf8 |
| Not covered | grouped, multi_d, multi_abd, preshuffle, batched, batched_contraction, stream-K, MX, any quantised bridge |
| Not covered | `f32_r` / `xf32` shapes — classified `unsupported_dtype` |

Every excluded operator has its own driver and its own default config. Do not point this pipeline at them.

## gfx1250 is not gfx942/gfx950

| | gfx942 / gfx950 | gfx1250 |
|---|---|---|
| Wave size | 64 | **32** |
| Matrix core | MFMA | **RDNA-style WMMA** |
| bf16 warp tile | (arch-specific MFMA tiles) | **16×16×32** |
| fp8 warp tile | (arch-specific MFMA tiles) | **16×16×64** |

The configs are **not interchangeable**. Feeding a gfx942 default config to gfx1250 produces **zero valid
kernels**, and the run looks like a total failure when it is really a config mismatch.

## What "16 kernels" means

Each shape is run against **16 tuning variants of the same operator**, not 16 different operators. The
variants are the cross product of:

- pipeline: `compv3` / `compv4` / `mem`
- scheduler: `intrawave` / `interwave`
- epilogue: `cshuffle` / `default`
- persistent: `True` / `False`

Not every variant accepts every shape. A variant that refuses a shape returns status −1 and produces no
output. **A shape counts as verified if at least one variant ran and matched the fp32 reference.**

## Shape provenance

The shapes come from hipBLASLt production perf runs (`ml_collection_*` sheets) — real GEMM calls from real
model workloads, not synthetic sizes. Four sheets are used:

| id_perf_run | test_set | rows |
|---|---|---|
| 8611 | llama405b ("Meta shapes") | 210 |
| 8612 | wrt1-15 | 2,199 |
| 8613 | wrt16-39 | 827 |
| 8614 | rr | 745 |

Each sheet also carries `gflops` and `us` columns. **Those numbers came from the collection host, which is
a different GPU.** They are provenance metadata, not a gfx1250 baseline. Never compare a gfx1250 result
against them.
