# 5. Interpreting results

This is the step people get wrong. Read it before you report anything.

## 5.1 Four distinct outcomes

There are **four** outcomes, not two. Collapsing them into pass/fail produces a wrong report.

| Outcome | How to detect it | What it means | Is it a bug? |
|---|---|---|---|
| **Verified** | `status=runnable`, `verified=1` | at least one kernel ran and matched the fp32 reference | no |
| **Runs but incorrect** | `status=runnable`, `has_working_kernel=1`, `verified=0` | produced output, failed tolerance | **YES — escalate immediately** |
| **Never ran** | `status=runnable`, `has_working_kernel=0` | all 16 variants refused the shape (status −1, `IsSupportedArguments` / launch reject) | no — a **coverage gap** |
| **Unsupported dtype** | `status != runnable` | `f32_r` / `xf32`; no kernel exists | no — out of scope |

### Never-ran is not a wrong answer

When all 16 variants refuse a shape, the kernel produced **no output at all**. There is nothing that could
be numerically incorrect. Reporting "zero-working" as a correctness failure is factually wrong, and it
inflates the apparent defect count.

Never-ran is a **coverage gap**: the bridge does not currently have a kernel variant that accepts that
shape. That is worth reporting — as a coverage gap.

### A skip is not a pass

The symmetric error. An unsupported-dtype row and a never-ran row must never be counted toward a pass rate.
State all four categories explicitly, with counts, every time.

## 5.2 Key columns in `master_results.csv`

| Column | Meaning |
|---|---|
| `status` | `runnable` / `unsupported_dtype` / `unsupported_layout` |
| `has_working_kernel` | 1 if at least one of the 16 variants produced output |
| `kernels_ok` | how many of the 16 produced output |
| `verified` | 1 if at least one produced output **and** matched the reference |
| `verified_count` | how many of the 16 verified |
| `working_kernel` | the fastest kernel from the verified pool (falls back to the working pool) |
| `verify_max_rel` | the **best** (minimum) max relative error among verified kernels |
| `best_tflops`, `best_latency_ms` | performance of `working_kernel` — **gfx1250 numbers** |
| `verified_kernels` | compact trait tags, e.g. `compv4_default_intrawave_pT` |
| `likely_fail_reason` | added by `add_fail_reason.py`; **a heuristic** |
| `unsupported` | 1 if `status` starts with `unsupported` |

## 5.3 Reference results — use these to validate a fresh setup

### llama405b sheet (id_perf_run 8611)

| | |
|---|---|
| Rows | 210 |
| Unsupported | 0 |
| Runnable | 210 |
| **Verified** | **210 / 210 (100%)** |
| Runs-but-incorrect | 0 |
| Never-ran | 0 |

If a fresh environment does not reproduce **210/210**, the environment is wrong — not the bridge. Check the
arch, the config file, and that `cwd` is the CK-Tile checkout root.

### All four sheets combined

| Category | Count |
|---|---|
| Total rows | 3,981 |
| Unsupported dtype | 1,220 |
| Runnable | 2,761 |
| **Verified** | **2,291 (83% of runnable)** |
| **Runs but incorrect** | **0** |
| Never ran | 470 |

Zero runs-but-incorrect across 2,291 verified shapes is the headline correctness result. Everything that
executed was numerically correct.

### Never-ran breakdown (470 shapes)

| Cause | Count |
|---|---|
| ColumnMajor-A with odd/large dims | 182 |
| **K not divisible by 8** | 180 |
| Degenerate dim < 8 | 75 |
| Extreme dim ≥ 1e5 | 27 |
| Unclassified | 6 |

On `K % 8 != 0`: bf16 WMMA needs an 8-aligned contraction dimension, and **`pad_k` does not rescue it**.
This is the single largest actionable coverage gap.

### Failure rate by layout

| Layout | Never-ran rate | Sample size |
|---|---|---|
| rcr | 10.4% | large |
| crr | 18.3% | large |
| ccr | 22.5% | large |
| **rrr** | **42.3%** | **only 52 shapes** |

The rrr number rests on 52 shapes. **Flag the small sample** whenever you quote it; do not present 42.3% as
an equally solid figure.

## 5.4 `likely_fail_reason` is a heuristic

`add_fail_reason.py` annotates never-ran shapes by inspecting M/N/K and the layout. It is a plausible
explanation of a generic status −1, **not a confirmed per-shape root cause**. The driver does not report
*which* constraint each variant rejected.

Always label it as a heuristic in any summary. Confirming a specific cause requires reading the driver log
or instrumenting `IsSupportedArguments`.

## 5.5 Do not compare against the sheet's `gflops` / `us`

Those columns are performance from the **collection host**, which is a different GPU generation. They tell
you where the shape came from. They are **not** a gfx1250 baseline, and a gfx1250-vs-sheet ratio is
meaningless.

For genuine bridge-vs-TileEngine performance parity you need an A/B measurement of the **same kernel** on
the **same GPU**, interleaved per shape. That is a different exercise from this sweep.
