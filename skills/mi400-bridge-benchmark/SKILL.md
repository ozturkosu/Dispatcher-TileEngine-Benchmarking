---
name: mi400-bridge-benchmark
description: Run CK-Tile TileEngine→Dispatcher bridge correctness/perf sweeps on MI400 (gfx1250) using real hipBLASLt production shape sheets (llama405b / wrt / rr). Handles node access, container setup, locating the prebuilt CK-Tile (build only as a fallback), shape-sheet→bridge-input conversion, smoke test, full sweep, and result interpretation. TRIGGER when the user asks to benchmark or validate the CK-Tile bridge on MI400/gfx1250, replay production GEMM shapes, run a default_config sweep, or convert a hipBLASLt run CSV into bridge input. SKIP for gfx942/gfx950 work (different warp tiles and configs) or for non-GEMM operators.
---
# MI400 (gfx1250) CK-Tile bridge benchmarking
You are running correctness + performance sweeps of the CK-Tile **TileEngine→Dispatcher bridge** on MI400 /
gfx1250, driven by **real production GEMM shapes** captured by hipBLASLt.
Full human guide (keep in sync with this skill): `<INTERNAL_WIKI_LINK>`
## Scope — state this to the user before starting
- Operator: **`gemm_universal` only** (plain `D = A × B`). NOT grouped / multi_d / multi_abd / preshuffle /
batched / batched_contraction / stream-K / MX / any quantised bridge. Those have separate drivers.
- Dtypes the bridge supports: **fp16, bf16, fp8, bf8**. `f32_r` / `xf32` rows are `unsupported_dtype`.
- gfx1250 is **wave32 + RDNA-style WMMA**, unlike gfx942/gfx950 (wave64 + MFMA). Warp tiles are
**16×16×32 bf16**, **16×16×64 fp8**. gfx942 configs will produce zero valid kernels — never reuse them.
- "16 kernels" = **16 tuning variants of the same operator** (pipeline compv3/compv4/mem × scheduler
intrawave/interwave × epilogue cshuffle/default × persistent T/F), not 16 different ops.
## Step 1 — Access and container
Node is a Conductor SUT and needs a live reservation + registered SSH key. The login banner is large ASCII
noise; ignore it. Work under `$HOME` (`/data/work` is not writable).
```bash
ssh <USER>@<GFX1250_NODE>
ls /dev/kfd /dev/dri/card*         # expect card0..card7
```
```bash
export CK_CONTAINER="ck-$USER"
export CK_WS="$HOME/ck_ws"
export CK_IMAGE="<CK_IMAGE_TAG>"   # confirm the current tag with your team; tags rot
mkdir -p "$CK_WS"
docker run -d --name "$CK_CONTAINER" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16G \
  --network host \
  -v "$CK_WS":/ck -w /ck \
  "$CK_IMAGE" sleep infinity
docker exec -it "$CK_CONTAINER" bash
```
Inside: `/opt/rocm/bin/rocminfo | grep -i gfx1250` (rocminfo is **not** on PATH).
Hard-won environment facts:
- **No `render` group on this host** — `--group-add render` makes `docker run` fail. `video` is enough.
- Host `$CK_WS` is bind-mounted to `/ck`. **Always write results under `/ck/...`** so they survive the
container.
- Often **only device 0 is usable** inside the container even with 8 cards. Check, then use `--devices 0`.
## Step 2 — Locate the prebuilt CK-Tile (do NOT build)
This is **CK-Tile / TileEngine**: the driver compiles the kernel variants it needs on demand, so there is no
CK-Tile library to build. The `ck-wmma-instances` image already ships the CK-Tile checkout with the gfx1250
WMMA instances in place. Locate it and point everything at it via `$CK_SRC`.
```bash
export CK_SRC=$(find / -name gemm_full_benchmark.py 2>/dev/null | head -1 \
  | sed 's:/tile_engine/ops/gemm/gemm_full_benchmark.py::')
echo "CK_SRC=$CK_SRC"                                   # e.g. /composable_kernel
ls "$CK_SRC"/tile_engine/ops/gemm/configs/ | grep 1250  # confirm gfx1250 configs exist
```
Use `cd "$CK_SRC"` and `--ck "$CK_SRC"` in every later step. Keep your own scripts / data / results under
the bind-mounted `/ck`; only the CK-Tile **source** lives in the image.

**Fallback ONLY if `$CK_SRC` came back empty** (image has no CK-Tile source). Shallow-clone the standalone
repo (~261 MB, ~20 s); the driver still builds kernels itself at run time, so you do **not** run a full
`make`. The `rocm-libraries` monorepo blobless clone is painfully slow; avoid it.
```bash
cd /ck && git clone --depth 1 https://github.com/ROCm/composable_kernel.git ck
git config --global --add safe.directory /ck/ck        # bind-mount UID mismatch
export CK_SRC=/ck/ck
/opt/rocm/bin/hipcc --version                          # the driver needs a working hipcc
ls "$CK_SRC"/tile_engine/ops/gemm/configs/ | grep 1250 # and the gfx1250 configs
```
For long jobs use `docker exec -d` plus a done-marker file, never `pgrep` — a `nohup` inside `docker exec`
dies when the SSH session ends and leaves a zombie that `pgrep -f git.clone` still matches, so a polling
loop waits forever.
## Step 3 — Convert the shape sheet into bridge input
Source sheets are hipBLASLt production perf runs (`ml_collection_*`), one CSV per test set:
| id_perf_run | test_set | note |
|---|---|---|
| 8611 | **llama405b** | the "Meta shapes"; 210 rows, all `T/N` → rcr, 0 unsupported. Best first sheet. |
| 8612 | wrt1-15 | 2,199 rows |
| 8613 | wrt16-39 | 827 rows |
| 8614 | rr | 745 rows |
Two mapping rules — apply exactly:
```python
LAYOUT_MAP = {("T","N"):"rcr", ("N","N"):"ccr", ("N","T"):"crr", ("T","T"):"rrr"}
def dtype_of(a_type):
    if a_type == "bf16_r": return "bf16"
    if a_type == "f8_r": return "fp8"
    return None  # f32_r / xf32 -> unsupported_dtype
```
Group rows by `(dtype, layout)`, give each a **group-local** `problem_idx`, and emit one input file per
group. That file is the **entire** bridge input format:
```json
{"problems": [{"M": 16032, "N": 9, "K": 16384},
              {"M": 16032, "N": 17, "K": 16384}]}
```
Layout and dtype go on the command line, not in the JSON. `lda/ldb/strides/alpha/beta` are **not** passed —
the bridge derives its own from M/N/K + layout.
Also write `shape_index.csv` (one row per original CSV row, with `id_csv_entry`, mapped dtype/layout,
`status`, `problem_idx`). **Without it a `problem_idx` is meaningless**, because indices restart per group.
Verify tolerances: **bf16 `0.02`, fp8 `0.15`** (fp8 genuinely carries less precision; 0.02 would reject
correct results). The converter is `scripts/sheet_to_bridge_input.py` and runs GPU-free.
## Step 4 — Smoke test before any long run
Never launch a multi-hour sweep before proving the plumbing. Take 10 problems and run:
```bash
cd "$CK_SRC"
python3 tile_engine/ops/gemm/gemm_full_benchmark.py \
  --variant gemm_universal --arch gfx1250 \
  --dtype bf16 --layout rcr \
  --problems /ck/work/smoke_bf16_rcr.json \
  --devices 0 --workers 16 --kernel-timeout 900 \
  --verify --verify-tol 0.02 \
  --csv /ck/work/smoke_bf16_rcr.csv \
  tile_engine/ops/gemm/configs/default_ci_config_gfx1250_pad.json
```
- The **config file is POSITIONAL and goes LAST** — it is not `--config`.
- `cwd` must be the CK-Tile checkout root; the driver resolves its config and helpers relative to it.
- fp8 swaps in `--dtype fp8 --verify-tol 0.15` and `configs/default_ci_config_gfx1250_fp8_pad.json`.
Expect ~10 × 16 = 160 rows, minus rejected combinations. Reference from the real smoke phase: **96
succeeded / 64 rejected**, verification 96/96 bf16 and 48/48 fp8. **64 rejections out of 160 is normal** —
not every tuning variant accepts every shape.
## Step 5 — Full sweep
```bash
cd "$CK_SRC"
python3 /ck/batch_mi400_pipeline.py \
  --run-csv /ck/data/run_8611.csv \
  --out /ck/work/run8611 \
  --ck "$CK_SRC" --devices 0 --label run8611
python3 /ck/add_fail_reason.py /ck/work/run8611/master_results.csv
```
Flags: `--only-layouts rcr,rrr` runs a subset (for splitting one sheet across two nodes);
`--merge-only` rebuilds `master_results.csv` from existing `results_*.csv` without running anything.
Indices line up across nodes **provided both used the identical `--run-csv`**.
Outputs: `shape_index.csv`, `inputs/problems_*.json`, `results_<dtype>_<layout>.csv` + `.log`,
`master_results.csv`, `not_verified_shapes.csv`, `zero_working_shapes.csv`, and `file.done` (written last —
use it as the done-marker).
Performance note: with `--verify` on, the fp32 reference is recomputed **for every one of the 16 kernels**
unless the worker caches it per shape. For large shapes the reference dominates runtime. Check whether the
checkout already caches before patching.
## Step 6 — Interpret results honestly
**This is the step people get wrong.** There are four distinct outcomes:
| Outcome | Detect | Meaning | Bug? |
|---|---|---|---|
| **Verified** | `status=runnable`, `verified=1` | ran and numerically correct | no |
| **Runs but incorrect** | `runnable`, `has_working_kernel=1`, `verified=0` | produced output, failed tolerance | **YES — escalate** |
| **Never ran** | `runnable`, `has_working_kernel=0` | all 16 refused the shape (status −1, `IsSupportedArguments`/launch reject). **No output was produced, so nothing can be wrong.** | no — coverage gap |
| **Unsupported dtype** | `status != runnable` | `f32_r`/`xf32`, no kernel exists | no — out of scope |
Never report "zero-working" as a correctness failure. Never report a **skip** as a pass.
Reference result for the llama405b sheet — use it to validate a fresh setup: **210/210 verified (100%),
0 runs-but-wrong, 0 never-ran.** If you do not get 210/210, your environment is wrong, not the bridge.
Across all four sheets: 3,981 rows → 1,220 unsupported dtype, 2,761 runnable, 2,291 verified (83%),
**0 runs-but-incorrect**, 470 never-ran. Never-ran causes: ColumnMajor-A odd/large dims 182; **K not
divisible by 8** 180 (WMMA needs 8-aligned K and `pad_k` does *not* rescue it); degenerate dim <8 75;
extreme dim ≥1e5 27; unclassified 6. Failure rate by layout: rcr 10.4%, crr 18.3%, ccr 22.5%, rrr 42.3%
(rrr on only 52 shapes — treat that percentage with caution).
## Reporting rules — non-negotiable
- Distinguish **PASS / FAIL / never-ran / unsupported** explicitly in every summary. A skip is not a pass.
- State the **arch you actually ran on**. Never present a gfx1250 result as gfx942/gfx950 evidence, or
vice versa.
- `likely_fail_reason` from `add_fail_reason.py` is a **heuristic**, not a confirmed root cause. Say so.
- The hipBLASLt `gflops`/`us` columns in the source sheet are from the collection host (a different GPU) —
they are provenance, not a gfx1250 baseline. Do not compare against them.
- Report real compiler/runtime errors verbatim; do not summarise a genuine build failure as "environmental".
- If a run is incomplete, say what is missing rather than extrapolating.
