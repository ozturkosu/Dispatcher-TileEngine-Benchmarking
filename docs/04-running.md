# 4. Running the sweep

## 4.1 Smoke test first — always

Never launch a multi-hour sweep before proving the plumbing on 10 problems.

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

Two things people get wrong:

1. **The config file is a POSITIONAL argument and it goes LAST.** It is not `--config`.
2. **`cwd` must be the CK-Tile checkout root.** The driver resolves its config paths and helper modules
   relative to the current directory.

The fp8 variant swaps three things:

```bash
  --dtype fp8 --verify-tol 0.15 \
  ... tile_engine/ops/gemm/configs/default_ci_config_gfx1250_fp8_pad.json
```

### Expected smoke output

10 problems × 16 kernels = 160 rows, minus rejected combinations. The real smoke phase produced:

| | Result |
|---|---|
| Succeeded | 96 |
| Rejected | 64 |
| bf16 verification | 96/96 |
| fp8 verification | 48/48 |

**64 rejections out of 160 is normal.** Not every tuning variant accepts every shape. Rejections are not
failures.

## 4.2 Full sweep

```bash
cd "$CK_SRC"
python3 /ck/batch_mi400_pipeline.py \
  --run-csv /ck/data/run_8611.csv \
  --out /ck/work/run8611 \
  --ck "$CK_SRC" --devices 0 --label run8611

python3 /ck/add_fail_reason.py /ck/work/run8611/master_results.csv
```

The pipeline does conversion, execution, and aggregation in one pass.

### Flags

| Flag | Effect |
|---|---|
| `--run-csv` | the hipBLASLt sheet |
| `--out` | output directory |
| `--ck` | CK-Tile checkout root (use `$CK_SRC` from §2.3; default `/ck/ck`) |
| `--devices` | comma list passed through to the driver (usually `0`) |
| `--label` | tag prefixed to progress lines |
| `--only-layouts rcr,rrr` | run a subset of layout groups — for splitting one sheet across two nodes |
| `--merge-only` | rebuild `master_results.csv` from existing `results_*.csv` without running anything |

Indices line up across nodes **only if both nodes used the identical `--run-csv`**. If one node used a
filtered or re-sorted sheet, every `problem_idx` is off and the merge is silently wrong.

### Outputs

| File | Contents |
|---|---|
| `shape_index.csv` | one row per original CSV row; the join table |
| `inputs/problems_<dtype>_<layout>.json` | bridge input, one per group |
| `results_<dtype>_<layout>.csv` | raw driver output, one row per (shape × kernel) |
| `results_<dtype>_<layout>.log` | driver stdout+stderr — read this when something fails |
| `master_results.csv` | one row per original shape, aggregated |
| `not_verified_shapes.csv` | runnable shapes with `verified=0` |
| `zero_working_shapes.csv` | runnable shapes where no kernel ran at all |
| `file.done` | written **last** — use it as the done-marker |

## 4.3 Running it in the background

```bash
docker exec -d "$CK_CONTAINER" bash -lc "
  cd $CK_SRC && python3 /ck/batch_mi400_pipeline.py \
    --run-csv /ck/data/run_8611.csv --out /ck/work/run8611 \
    --ck $CK_SRC --devices 0 --label run8611 > /ck/work/run8611.log 2>&1"

while ! docker exec "$CK_CONTAINER" test -f /ck/work/run8611/file.done; do sleep 300; done
docker exec "$CK_CONTAINER" cat /ck/work/run8611/file.done
```

## 4.4 Runtime warning: the verification reference

With `--verify` on, the fp32 CPU reference is recomputed **for each of the 16 kernels** unless the worker
caches it per shape. For large shapes the reference computation, not the GPU, dominates wall time.

Check whether your checkout already caches the reference before you patch anything.

## 4.5 Splitting a large sheet across two nodes

```bash
# node A
python3 /ck/batch_mi400_pipeline.py --run-csv /ck/data/run_8612.csv \
  --out /ck/work/run8612 --only-layouts rcr,ccr --label run8612A

# node B — SAME --run-csv
python3 /ck/batch_mi400_pipeline.py --run-csv /ck/data/run_8612.csv \
  --out /ck/work/run8612 --only-layouts crr,rrr --label run8612B

# after both finish, on either node, with all results_*.csv in one directory
python3 /ck/batch_mi400_pipeline.py --run-csv /ck/data/run_8612.csv \
  --out /ck/work/run8612 --merge-only --label run8612merge
```
