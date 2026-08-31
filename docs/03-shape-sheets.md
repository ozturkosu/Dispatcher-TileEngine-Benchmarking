# 3. Shape sheets and bridge input

## 3.1 Source format

A hipBLASLt run CSV has one row per production GEMM call. The columns this pipeline reads are:

| Column | Use |
|---|---|
| `id_csv_entry` | stable identity of the row; carried into `shape_index.csv` |
| `id_run` | which sheet (8611 / 8612 / 8613 / 8614) |
| `m`, `n`, `k` | the problem size |
| `transA`, `transB` | mapped to a bridge layout |
| `a_type` | mapped to a bridge dtype |
| `b_type`, `c_type`, `d_type`, `compute_type` | recorded for provenance only |
| `gflops`, `us` | **provenance from the collection host — a different GPU. Not a baseline.** |

Everything else (`lda`, `ldb`, `stride_*`, `alpha`, `beta`, `solution_name`, ...) is ignored.

## 3.2 The two mapping rules

```python
LAYOUT_MAP = {("T","N"):"rcr", ("N","N"):"ccr", ("N","T"):"crr", ("T","T"):"rrr"}

def dtype_of(a_type):
    if a_type == "bf16_r": return "bf16"
    if a_type == "f8_r":   return "fp8"
    return None            # f32_r / xf32 -> unsupported_dtype
```

Apply them exactly. `f32_r` and `xf32` rows are not failures — the bridge has no kernel for them, and they
are reported as `unsupported_dtype`.

## 3.3 Bridge input format

One JSON file per `(dtype, layout)` group. The **entire** format is:

```json
{"problems": [{"M": 16032, "N": 9,  "K": 16384},
              {"M": 16032, "N": 17, "K": 16384}]}
```

- dtype and layout go on the **command line**, not in the JSON.
- `lda` / `ldb` / strides / `alpha` / `beta` are **not** passed. The bridge derives its own from
  M/N/K plus the layout.

## 3.4 `problem_idx` is group-local

Indices restart at 0 for **every** `(dtype, layout)` group. `problem_idx=3` in `results_bf16_rcr.csv` and
`problem_idx=3` in `results_fp8_rcr.csv` are different shapes.

`shape_index.csv` is the join table that makes an index meaningful. It has one row per **original CSV row**
(including unsupported ones) with:

`shape_name, id_csv_entry, id_run, dtype, layout, M, N, K, transA, transB, a_type, b_type, c_type, d_type, compute_type, status, problem_idx`

`status` is one of `runnable`, `unsupported_dtype`, `unsupported_layout`. Unsupported rows carry
`problem_idx = -1`.

**Never ship results without `shape_index.csv`.**

## 3.5 Doing the conversion (no GPU needed)

```bash
python3 scripts/sheet_to_bridge_input.py \
  --run-csv run_8611.csv \
  --out ./run8611
```

Output:

```
total=210 runnable=210 unsupported=0
  bf16  rcr  : 118 problems
  fp8   rcr  : 92 problems
wrote run8611/inputs/problems_bf16_rcr.json (118 problems)
wrote run8611/inputs/problems_fp8_rcr.json (92 problems)
wrote run8611/shape_index.csv (210 rows)
```

Use `--dry-run` to print the group summary without writing anything — useful for sizing a sweep before you
commit a node to it.

The full pipeline (`batch_mi400_pipeline.py`) performs the same conversion internally, so you do not need
to run this tool first. It exists so you can inspect and review the inputs off-GPU, and so you can reuse
the conversion logic for other sheets.

## 3.6 Verify tolerances

| dtype | `--verify-tol` |
|---|---|
| bf16 | `0.02` |
| fp8 | `0.15` |

fp8 genuinely carries less precision. Using `0.02` for fp8 rejects results that are correct.
