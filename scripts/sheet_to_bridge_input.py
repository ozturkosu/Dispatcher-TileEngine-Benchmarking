#!/usr/bin/env python3
"""Convert a hipBLASLt production run CSV into CK bridge input files.

This is the GPU-free half of ``batch_mi400_pipeline.py``. It performs only the
conversion step, so you can prepare (and review) bridge inputs on a laptop and
ship them to the GPU node afterwards.

For every row of the run CSV it:
  * maps ``(transA, transB)`` to a bridge layout (rcr / ccr / crr / rrr)
  * maps ``a_type`` to a bridge dtype (bf16_r -> bf16, f8_r -> fp8);
    anything else (f32_r, xf32, ...) is marked ``unsupported_dtype``
  * groups the runnable shapes by ``(dtype, layout)`` and assigns each a
    GROUP-LOCAL ``problem_idx``

Outputs, under ``--out``:
  inputs/problems_<dtype>_<layout>.json   {"problems": [{"M":..,"N":..,"K":..}, ...]}
  shape_index.csv                         one row per ORIGINAL csv row

The ``problem_idx`` restarts at 0 for every (dtype, layout) group, so it is
meaningless on its own -- always keep ``shape_index.csv`` next to the results.

Layout and dtype are passed to the benchmark driver on the COMMAND LINE, not in
the JSON. Strides / lda / ldb / alpha / beta are deliberately NOT emitted: the
bridge derives its own from M/N/K plus the layout.

Usage:
  python3 sheet_to_bridge_input.py --run-csv run_8611.csv --out ./run8611
  python3 sheet_to_bridge_input.py --run-csv run_8611.csv --out ./run8611 --dry-run
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# (transA, transB) -> bridge layout
LAYOUT_MAP = {("T", "N"): "rcr", ("N", "N"): "ccr", ("N", "T"): "crr", ("T", "T"): "rrr"}

INDEX_COLS = [
    "shape_name", "id_csv_entry", "id_run", "dtype", "layout", "M", "N", "K",
    "transA", "transB", "a_type", "b_type", "c_type", "d_type", "compute_type",
    "status", "problem_idx",
]


def dtype_of(a_type):
    """Map the hipBLASLt A-operand type to a bridge dtype, or None if unsupported."""
    if a_type == "bf16_r":
        return "bf16"
    if a_type == "f8_r":
        return "fp8"
    return None  # f32_r / xf32 / ... -> unsupported_dtype


def classify(rows):
    """Return (groups, index).

    groups: {(dtype, layout): [shape record, ...]} in group-local order
    index:  [record, ...] one per input row, in original order
    """
    groups = {}
    index = []
    for r in rows:
        M, N, K = int(r["m"]), int(r["n"]), int(r["k"])
        layout = LAYOUT_MAP.get((r["transA"], r["transB"]))
        dtype = dtype_of(r["a_type"])
        rec = {
            "id_csv_entry": r.get("id_csv_entry", ""), "id_run": r.get("id_run", ""),
            "M": M, "N": N, "K": K, "transA": r["transA"], "transB": r["transB"],
            "a_type": r["a_type"], "b_type": r.get("b_type", ""),
            "c_type": r.get("c_type", ""), "d_type": r.get("d_type", ""),
            "compute_type": r.get("compute_type", ""),
        }
        if dtype is None:
            rec.update(dtype=r["a_type"], layout=layout or "?", status="unsupported_dtype",
                       shape_name=f"{r['a_type']}_{M}x{N}x{K}", problem_idx=-1)
        elif layout is None:
            rec.update(dtype=dtype, layout="?", status="unsupported_layout",
                       shape_name=f"{dtype}_{M}x{N}x{K}", problem_idx=-1)
        else:
            g = groups.setdefault((dtype, layout), [])
            rec.update(dtype=dtype, layout=layout, status="runnable",
                       shape_name=f"{dtype}_{layout}_{M}x{N}x{K}", problem_idx=len(g))
            g.append(rec)
        index.append(rec)
    return groups, index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-csv", required=True, help="hipBLASLt run CSV (e.g. run_8611.csv)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the group summary only; write nothing")
    a = ap.parse_args(argv)

    with open(a.run_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    groups, index = classify(rows)

    n_run = sum(len(v) for v in groups.values())
    n_unsup = len(index) - n_run
    print(f"total={len(index)} runnable={n_run} unsupported={n_unsup}")
    for (dtype, layout), g in sorted(groups.items()):
        print(f"  {dtype:5s} {layout:4s} : {len(g)} problems")
    if a.dry_run:
        print("(dry run - nothing written)")
        return 0

    out = Path(a.out)
    inputs = out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for (dtype, layout), g in sorted(groups.items()):
        problems = [{"M": r["M"], "N": r["N"], "K": r["K"]} for r in g]
        path = inputs / f"problems_{dtype}_{layout}.json"
        with open(path, "w") as f:
            json.dump({"problems": problems}, f)
        print(f"wrote {path} ({len(problems)} problems)")

    idx_path = out / "shape_index.csv"
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(index)
    print(f"wrote {idx_path} ({len(index)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
