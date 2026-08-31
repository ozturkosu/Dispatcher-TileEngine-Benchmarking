#!/usr/bin/env python3
"""Add a 'likely_fail_reason' column to a master_results.csv.

For shapes that did not run (has_working_kernel==0), annotate the most likely
cause(s) based on shape/layout analysis. These are heuristic explanations of a
generic status -1 ("no CI kernel accepted the shape"), not confirmed per-shape
root causes. Working shapes get an empty reason.
"""
import csv, sys

def reason(row):
    if row.get("status") == "unsupported_dtype":
        return "bridge does not support this dtype (f32/xf32)"
    if row.get("status") != "runnable":
        return ""
    if str(row.get("has_working_kernel", "")) == "1":
        return ""  # it ran
    try:
        M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
    except (KeyError, ValueError):
        return "status -1 (no kernel accepted shape)"
    lay = row.get("layout", "")
    r = []
    if M < 8 or N < 8 or K < 8:
        small = [f"{d}={v}" for d, v in (("M", M), ("N", N), ("K", K)) if v < 8]
        r.append("degenerate dim <8 (" + ",".join(small) + "): below WMMA/tile granularity, cannot tile even with padding")
    if K % 8 != 0:
        r.append(f"K={K} not divisible by 8: bf16 WMMA needs 8-aligned contraction dim; pad_k ineffective for this layout")
    if M >= 100000 or N >= 100000:
        r.append("extreme dim >=1e5: exceeds practical grid/tiling for this layout")
    if not r and lay in ("ccr", "crr"):
        r.append(f"ColumnMajor-A layout ({lay}) with odd/large dims ({M}x{N}x{K}): no CI kernel variant accepted this combination")
    if not r:
        r.append("status -1 (no CI kernel accepted shape; edge-case dim combination)")
    return "; ".join(r)

def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("empty:", path); return
    fields = list(rows[0].keys())
    for col in ("unsupported", "likely_fail_reason"):
        if col not in fields:
            fields.append(col)
    for x in rows:
        x["unsupported"] = 1 if str(x.get("status", "")).startswith("unsupported") else 0
        x["likely_fail_reason"] = reason(x)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    nz = sum(1 for x in rows if x["likely_fail_reason"] and x.get("status") == "runnable")
    print(f"{path}: annotated, {nz} runnable shapes with a fail reason")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
