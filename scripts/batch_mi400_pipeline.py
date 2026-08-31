#!/usr/bin/env python3
"""Per-file MI400 gfx1250 gemm_universal bridge VERIFY-ALL pipeline.

For one hipBLASLt run CSV:
  - map transA/transB -> bridge layout (rcr/ccr/crr/rrr)
  - classify dtype: bf16_r->bf16, f8_r->fp8 (runnable); f32_r/xf32->unsupported_dtype
  - group runnable shapes by (dtype, layout)
  - for EVERY shape, run ALL 16 CI kernels with --verify ON (fp32 reference,
    cached once per shape in the worker) -> results_<dtype>_<layout>.csv
  - master_results.csv: per shape, which kernels verify + which work + perf
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

LAYOUT_MAP = {("T", "N"): "rcr", ("N", "N"): "ccr", ("N", "T"): "crr", ("T", "T"): "rrr"}

def dtype_of(a_type):
    if a_type == "bf16_r":
        return "bf16"
    if a_type == "f8_r":
        return "fp8"
    return None  # f32_r / xf32 etc -> unsupported

def run_driver(ck, args, logpath):
    cmd = [sys.executable, "tile_engine/ops/gemm/gemm_full_benchmark.py"] + args
    with open(logpath, "w") as lf:
        p = subprocess.run(cmd, cwd=ck, stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode

def read_csv_by_idx(path):
    """Return dict: problem_idx -> list of rows(dict)."""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        try:
            pi = int(r["problem_idx"])
        except (KeyError, ValueError):
            continue
        out.setdefault(pi, []).append(r)
    return out

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def shorttag(kernel):
    """compact trait tag: pipeline_epilogue_scheduler_p<persistent> from kernel name."""
    try:
        parts = kernel.split("_")
        # gemm <dtype> <layout> <pipeline> <epilogue> <scheduler> <padM padN padK persistent> ...
        pipe, epi, sched = parts[3], parts[4], parts[5]
        persistent = parts[9]  # 4th bool after scheduler
        return f"{pipe}_{epi}_{sched}_p{persistent[0]}"
    except Exception:
        return kernel[:24]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ck", default="/ck/ck")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--label", default="")
    ap.add_argument("--only-layouts", default="", help="comma list; only RUN these layout groups (others skipped, for cross-node split)")
    ap.add_argument("--merge-only", action="store_true", help="do not run kernels; build master from existing results_*.csv (merge step)")
    a = ap.parse_args()

    ck = a.ck
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "bf16": "tile_engine/ops/gemm/configs/default_ci_config_gfx1250_pad.json",
        "fp8": "tile_engine/ops/gemm/configs/default_ci_config_gfx1250_fp8_pad.json",
    }
    tol = {"bf16": "0.02", "fp8": "0.15"}

    rows = list(csv.DictReader(open(a.run_csv)))
    groups = {}   # (dtype,layout) -> list of shape dicts (with local idx)
    index = []
    for r in rows:
        M, N, K = int(r["m"]), int(r["n"]), int(r["k"])
        lay = LAYOUT_MAP.get((r["transA"], r["transB"]))
        dt = dtype_of(r["a_type"])
        rec = {
            "id_csv_entry": r.get("id_csv_entry", ""), "id_run": r.get("id_run", ""),
            "M": M, "N": N, "K": K, "transA": r["transA"], "transB": r["transB"],
            "a_type": r["a_type"], "b_type": r["b_type"], "c_type": r["c_type"], "d_type": r["d_type"],
            "compute_type": r.get("compute_type", ""),
        }
        if dt is None:
            rec.update(dtype=r["a_type"], layout=lay or "?", status="unsupported_dtype",
                       shape_name=f"{r['a_type']}_{M}x{N}x{K}", problem_idx=-1)
            index.append(rec); continue
        if lay is None:
            rec.update(dtype=dt, layout="?", status="unsupported_layout",
                       shape_name=f"{dt}_{M}x{N}x{K}", problem_idx=-1)
            index.append(rec); continue
        key = (dt, lay)
        g = groups.setdefault(key, [])
        pidx = len(g)
        rec.update(dtype=dt, layout=lay, status="runnable",
                   shape_name=f"{dt}_{lay}_{M}x{N}x{K}", problem_idx=pidx)
        g.append(rec); index.append(rec)

    cols = ["shape_name", "id_csv_entry", "id_run", "dtype", "layout", "M", "N", "K",
            "transA", "transB", "a_type", "b_type", "c_type", "d_type", "compute_type",
            "status", "problem_idx"]
    with open(out / "shape_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(index)

    n_run = sum(len(v) for v in groups.values())
    n_unsup = sum(1 for x in index if x["status"] != "runnable")
    print(f"[{a.label}] total={len(rows)} runnable={n_run} unsupported={n_unsup} groups={sorted(groups)}", flush=True)

    # ---- combined VERIFY + PERF: all 16 kernels x all shapes, --verify ON ----
    inputs = out / "inputs"; inputs.mkdir(exist_ok=True)
    only = set(x.strip() for x in a.only_layouts.split(",") if x.strip())
    result_paths = {}
    for (dt, lay), g in sorted(groups.items()):
        res_csv = out / f"results_{dt}_{lay}.csv"
        result_paths[(dt, lay)] = res_csv  # master reads this whether or not we run it here
        if a.merge_only:
            continue
        if only and lay not in only:
            print(f"[{a.label}] SKIP {dt}/{lay} (not in --only-layouts)", flush=True)
            continue
        probs = [{"M": r["M"], "N": r["N"], "K": r["K"]} for r in g]
        pj = inputs / f"problems_{dt}_{lay}.json"
        json.dump({"problems": probs}, open(pj, "w"))
        print(f"[{a.label}] VERIFY+PERF {dt}/{lay} : {len(probs)} shapes x 16 kernels", flush=True)
        run_driver(ck, ["--variant", "gemm_universal", "--arch", "gfx1250",
                        "--dtype", dt, "--layout", lay, "--problems", str(pj),
                        "--devices", a.devices, "--workers", "16", "--kernel-timeout", "900",
                        "--verify", "--verify-tol", tol[dt],
                        "--csv", str(res_csv), cfg[dt]],
                   out / f"results_{dt}_{lay}.log")

    # ---- master: per shape, which kernels verify / work + perf ----
    master_cols = ["shape_name", "id_csv_entry", "id_run", "dtype", "layout", "M", "N", "K",
                   "transA", "transB", "compute_type", "status",
                   "has_working_kernel", "kernels_ok", "verified", "verified_count",
                   "working_kernel", "verify_max_rel", "best_tflops", "best_latency_ms",
                   "verified_kernels"]
    res_cache = {k: read_csv_by_idx(str(v)) for k, v in result_paths.items()}
    master = []
    for rec in index:
        row = {c: rec.get(c, "") for c in master_cols}
        if rec["status"] == "runnable":
            key = (rec["dtype"], rec["layout"])
            runs = res_cache.get(key, {}).get(rec["problem_idx"], [])
            oks = [r for r in runs if (fnum(r.get("tflops")) or 0) > 0]
            vok = [r for r in oks if str(r.get("verified", "")).lower() == "true"]
            row["kernels_ok"] = len(oks)
            row["has_working_kernel"] = 1 if oks else 0
            row["verified_count"] = len(vok)
            row["verified"] = 1 if vok else 0
            pool = vok if vok else oks
            if pool:
                best = max(pool, key=lambda r: fnum(r["tflops"]) or 0)
                row["working_kernel"] = best["kernel"]
                row["best_tflops"] = round(fnum(best["tflops"]) or 0, 3)
                row["best_latency_ms"] = round(fnum(best["latency_ms"]) or 0, 6)
            if vok:
                rels = [fnum(r.get("max_rel")) for r in vok if fnum(r.get("max_rel")) is not None]
                if rels:
                    row["verify_max_rel"] = min(rels)
                row["verified_kernels"] = ";".join(sorted({shorttag(r["kernel"]) for r in vok}))
        master.append(row)
    with open(out / "master_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=master_cols, extrasaction="ignore"); w.writeheader(); w.writerows(master)

    # traces: shapes that never verified, and shapes with no working kernel at all
    not_verified = [r for r in master if r["status"] == "runnable" and not r.get("verified")]
    zero_working = [r for r in master if r["status"] == "runnable" and not r.get("has_working_kernel")]
    with open(out / "not_verified_shapes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=master_cols, extrasaction="ignore"); w.writeheader(); w.writerows(not_verified)
    with open(out / "zero_working_shapes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=master_cols, extrasaction="ignore"); w.writeheader(); w.writerows(zero_working)

    summ = {}
    for rec in master:
        summ[rec["status"]] = summ.get(rec["status"], 0) + 1
    nverif = sum(1 for r in master if r["status"] == "runnable" and r.get("verified"))
    nwork = sum(1 for r in master if r["status"] == "runnable" and r.get("has_working_kernel"))
    print(f"[{a.label}] DONE status={summ} shapes_working={nwork} shapes_verified={nverif} "
          f"not_verified={len(not_verified)} zero_working={len(zero_working)}", flush=True)
    (out / "file.done").write_text(
        f"status={summ} working={nwork} verified={nverif} not_verified={len(not_verified)} zero_working={len(zero_working)}\n")

if __name__ == "__main__":
    main()
