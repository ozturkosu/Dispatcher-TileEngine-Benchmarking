# 6. Troubleshooting

## Build

### Every target reports "unknown", supported-target list is empty

You omitted the compiler. CMake must be told explicitly:

```bash
cmake -DGPU_TARGETS=gfx1250 -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_PREFIX_PATH=/opt/rocm ..
```

This is the single most common setup failure.

### `git` refuses to operate on `/ck/ck`

Bind-mount UID mismatch between host and container:

```bash
git config --global --add safe.directory /ck/ck
```

### `ninja: command not found`

The image does not ship ninja. Use `make -j$(nproc)`.

### The monorepo clone never finishes

Do not clone `rocm-libraries` for this work. Shallow-clone the standalone repo:

```bash
git clone --depth 1 https://github.com/ROCm/composable_kernel.git ck
```

## Container and node

### `docker run` fails on `--group-add render`

There is no `render` group on this host. Drop the flag; `--group-add video` is sufficient.

### Results disappear when the container is removed

You wrote outside the bind mount. Everything must go under `/ck/...` (host `$CK_WS`).

### 8 cards on the host, only device 0 works in the container

Common. Verify with `/opt/rocm/bin/rocminfo`, then pass `--devices 0`. Do not assume 8-way parallelism.

### `rocminfo: command not found`

It is not on `PATH`. Use the full path `/opt/rocm/bin/rocminfo`.

### A background job seems to run forever

You used `nohup` inside `docker exec` and polled with `pgrep`. The `nohup` process dies with the SSH
session but leaves a zombie that `pgrep -f` still matches, so the loop never exits.

Use `docker exec -d` plus a **done-marker file** and poll for the file.

## Driver invocation

### The driver cannot find its config

Two causes, both common:

1. You passed the config as `--config`. It is a **positional** argument and must come **last**.
2. Your `cwd` is not the CK checkout root. `cd /ck/ck` first.

### Zero valid kernels, every shape rejected

Almost always an arch/config mismatch. gfx1250 is wave32 + WMMA; gfx942/gfx950 are wave64 + MFMA. A gfx942
config produces no valid gfx1250 kernels. Confirm you are using
`default_ci_config_gfx1250_pad.json` (bf16) or `default_ci_config_gfx1250_fp8_pad.json` (fp8).

### All fp8 shapes fail verification

Check the tolerance. fp8 needs `--verify-tol 0.15`; `0.02` rejects correct fp8 results.

### The sweep is far slower than expected

With `--verify` on, the fp32 CPU reference may be recomputed for each of the 16 kernels. For large shapes
this dominates wall time. Check whether the worker caches the reference per shape.

## Results

### `problem_idx` does not match between two result files

It is **group-local** — it restarts at 0 for each `(dtype, layout)` group. Join through `shape_index.csv`.

### Two nodes' results do not merge correctly

`--only-layouts` splitting is only valid if **both** nodes used the byte-identical `--run-csv`. A filtered
or re-sorted sheet shifts every index silently.

### The pass rate looks bad

Before escalating, split it apart:

```bash
python3 - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open("master_results.csv")))
c = collections.Counter()
for r in rows:
    if r["status"] != "runnable":          c["unsupported"] += 1
    elif r.get("verified") == "1":         c["verified"] += 1
    elif r.get("has_working_kernel") == "1": c["RUNS_BUT_INCORRECT"] += 1
    else:                                  c["never_ran"] += 1
print(dict(c))
EOF
```

Only `RUNS_BUT_INCORRECT` is a correctness defect. `never_ran` is a coverage gap; `unsupported` is out of
scope. See `docs/05-interpreting-results.md`.

### A shape has `has_working_kernel=0` — is that a wrong answer?

No. No output was produced, so nothing can be incorrect. It is a coverage gap.
