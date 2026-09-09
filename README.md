# Dispatcher-TileEngine-Benchmarking

Teach an AI agent to run **CK-Tile TileEngine→Dispatcher bridge** correctness and coverage
sweeps on AMD GPUs — specifically MI400 / **gfx1250** — driven by real production GEMM shapes captured from
hipBLASLt.

The deliverable is an **agent skill** (`skills/mi400-bridge-benchmark/SKILL.md`) plus the scripts, docs and
example data needed to make it work. Start with **[AGENTS.md](AGENTS.md)**.

## What is "the bridge"?

CK-Tile has two paths from a kernel description to a runnable GEMM: **TileEngine (TE)**, which codegens and
builds one benchmark executable per kernel instance, and the newer **Dispatcher**, a generic
kernel-selection and launch layer. The **TileEngine→Dispatcher bridge** lets a TileEngine-style operator
description execute through the Dispatcher instead of TE's own build-and-run machinery. Benchmarking it
means proving two things: that it is numerically correct on real shapes, and that it is at performance
parity with the old path. This repository covers the **correctness/coverage** half on gfx1250.

## Quick start

```bash
git clone <this repo> && cd Dispatcher-TileEngine-Benchmarking

# 1. install the agent skill
mkdir -p ~/.claude/skills/mi400-bridge-benchmark
cp skills/mi400-bridge-benchmark/SKILL.md ~/.claude/skills/mi400-bridge-benchmark/SKILL.md

# 2. fill in your environment
cp config.example.env config.env && $EDITOR config.env

# 3. try the GPU-free conversion step on any hipBLASLt run CSV
python3 scripts/sheet_to_bridge_input.py --run-csv run_8611.csv --out ./run8611 --dry-run
```

Then read [AGENTS.md](AGENTS.md) for example prompts, and `docs/` for the human walkthrough.

## Repository layout

```
README.md                       this file
AGENTS.md                       how to teach your agent — start here
LICENSE                         MIT
config.example.env              placeholders you must fill in
skills/
  mi400-bridge-benchmark/
    SKILL.md                    the agent skill
scripts/
  sheet_to_bridge_input.py      hipBLASLt run CSV -> bridge input JSON + shape_index.csv (no GPU needed)
  batch_mi400_pipeline.py       full sweep: convert, run all 16 kernels with --verify, aggregate
  add_fail_reason.py            heuristic annotation of never-ran shapes
docs/
  01-concepts.md                what the bridge is; gfx1250 vs gfx942/gfx950; what "16 kernels" means
  02-setup.md                   node, container, locate CK-Tile, long-running jobs
  03-shape-sheets.md            layout/dtype mapping, bridge input format, group-local problem_idx
  04-running.md                 smoke test, full sweep, flags, outputs, cross-node split
  05-interpreting-results.md    the four outcomes, reference numbers, honest reporting
  06-troubleshooting.md         symptom -> cause -> fix
```

## Sanitization — you must fill in placeholders

This repository contains **no infrastructure identifiers**. Hostnames, usernames, container image tags and
internal wiki links are replaced with placeholders:

`<GFX1250_NODE>` `<GPU_NODE>` `<USER>` `<CK_IMAGE_TAG>` `<INTERNAL_WIKI_LINK>` `<INTERNAL_CI_HOST>`

See `config.example.env`. Nothing will run until you supply real values. **Do not commit your filled-in
`config.env`** — it is git-ignored.

## Scope limits

| | |
|---|---|
| Operator | `gemm_universal` only |
| Dtypes | fp16, bf16, fp8, bf8. `f32_r`/`xf32` are `unsupported_dtype` |
| Arch | gfx1250 (wave32 + WMMA). gfx942/gfx950 configs are **not** interchangeable |
| Not covered | grouped, multi_d, multi_abd, preshuffle, batched, batched_contraction, stream-K, MX, quantised bridges |
| Not covered | bridge-vs-TileEngine performance parity A/B measurement |



