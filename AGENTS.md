# Teaching your AI agent to run this

This repository ships an **agent skill** — a single Markdown file that gives a coding agent (Claude Code,
or any agent that reads skill/instruction files) enough operational knowledge to run a CK
TileEngine→Dispatcher bridge sweep on gfx1250 end to end, without you re-explaining the environment every
time.

This document is the point of the repository. Read it before the `docs/`.

## 1. Install the skill

### Claude Code

```bash
mkdir -p ~/.claude/skills/mi400-bridge-benchmark
cp skills/mi400-bridge-benchmark/SKILL.md ~/.claude/skills/mi400-bridge-benchmark/SKILL.md
```

Then **fill in the placeholders** in your copy — the committed file is sanitised:

| Placeholder | Fill with |
|---|---|
| `<GFX1250_NODE>` | your gfx1250 host |
| `<GPU_NODE>` | any other GPU host you use |
| `<USER>` | your username on that host |
| `<CK_IMAGE_TAG>` | the CK/ROCm container image tag your team is currently on |
| `<INTERNAL_WIKI_LINK>` | your internal wiki copy of the guide, if you keep one |

Verify the agent picked it up by asking it to list its available skills. The skill's `description`
front-matter carries the trigger conditions, so it activates on prompts about MI400/gfx1250 bridge
benchmarking without you naming it.

### Other agents

The skill is plain Markdown with YAML front matter. For an agent that reads a project instruction file
instead, append the body of `SKILL.md` to that file (for example `AGENTS.md` or `CLAUDE.md` in your CK
working copy), and keep the `docs/` directory alongside it.

## 2. What the skill covers — and what it does not

**Covers:**

- node access, container launch, and the environment traps (no `render` group, bind-mount only, often only
  device 0)
- building CK for gfx1250, including the mandatory `-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc`
- converting a hipBLASLt run CSV into bridge input (layout map, dtype map, group-local `problem_idx`)
- the smoke test, and what a normal smoke result looks like
- the full sweep, its flags, and its outputs
- how to run long jobs without losing them (`docker exec -d` + done-marker, never `pgrep`)
- how to interpret the four result categories honestly

**Does not cover:**

- any operator other than `gemm_universal` — grouped, multi_d, multi_abd, preshuffle, batched,
  batched_contraction, stream-K, MX, and the quantised bridges all have separate drivers and configs
- gfx942 / gfx950 (wave64 + MFMA — the configs are not interchangeable with gfx1250's wave32 + WMMA)
- bridge-vs-TileEngine **performance parity** A/B measurement, which is a different exercise
- obtaining a node reservation, or getting your SSH key registered

## 3. Example prompts

Paste these directly.

### Validate a fresh environment

> Set up the CK bridge benchmark environment on our gfx1250 node and run the llama405b sheet
> (`run_8611.csv`) end to end. Do the smoke test first and stop if it does not look normal. The reference
> result is 210/210 verified — tell me explicitly if we do not hit it, and do not paper over a build
> failure as "environmental".

### Convert a sheet without touching a GPU

> Take `run_8613.csv` and convert it to bridge input with `scripts/sheet_to_bridge_input.py`. Show me the
> per-group counts first with `--dry-run`, then write the files. Tell me how many rows are
> `unsupported_dtype` and why.

### Investigate never-ran shapes

> Here is `master_results.csv` from the wrt1-15 sweep. Break it into verified / runs-but-incorrect /
> never-ran / unsupported with counts. For the never-ran shapes, run `add_fail_reason.py` and group by the
> heuristic reason. Be explicit that the reasons are heuristics, and do not report never-ran shapes as
> correctness failures.

### Split a big sheet across two nodes

> Sheet 8612 is 2,199 rows and too slow for one node. Split it across two gfx1250 nodes by layout using
> `--only-layouts`, run both, then merge with `--merge-only`. Confirm both nodes use the identical
> `--run-csv` before starting — otherwise the indices will not line up.

## 4. Adapting to a different arch or operator

### Different arch (gfx942 / gfx950)

Do **not** just change `--arch`. You must also change:

1. **the config file** — `default_ci_config_gfx1250_pad.json` is gfx1250-specific and produces zero valid
   kernels elsewhere. Find the arch's own default CI config.
2. **`-DGPU_TARGETS`** at CMake time.
3. **the warp-tile expectations in the skill text** — gfx1250's 16×16×32 bf16 / 16×16×64 fp8 are WMMA
   numbers and do not describe MFMA archs.
4. **the reference results** — 210/210 and the 83% figure are gfx1250 measurements. Delete them rather
   than let an agent quote them for another arch.

Copy the skill to a new name (`gfx950-bridge-benchmark`) rather than making one skill conditional. A skill
that tries to cover both archs will confuse the two config sets.

### Different operator

`batch_mi400_pipeline.py` hardcodes `--variant gemm_universal` and the two gemm config paths. For another
operator you need: its driver's argument shape, its config files, its dtype/layout support matrix, and its
own verify tolerances. The conversion logic in `sheet_to_bridge_input.py` is reusable as long as the
operator's input format is still `{"problems": [...]}`.

### Different shape source

Only five source columns matter: `m`, `n`, `k`, `transA`, `transB`, `a_type` (plus `id_csv_entry` for
identity). Any CSV with those can be fed through `classify()` in `sheet_to_bridge_input.py`.

## 5. Reporting rules the agent must not violate

These are non-negotiable. They are in `SKILL.md` too — keep both copies in sync.

1. **A skip is not a pass.** An unsupported-dtype row and a never-ran row must never be counted toward a
   pass rate.
2. **Never-ran (status −1) is not a wrong answer.** All 16 variants refused the arguments and produced no
   output, so nothing can be numerically incorrect. It is a **coverage gap**. Reporting it as a
   correctness failure is factually wrong.
3. **Distinguish all four categories explicitly** in every summary, with counts: PASS (verified) / FAIL
   (runs-but-incorrect) / never-ran / unsupported.
4. **Always state the arch actually tested.** Never present a gfx1250 result as gfx942/gfx950 evidence, or
   the reverse.
5. **`likely_fail_reason` is a heuristic,** derived from M/N/K and layout — not a confirmed root cause.
   Label it as such every time it is quoted.
6. **The `gflops` / `us` columns in the source sheets are from the collection host — a different GPU.**
   They are provenance, not a baseline. Never compute a ratio against them.
7. **Report real compiler and runtime errors verbatim.** Never summarise a genuine build failure as
   "environmental" or "a setup issue" without the actual error text.
8. **If a run is incomplete, say what is missing.** Do not extrapolate a partial sweep into a full-sheet
   percentage.
9. **Flag small samples.** The rrr layout's 42.3% failure rate rests on 52 shapes; it is not as solid as
   the rcr figure and must not be quoted as if it were.
