# `/joint-dev` Collaboration Patterns, Prompt Starters & Triage Guide

Use these as **flexible starting points**—adapt freely to the repo, language, and task shape rather than treating them as rigid forms.

---

## 1. Technical Closure Prompt Starter (`astra --mode tech-closure --effort max`)

Give `Astra Max` your Product/UX & Macro Architecture direction (`macro_brief.md`) plus full repo + `--add-dir` upstream access so it closes the hard technical details without altering UX simplicity:

```markdown
# Technical Closure: <Topic>

## Product UX & Macro Direction (Agreed Baseline)
- Goal & user/agent experience: <what it does, CLI/API ergonomics, non-goals>
- Macro architecture & subtraction targets: <SSOT boundaries, legacy/v1-v2/silent-fallback code to delete>
- Real production entrypoint(s) to wire end-to-end: <entrypoints>

## Ground-Truth Sources (Verified Fresh)
- Target repo (`-C`): `<repo_path>` | Reference repos (`--add-dir`): `<upstream_paths>`
- Open issues / context to absorb if relevant: <issue numbers or constraints>

## What to Close Technically
1. Verify our macro plan against actual code & upstream contracts (`file:line`), flagging any hidden mismatch.
2. Lock down exact data structures, formulas/algorithms, state/lock/timeout invariants, and cross-platform details.
3. Propose a clean file-level edit & subtraction plan (and natural disjoint worktree boundaries if parallelizable), plus concise orthogonal verification cases.
```

> **Tip — Math / Theory Heavy Core (`--mode author --write`)**: When a small module is 90% hard math, formal invariants, or subtle algorithm derivation and 10% boilerplate, you can let `Astra Max` author that core directly in an isolated worktree via `joint_dispatch.py astra --mode author --write`, while keeping heavy multi-file engineering on `dsh` / `grok`.

---

## 2. Worktree Coder Brief Starter (`code --engine dsh` or `grok`)

Keep coder briefs concrete and outcome-oriented:

```markdown
# Coding Task: <Slice Name>

- **Worktree & Scope**: `<wt_path>` — focus on `<in_scope_files>` (leave `<out_of_scope>` and `README`/`docs/` untouched).
- **What to Build & Subtract**:
  - Delete `<legacy branches / duplicate definitions / silent fallbacks>`.
  - Implement `<exact contracts, formulas, schemas, error handling>`.
  - Wire end-to-end through `<real production CLI/API entrypoint>` and keep logic structurally general (no hardcoded case/model IDs or `ZERO`/`ONE` formula bypasses).
- **Verification**:
  Run `<exact local or remote server test/benchmark commands>` and summarize changed files + test results.
```

---

## 3. Systemic Review Prompt Starter (`astra --mode deep-review --effort max`)

```markdown
# Deep Systemic Audit (Round <1|2>): <Target>

- **Scope**: `<wt_path>` (diff vs `<base_sha>`, plus callers, real entrypoints, and `--add-dir` upstream sources).
- **Intended Product & Architecture**: <brief summary>

## What to Hunt & Required Two-Part Output
1. **Real Functional / Math / Factual / Runtime Bugs**: broken invariants, mismatch with real upstream code, disconnected real-entrypoint wiring, silent fallbacks, magic-constant formula bypasses, concurrency/lock/timeout/OS bugs, or CI/perf bottlenecks.
2. **Unified Systemic Fix Blueprint (NOT Terse One-Line Bullet Tails)**:
   - Do **not** compress fixes into 1–2 terse sentences at the end of each bullet (which tempts coders into patching 15 bullets with 15 `if` branches).
   - Output **two structured sections**:
     - `## Part 1 — Verified Findings & Counterexamples`: `[P0 | P1 | SYSTEMIC | DOC-FACT] <file:line>`, concrete failure path / reproduced counterexample, and assigned Root-Cause Group (`RC-1`, `RC-2`, ...).
     - `## Part 2 — Detailed Systemic Fix Blueprint (By Root-Cause Group)`: Group related findings by shared structural root cause and write out the **complete engineering blueprint** for each group (unified data structures & invariants, exact single-pass algorithm / call-chain steps, edge-case rules, and obsolete branches to delete).
   - Skip stylistic nitpicks and imaginary-enemy defensive checks. If clean, end with `VERDICT: PASS`.
```

---

## 4. Secondary Gatekeeping Guide (Filtering Astra's Output)

Never forward raw review output blindly to `dsh` / `grok`—apply engineering judgment:

- **Systemic Root Cause / Real `P0`–`P1` Bug** -> **Keep**: fix the root structure cleanly and delete obsolete paths.
- **Real Bug + Band-Aid Suggestion** -> **Upgrade**: keep the diagnosis, rewrite the fix into a clean structural solution instead of adding special-case `if` branches or constants.
- **Imaginary-Enemy Defensive Programming / Nitpick** -> **Drop**: reject internal `isinstance`/None paranoia, "just-in-case" fallback shims, single-user local ACL/symlink checks, and style bike-shedding.
- **Conflict Between Review, Issue, or Benchmark** -> **Verify Factually First**: check fresh upstream code or run a quick measurement on real hardware/server; escalate to the user only when product intent itself is ambiguous.
