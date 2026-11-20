# `/joint-dev` Collaboration Patterns, Prompt Starters & Triage Guide

Use these as **flexible starting points**—adapt freely to the repo, language, and task shape rather than treating them as rigid forms.

---

## 1. Compaction-Proof `<scratch>/macro_brief.md` Header

Keep `<scratch>/macro_brief.md` concise and updated at stage transitions so context compaction never causes Coordinator amnesia or regression into solo single-threaded grinding:

```markdown
# Macro Brief: <Topic>
- **Collaboration State**: Wave <N> (<status>) | Heartbeat TaskId: `<task_id_or_none>`
- **Active Worktrees & Branches**: `<repo>: <wt_path> (<branch>)`
- **Active Subagents / Runs**: `<role>: <conversation_id_or_run_dir>`
- **Core UX & Systemic Goal**: <1-paragraph product & clean-core evolution goal>
- **Empirical / Upstream Truths Locked**: <verified commits, hardware SKUs, benchmark packets>
- **Open Product Decisions**: <none or exact question>
```

---

## 2. Technical Closure Prompt Starter (`astra --mode tech-closure --effort max`)

Give `Astra Max` your Product/UX & Macro Architecture direction (`macro_brief.md`) plus full repo + `--add-dir` upstream access so it closes hard technical details on real local code:

```markdown
# Technical Closure: <Topic>

## Product UX & Macro Direction (Agreed Baseline)
- Goal & user/agent experience: <what it does, CLI/API ergonomics, non-goals>
- Macro architecture & subtraction targets: <SSOT boundaries, legacy/v1-v2/silent-fallback code to delete>
- Real production entrypoint(s) to wire end-to-end: <entrypoints>

## Ground-Truth Sources (Verified Fresh)
- Target repo (`-C`): `<repo_path>` | Reference repos (`--add-dir`): `<upstream_paths>`
- Empirical / benchmark artifacts on disk: `<paths_if_any>`
- Open issues / context to absorb if relevant: <issue numbers or constraints>

## What to Close Technically
1. Verify our macro plan against actual local code & upstream contracts (`file:line`), flagging any hidden mismatch.
2. Lock down exact data structures, conservation equations/algorithms, state/lock/timeout invariants, and cross-platform details.
3. Propose a clean file-level edit & subtraction plan (including extracting `< 400`-line helpers if an existing `> 800`-line monolith owns a root cause), natural same-wave worktree boundaries, and independent verification gates.
```

> **Tip — Math / Theory Heavy Core (`--mode author --write`)**: When a self-contained module is 90% hard math, formal invariants, or subtle algorithm derivation and 10% boilerplate, let `Astra Max` author that core directly via `joint_dispatch.py astra --mode author --write`, while keeping multi-file engineering on `dsh` / `grok`.

---

## 3. Mid-Campaign Systemic Re-Closure (`Empirical / Side-Track Discovery -> Core Evolution`)

Trigger this pattern whenever **live hardware measurements, accuracy calibration, or urgent side-missions** reveal that an earlier wave's assumptions or scalar parameters cannot physically close the problem:

1. **Stop Parameter Tuning (`<= 2`-Attempt Circuit Breaker)**: Never let a coder spin 500+ steps nudging multipliers against a structural mismatch (e.g., per-owner bandwidth caps, un-deduplicated gather fan-out, UB/SRAM spills, dual host-submission paths in a legacy monolith, or `L2 == L3` duplicated test records).
2. **Distill a Crisp Delta Spec (`1–2` pages)**: Extract the proven equations, regime boundaries, and counterexamples from the empirical packet or side-track solver.
3. **Re-Open Architecture Cleanly**: Run a focused `Astra Max` `tech-closure` consult if math/invariants changed, reopen whatever core files own the mechanism (extracting clean `< 400`-line submodules if a legacy file is `> 800` lines), and execute as clean parallel slices before running final calibration & `joint-docs-writer`.

---

## 4. Worktree Coder Brief Starter (`code --engine dsh` or `grok`)

Keep coder briefs concrete, wave-scoped, and protected by the anti-spin circuit breaker:

```markdown
# Coding Task (Wave <N>): <Slice Name>

- **Worktree & Wave Scope**: `<wt_path>` — own `<in_scope_files>` in this wave to avoid parallel merge conflicts (leave `<parallel_sibling_files>` and `README`/`docs/` untouched).
- **What to Build & Subtract**:
  - Delete `<legacy branches / duplicate definitions / silent fallbacks>`.
  - Implement `<exact contracts, equations, schemas, error handling>`.
  - Wire end-to-end through `<real production CLI/API entrypoint>` and keep logic structurally general (zero hardcoded case/model IDs or `ZERO`/`ONE` formula bypasses).
- **Verification & Anti-Spin Circuit Breaker**:
  - Run `<exact local or remote server test/benchmark commands>`.
  - **Circuit Breaker**: If a test or `< X%` accuracy bound does not converge after **2 principled attempts** due to a structural mismatch in an out-of-slice or legacy module, **STOP tuning constants immediately**, leave clean state, and report the exact `file:line` + counterexample in your summary.
```

---

## 5. Systemic Review & Secondary Gatekeeping (`astra --mode deep-review --effort max`)

```markdown
# Deep Systemic Audit (Round <1|2>): <Target>

- **Scope**: `<wt_path>` (diff vs `<base_sha>`, plus callers, real entrypoints, verification datasets, and `--add-dir` upstream sources).
- **Intended Product & Architecture**: <brief summary>

## What to Hunt & Required Two-Part Output
1. **Real Functional / Math / Evaluation-Honesty / Runtime Bugs**: broken invariants, mismatch with real upstream code, disconnected real-entrypoint wiring, `train`/`heldout` leakage or `L2 == L3` duplicated evaluation records, heuristic fudge factors masking structural bugs, silent fallbacks, concurrency/lock/timeout/OS bugs, or CI/perf bottlenecks.
2. **Unified Systemic Fix Blueprint (NOT Terse One-Line Bullet Tails)**:
   - `## Part 1 — Verified Findings & Counterexamples`: `[P0 | P1 | SYSTEMIC | DOC-FACT] <file:line>`, concrete failure path / counterexample, and assigned Root-Cause Group (`RC-1`, `RC-2`, ...).
   - `## Part 2 — Detailed Systemic Fix Blueprint (By Root-Cause Group)`: Complete engineering blueprint per root-cause group (unified data structures & invariants, exact single-pass algorithm / call-chain steps, edge-case rules, and obsolete/band-aid branches to delete).
   - Skip stylistic nitpicks and imaginary-enemy defensive checks. If clean, end with `VERDICT: PASS`.
```

### Secondary Gatekeeping Triage Matrix
- **Systemic Root Cause / Evaluation Leakage / Real `P0`–`P1` Bug** -> **Keep**: fix the root structure cleanly and delete obsolete paths.
- **Real Bug + Band-Aid Suggestion** -> **Upgrade**: keep the diagnosis, rewrite into a clean structural solution (splitting oversized legacy modules if needed) instead of adding special-case `if` branches or scalar nudges.
- **Imaginary-Enemy Defensive Programming / Nitpick** -> **Drop**: reject internal `isinstance`/None paranoia, "just-in-case" fallback shims, single-user local ACL/symlink checks, and style bike-shedding.
- **Conflict Between Review, Issue, or Benchmark** -> **Verify Factually First**: check fresh upstream code or run a quick measurement on real hardware/server; escalate to the user only when product intent or release scope is genuinely ambiguous.
