---
name: joint-dev
description: "Manual-trigger (/joint-dev) multi-agent collaboration framework: clean Coordinator with compaction-proof macro_brief anchor & bounded 30m heartbeats, empirical-first systemic evolution, cross-platform subagents + Codex Astra Max + DSH/Grok routing with anti-spin circuit breakers, anti-defensive review gatekeeping, and late-stage strong-model documentation."
disable-model-invocation: true
---

# Joint Dev Collaboration Framework (`联合设计 + 系统性演进 + DSH/Grok 开发 + 联合深审`)

This skill is a **flexible multi-agent collaboration framework, NOT a rigid waterfall**. Adapt team size, waves, and engine routing dynamically so the codebase undergoes **clean systemic evolution** rather than piecemeal patching or endless parameter tuning.

- **Plumbing Helper**: `~/.gemini/config/plugins/joint-dev/skills/joint-dev/scripts/joint_dispatch.py`
  *(Invoke via `python3` on macOS/Linux, or `uv run --no-project python` on Windows when `python3` is a Store stub; run `doctor --cwd <repo>` once up front.)*
- **Reference Patterns & Triage Guide** *(on-demand)*: [references/playbooks-and-guardrails.md](./references/playbooks-and-guardrails.md)
- **Specialist Subagents** *(composable building blocks)*:
  - `joint-design-architect` (`Model: "inherit"`) — Codebase/upstream survey + `Astra Max` technical closure & mid-campaign systemic re-closure.
  - `joint-coder-runner` (`Model: "flash"`) — Lightweight worktree runner, remote server test offloader & mechanical retry helper.
  - `joint-review-gatekeeper` (`Model: "inherit"`) — Mandatory `Astra Max` systemic root-cause & evaluation-honesty review + anti-defensive filtering.
  - `joint-docs-writer` (`Model: "inherit"`) — Dedicated strong-model author for all human/agent-facing docs (invoked **after** code & metrics stabilize).

---

## 1. Standing Collaboration Agreements (The "Never Re-Explain" Defaults)

1. **Main Coordinator Context Purity & Compaction-Proof Anchor**:
   - Keep your main context **clean and high-signal**: own user/product UX intent, `<scratch>/macro_brief.md`, cross-track synthesis, and user communication—push heavy file reading, 40KB Astra outputs, diffs, and test/Excel inspection into subagents.
   - **Compaction-Proof State Anchor**: Maintain active subagent IDs, active heartbeat `TaskId`, worktree paths, and collaboration mode at the top of `<scratch>/macro_brief.md`. **Immediately after any context compaction (`# Resuming from a compaction`) or when the user adds a new urgent task mid-session**, re-read `<scratch>/macro_brief.md` FIRST and fan out parallel subagents—**never regress into running dozens of solo `read_file`/`run_command`/MCP steps in Main Chat**.
   - **Act decisively on engineering facts**: Only pause to ask the user when **product intent, scope trade-offs, or UX direction** genuinely fork.

2. **Bounded Reporting Rhythm (`30-Minute Heartbeat` + `Zero Orphan Crons`)**:
   - For runs `> 5–10` min, arm `schedule` (`CronExpression: "*/30 * * * *"`, `IsDaemon: false`, `MaxIterations: 16`) and record its `TaskId` in `macro_brief.md`. Keep each heartbeat check lightweight (`joint_dispatch.py status <scratch>` + `manage_subagents`), report concise progress/ETA/blockers **every 30 minutes** AND **at major stage breakthroughs**, and **immediately `manage_task(Action="kill")` the timer as soon as background work completes or pauses for user input** so orphan crons never bloat context.

3. **Empirical-First Systemic Evolution & Side-Track Knowledge Re-Entry (Never Lock Into Local Patching)**:
   - **Empirical Truth Before Freezing Slices**: When a campaign depends on live hardware/NPU/GPU measurements, full-range sweeps (e.g., `512 -> 400K` sequence grids, `TP / DP / Local-TP / PCP / DCP` topologies), or unmerged upstream PRs, verify the full empirical & upstream domain **before** locking final coder slices—or explicitly structure **Wave 1 (Structural Spine)** + **Wave 2 (Systemic Re-Closure)** once empirical truth lands.
   - **Evolve Mechanisms When Tuning Hits Physics**: File-ownership boundaries exist **only to prevent merge collisions between parallel coders in the same wave**—never as permanent walls across waves! Whenever calibration or verification hits a physical counterexample, `Astra` impossibility proof, `L2 == L3` evaluation coupling, or divergent logic inside a legacy `> 800`-line monolith, **stop tweaking scalar constants immediately**: reopen the right modules, split oversized legacy monoliths into clean `< 400`-line helpers, and fix the structural mechanism rather than adding `+10 line` band-aids or punting solvable defects to post-merge issues.
   - **Product-Spine-First & Side-Track Re-Entry**: For mid-session analytical/simulation deliverables, prefer driving or extending the real product engine over building detached shadow scripts in `scratch/`. Whenever standalone verifiers, red-team audits, or urgent side-missions uncover new physical/architectural truths to fold back into the main repo, run an explicit **`Distill Delta Spec -> Astra Max Re-Closure -> Clean Parallel Slices`** pass instead of dumping raw multi-MB JSONs onto one coder.

4. **Strength-Based Routing & Anti-Spin Circuit Breaker (Zero `1,000`-Step Super-Subagents)**:
   - **Technical Closure, Hard Math & Pitfall Hunting (`Subagents + Codex Astra Max`, `effort="max"`)**:
     - Use `joint_dispatch.py astra` (`gpt-6.2-astra` default, or `--model magpie` for `group/magpie-gpt-*-astra`) with full repo + `--add-dir` upstream access (`tech-closure` / `deep-review`, or `--mode author --write` for self-contained math/formal-heavy core modules). Confirm `sandbox_exec_errors == 0` in `joint_dispatch.py` output so `Astra` actually inspected local worktree files rather than falling back to stale GitHub `main`.
   - **High-Speed Engineering & Refactoring (`DSH` / `Grok` in Linked Worktrees)**:
     - **`dsh`** (`--engine dsh`, default) for fast multi-file implementation/subtraction; **`grok`** (`--engine grok`, `500k` context) for massive cross-file/systems slices or loop recovery; fallback to **narrowly scoped** `self` coder subagents only if neither CLI is available.
     - Run single-slice `joint_dispatch.py code` directly as a background command (`0` watcher tokens); use `joint-coder-runner` (`Model: "flash"`) only for multi-wave pipelines or remote server test suites.
   - **Anti-Spin Circuit Breaker (`<= 2` Attempts, Protect `429` Quota)**:
     - **Never overload one subagent** with multi-family mechanism rewrites + dataset/packet building + repo-wide test fixing + doc merging! Split by independent domain/wave.
     - Enforce a hard **2-attempt / pre-compaction circuit breaker** on every coder/calibrator: if a test or accuracy gate (`< X%`) fails to converge after **2 principled iterations** (or if the subagent compacts mid-debug), **STOP tuning numbers immediately**, save clean state, and escalate the exact `file:line` + counterexample to the Coordinator / `joint-design-architect` for structural resolution.
   - **Late-Stage Strong-Model Documentation (`joint-docs-writer`) & Mandatory `joint-review-gatekeeper`**:
     - Never delegate prose to `Astra` or `dsh`, and **never launch `joint-docs-writer` hours early against unfinalized code/metrics**—run it once implementation and numbers stabilize so docs and PR bodies stay 100% factual. Always run `joint-review-gatekeeper` before opening PRs unless the user explicitly waives review for an urgent deadline.

5. **Anti-Defensive Secondary Filtering & `<= 2`-Round Review Convergence**:
   - Require `Astra Max` `deep-review` to trace real execution paths and output **`Part 1 — Verified Findings & Counterexamples`** + **`Part 2 — Detailed Systemic Fix Blueprint (By Root-Cause Group)`** (including auditing train/heldout lineage isolation and multi-level evaluation independence).
   - Filter every finding with engineering judgment: **keep** real `P0`/`P1` and systemic unifications; **upgrade** `if`/constant band-aids into structural fixes; **drop** nitpicks and imaginary-enemy defensive programming (`isinstance` spam, silent fallbacks, local single-user ACL/symlink paranoia). Converge in `<= 2` rounds.

6. **Post-Confirmation End-to-End PR Creation, CI Green & Merge Closure**:
   - Present a concise delivery summary (honestly surfacing any residual trade-off for user sign-off) before pushing, unless pre-authorized.
   - Once confirmed, drive all the way to merged `main`: (1) validate PR bodies via `joint-docs-writer` against repo checklist gates; (2) commit, push, and `gh pr create` in bottom-up cross-repo topological order; (3) watch GitHub Actions CI to green (`gh pr checks --watch` in background, fixing real CI regressions); (4) merge PRs, `git pull --ff-only` base checkouts, `git worktree remove` temporary worktrees, and kill any remaining scheduled heartbeat task.

---

## 2. Engineering Habits Baked In Across All Projects

- **Fresh Ground Truth & Open Issues**: Sync target and `--add-dir` upstream repos to fresh `main` before designing/reviewing, and check open GitHub issues (`gh issue list`) so related root causes are closed systematically in one pass.
- **Subtraction & Zero Silent Fallbacks**: Delete broken/legacy `v1`/`v2` code paths, duplicate structures across sibling repos/modules, and deceptive silent fallback paths cleanly instead of piling on compatibility branches.
- **Real Entrypoint Wiring, Evaluation Independence & Structural Generalization**: Verify through the **real production CLI/API entrypoint**, ensure multi-level benchmarks (`L1` operator / `L2` subgraph / `L3` step, `train` vs `heldout`) test genuinely independent lineages rather than duplicated records, and generalize by structural/tensor properties (zero hardcoded model/case IDs or `ZERO`/`ONE` dodges).
- **Test Orthogonality & Cross-Platform Remote Offload**: Keep tests orthogonal and fast on 2-core CI runners. Offload heavy sweeps/pytest suites to authorized remote Linux/GPU/NPU/WSL servers (`rsync --exclude='.git'` on POSIX, or Python `tarfile` + `scp/ssh -o ControlMaster=no -o BatchMode=yes` on Windows) rather than bottlenecking on local laptop CPU or hitting Windows-only symlink/SQLite quirks.

---

## 3. Scale-Adaptive Orchestration (Choose What Fits the Task)

1. **Focused Fix / Small Feature (`1–3` files)**:
   - Align macro intent -> single `dsh`/`grok` pass (or 1 focused subagent) + 1 `Astra Max` review pass -> `joint-docs-writer` if user-facing -> confirm -> PR, CI green, merge & clean up worktree.
2. **Deep Subsystem Feature / Clean-Core Refactor**:
   - Write `<scratch>/macro_brief.md` -> empirical/upstream recon (if needed) + `joint-design-architect` (`Astra Max` closure + Wave-1 disjoint worktree specs) -> **present concise Design Artifact (`RequestFeedback: true`) and pause for user approval** *(skip pause only when user requests unattended execution)* -> parallel Wave-1 coders -> **Wave-2 Systemic Re-Closure** if integration/calibration uncovers deeper structural or cross-file root causes -> `joint-review-gatekeeper` -> `joint-docs-writer` -> confirm -> topological PRs, CI green, merge, worktree & cron cleanup.
3. **Multi-Repo / Multi-Track Campaign with Urgent Mid-Flight Tasks**:
   - Isolate urgent deliverables into parallel subagent brigades (always parallelizing independent model/sheet/module groups rather than working solo in Main Chat), distill any newly discovered physics/contracts via a **Systemic Re-Entry Blueprint** before folding back into the main repo, and keep `macro_brief.md` updated as the single cross-compaction source of truth.
