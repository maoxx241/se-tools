---
name: joint-dev
description: "Manual-trigger (/joint-dev) multi-agent collaboration framework: Main Antigravity keeps context clean, owns product UX/macro direction & 30m + milestone progress reports, flexibly orchestrates subagents + Codex Astra Max + DSH/Grok by model strengths, filters defensive bloat, and assigns all documentation to a strong-model subagent."
disable-model-invocation: true
---

# Joint Dev Collaboration Framework (`联合设计 + DSH/Grok 开发 + 联合深审`)

This skill is a **flexible multi-agent collaboration framework, NOT a rigid state machine**. Adapt team size, parallelism, and routing dynamically to the actual problem while upholding the standing working agreements below so the user never has to re-explain them.

- **Plumbing Helper**: `~/.gemini/config/plugins/joint-dev/skills/joint-dev/scripts/joint_dispatch.py`
- **Reference Patterns & Triage Guide** *(on-demand)*: [references/playbooks-and-guardrails.md](./references/playbooks-and-guardrails.md)
- **Available Specialist Subagents** *(use as building blocks, not mandatory bureaucracy)*:
  - `joint-design-architect` (`Model: "inherit"`) — Deep codebase survey + `Astra Max` technical closure.
  - `joint-coder-runner` (`Model: "flash"`) — Lightweight worktree runner, remote test offloader & retry helper.
  - `joint-review-gatekeeper` (`Model: "inherit"`) — `Astra Max` systemic root-cause review + anti-defensive filtering.
  - `joint-docs-writer` (`Model: "inherit"`) — Strong-model author for all human/agent-facing documentation.

---

## 1. Standing Collaboration Agreements (The "Never Re-Explain" Defaults)

1. **Main Coordinator Context Purity & Autonomous Judgment**:
   - Keep your main context **clean and high-signal**: hold the user's core intent, product/UX vision, high-level macro plan (`macro_brief.md`), and cross-track progress—push heavy codebase reading, 40KB Astra outputs, large diffs, and test logs into subagents.
   - **Act decisively on engineering principles**: Do NOT interrupt the user to confirm technical micro-decisions that are already clear from physical/codebase facts or the repository's design principles. Only ask when **product intent or UX direction** genuinely forks.

2. **Reporting Rhythm (`30-Minute Heartbeat` + `Major Progress`)**:
   - For any nontrivial run (`> 5–10` min), arm `schedule` (`CronExpression: "*/30 * * * *"`, `IsDaemon: false`) at start, check compact status (`joint_dispatch.py status <scratch>` / `manage_subagents`), and report progress, ETA, and blockers **every 30 minutes** AND **immediately upon major breakthroughs/stage transitions**. Cancel the timer via `manage_task(Action="kill")` as soon as work concludes.

3. **Play Each Model to Its Natural Strengths (Flexible, Not Handcuffed)**:
   - **Product, Usability & Macro Logic (`User + Antigravity`)**: You and the user lead product positioning, CLI/API ergonomics, interaction flow, SSOT boundaries, and legacy-baggage subtraction.
   - **Technical Closure, Hard Math & Systemic Pitfall Hunting (`Subagents + Codex Astra Max`, `effort="max"`)**:
     - Let `Astra Max` take all the time it needs with full repo + upstream `--add-dir` access (`tech-closure` / `deep-review`) to close hard technical details and uncover deep systemic root causes.
     - **Flexible exception**: For **math/formal-heavy, low-boilerplate core modules** where `dsh` is weaker on theory, you may let `Astra Max` author that core directly (`joint_dispatch.py astra --mode author --write`) or have a strong Antigravity subagent step in!
   - **High-Speed Engineering & Refactoring (`DSH` / `Grok` in Git Worktrees)**:
     - **`dsh`** (`--engine dsh`, default): Blazingly fast at executing concrete multi-file specs, deleting dead branches, and aligning tests.
     - **`grok`** (`--engine grok`, `500k` context): Great when requested, when spanning massive cross-file context / systems code, or when `dsh` hits a loop.
     - **Unblocking**: If `dsh`/`grok` can't diagnose a subtle runtime/perf root cause, dispatch a strong subagent to profile/diagnose the root cause rather than letting the coder spin.
     - **Token-smart watching**: For simple clean slices, run `joint_dispatch.py code` as a direct background command (**0 watcher subagent, 0 tokens while waiting**). Spawn `joint-coder-runner` (`Model: "flash"`) only when orchestrating multi-stage waves, remote server test suites, or mechanical retries.
   - **Dedicated Strong-Model Documentation (`joint-docs-writer`, `Model: "inherit"`)**:
     - Never let `Astra` or `dsh` write `README`, `docs/`, `CONTRIBUTING`, guides, or release notes. All natural-language documentation is owned by `joint-docs-writer` (verified against real landed code).

4. **Anti-Defensive Secondary Filtering & `<= 2`-Round Review Convergence**:
   - Require `Astra Max` `deep-review` to trace real execution paths and output **two explicit sections** (never 1–2 sentence fixes squeezed onto bullet tails): `Part 1 — Verified Findings & Counterexamples` + `Part 2 — Detailed Systemic Fix Blueprint (By Root-Cause Group)` grounded in fresh source.
   - **Always filter Astra's review before touching code**: embrace real `P0`/`P1` and systemic unifications; upgrade "band-aid `if` patches" into clean structural fixes; **ruthlessly discard nitpicks and imaginary-enemy defensive programming** (`isinstance` spam on internal contracts, "just-in-case" silent fallbacks, local single-user ACL/symlink paranoia). Converge in `<= 2` review rounds—never enter endless R3..R20 ping-pong.

5. **Post-Confirmation End-to-End PR Creation, CI Green & Merge Closure**:
   - After local gates, `Astra Max` systemic review, and `joint-docs-writer` docs are green, present a concise delivery summary and confirm with the user before pushing to remote (skip asking if the user already instructed to push/merge upfront).
   - **Once confirmed, drive the entire landing pipeline all the way to merged `main`—do NOT stop at `gh pr create`**:
     1. Author PR bodies via `joint-docs-writer` (pre-validated against any repo PR checklist gate);
     2. Commit, push branches, and create PRs (`gh pr create`), honoring bottom-up topological order on cross-repo changes;
     3. Wait for GitHub Actions CI checks to turn green (`gh pr checks --watch` as a background task; diagnose and fix any real CI failure);
     4. **Merge the PRs** (`gh pr merge --squash --delete-branch` or the repo's standard merge mode), fast-forward local base checkouts (`git pull --ff-only`), and remove temporary linked worktrees (`git worktree remove`) so local and remote states end completely clean.

---

## 2. Engineering Habits Baked In Across All Projects

- **Fresh Ground Truth & Open Issues**: Verify target and upstream `--add-dir` repos are up to date before designing/reviewing (never "fix" code against stale upstream sources!), and check open GitHub issues (`gh issue list`) when relevant so related bugs are solved systematically in one pass.
- **Subtraction & Zero Silent Fallbacks**: Delete broken/legacy `v1`/`v2` code paths, duplicate structures across sibling repos/modules, and deceptive silent fallback paths cleanly instead of piling on compatibility branches.
- **Real Entrypoint Wiring & Structural Generalization**: Always verify through the **real production CLI/API entrypoint** (never "unit tests pass on a helper while the real entrypoint is broken/slow"), generalize by structural properties rather than hardcoding entity/model/case IDs, and watch out for magic-constant formula bypasses (`ZERO`/`ONE` dodges).
- **Test Orthogonality, Remote Offload & CI Speed**: Keep tests orthogonal and business-meaningful (purge trivial string-matching filler tests). Offload heavy test/benchmark runs to authorized remote Linux/GPU/NPU/Win+WSL servers (`rsync --exclude='.git'` + `ssh -o ControlMaster=no`) instead of bottlenecking on local laptop CPU, and treat sluggish CI or dependency-ordered merges as first-class deliverables.
- **Pragmatic Engineering Over Rigid Dogma**: Guidelines like module line counts serve clarity—use engineering common sense rather than contorting clean code to hit arbitrary numbers.

---

## 3. Scale-Adaptive Orchestration (Choose What Fits the Task)

1. **Focused Fix / Small Feature (`1–3` files)**:
   - Align macro intent -> run a single subagent (or direct background `dsh`/`grok` + 1 `Astra Max` review pass) -> update docs via `joint-docs-writer` if user-facing -> confirm with user -> create PR, wait for CI green, merge & clean up worktree. Zero unnecessary ceremony.
2. **Deep Subsystem Feature / Clean-Core Refactor**:
   - Draft concise `macro_brief.md` -> `joint-design-architect` (`Astra Max` technical closure + disjoint worktree specs) -> **present a concise User-Facing Design Artifact (`RequestFeedback: true`) highlighting key trade-offs and pause for user review/approval** *(skip pausing only when the user explicitly asks for unattended end-to-end execution)* -> parallel `dsh`/`grok` worktrees -> `joint-review-gatekeeper` (`Astra Max` systemic audit + filter + `<= 2` closure passes) -> `joint-docs-writer` -> confirm with user -> create PRs, wait for CI green, merge & clean up worktrees.
3. **Multi-Repo / Asynchronous Multi-Track Campaign**:
   - Protect **mainline priorities first**; split independent repo groups or tracks (e.g., Accuracy vs Performance vs Docs/Tooling) into **autonomous parallel subagent leads** so fast tracks don't wait on slow tracks, and absorb newly filed issues or mid-flight user ideas smoothly into the right track without derailing in-flight work.
