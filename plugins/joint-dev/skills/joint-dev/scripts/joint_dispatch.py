#!/usr/bin/env python3
"""Headless Multi-Engine Collaboration Helper for Antigravity `/joint-dev`.

[Role]
Lightweight plumbing CLI that removes environment/CLI friction when collaborating across:
- `Codex Astra Max` (`gpt-6.2-astra` at `effort="max"` by default; read-only or `--write` for math/theory kernels)
- `DSH` (`dsh --profile headless`, high-speed multi-file coding & refactoring)
- `Grok` (`~/.grok/bin/grok`, `500k`-context deep cross-file coding & analysis)
- Live compact status aggregation (`status`) for clean 30-minute Coordinator heartbeats.

[Invariants]
- Pure Python 3.11+ stdlib; zero third-party deps, zero implicit git commits/pushes.
- Prevents accidental fallback to `gpt-6.1-sol` by pinning `gpt-6.2-astra` (or newer `gpt-*-astra`).
- Silences Codex background UI popups (`notify=[]`) and MCP cold-start overhead (`plugins={}`, `mcp_servers={}`).
- Appends only concise, practical collaboration reminders by default (`--raw-prompt` disables all additions).

[Timeouts & Locks]
- Defaults: `2400s` (`40m`) for `astra`, `1800s` (`30m`) for `code`; configurable via `--timeout`.
- Sets `GIT_OPTIONAL_LOCKS=0` so concurrent read-only `git status` checks never contend on index locks.

[Side-Effect Policy]
- Writes logs, `meta.json`, and extracted outputs strictly inside caller-supplied `--out` / `--run-dir`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

DEFAULT_ASTRA_MODEL = "gpt-6.2-astra"
PROXY_ENV_KEYS = ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY")

ASTRA_REMINDERS: dict[str, str] = {
    "tech-closure": """

---
### Collaboration Notes (Technical Closure)
- Treat the product goals, usability, interaction surface, and macro system logic above as our agreed direction; focus your depth on closing the technical implementation cleanly.
- Inspect the real repository and any mounted `--add-dir` reference sources thoroughly so nothing relies on guesswork or stale assumptions.
- Favor structural unification and deleting legacy/fallback baggage over adding complexity or imaginary-enemy defensive checks.
""",
    "deep-review": """

---
### Collaboration Notes (Deep Systemic Review & Detailed Root-Cause Fix Blueprint)
- Take full time to trace real production entrypoints, call graphs, and mounted `--add-dir` upstream sources end-to-end.
- **Do NOT compress fixes into 1–2 terse sentences tacked onto individual review bullets.** Unless `VERDICT: PASS`, structure your output into two explicit parts:
  1. `## Part 1 — Verified Findings & Counterexamples`: List each genuine `[P0 | P1 | SYSTEMIC | DOC-FACT] <file:line>` with the concrete reproduced failure mechanism / counterexample and tag which Root-Cause Group (`RC-1`, `RC-2`, ...) owns it.
  2. `## Part 2 — Detailed Systemic Fix Blueprint (By Root-Cause Group)`: Cluster related symptoms into a small number of unified structural designs (`RC-1`, `RC-2`, ...). For each group, specify the **complete, implementation-ready systemic solution**—unified data model / state invariants, exact algorithm & control-flow steps, function/module boundaries, what ad-hoc or legacy branches to delete, and resource/line-budget impact—so downstream engineers implement cohesive architecture rather than 15 piecemeal `if` band-aids.
- Skip stylistic nitpicks and imaginary-enemy defensive programming (e.g., internal `isinstance` spam, "just-in-case" fallback shims, or single-user local ACL/symlink paranoia).
""",
    "author": """

---
### Collaboration Notes (Direct Math / Core Authoring)
- Edit only the in-scope code/test files in the working tree (do not run `git commit`/`git push` or write user-facing `README`/`docs/` prose).
- Keep equations, invariants, and real-entrypoint wiring exact, general (not hardcoded to single case IDs), and free of silent fallbacks.
""",
    "consult": "",
}

CODER_REMINDER = """

---
### Collaboration Notes (Scoped Worktree Slice)
- **Scoped multi-agent slice**: Even if repository `AGENTS.md`/`CONTRIBUTING.md` requires PR-level doc updates, your slice in this pipeline covers **code, configs, and tests only**—a dedicated `joint-docs-writer` agent updates `README*`/`CONTRIBUTING*`/`docs/**` in the next step before PR/commit. Keep doc files untouched unless explicitly listed in your prompt above, and note any user-facing contract changes in your summary for the docs pass.
- Wire changes through the real production entrypoint, delete superseded legacy/silent-fallback branches cleanly, and keep logic structurally general rather than hardcoded to specific case IDs or magic-constant bypasses.
- Leave `git commit`/`git push` to the Coordinator. When using `ssh`/`scp`/`rsync` for remote server verification, pass `-o ControlMaster=no -o BatchMode=yes` so background muxes never hang stdout.
"""


def _augmented_path(base_path: str | None = None) -> str:
    parts = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        str(Path("~/.grok/bin").expanduser()),
        str(Path("~/.local/bin").expanduser()),
        base_path if base_path is not None else os.environ.get("PATH", ""),
    ]
    return ":".join(p for p in parts if p)


def _find_bin(name: str, extra_candidates: tuple[str, ...] = ()) -> str | None:
    found = shutil.which(name, path=_augmented_path())
    if found:
        return found
    for cand in extra_candidates:
        p = Path(cand).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def _resolve_bins() -> dict[str, str | None]:
    return {
        "codex": _find_bin(
            "codex",
            (
                "/opt/homebrew/bin/codex",
                "/usr/local/bin/codex",
                "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex",
            ),
        ),
        "dsh": _find_bin("dsh", ("/opt/homebrew/bin/dsh", "~/.local/bin/dsh")),
        "grok": _find_bin("grok", ("~/.grok/bin/grok", "/opt/homebrew/bin/grok")),
    }


def _detect_default_astra_model() -> str:
    catalog = Path("~/.codex/magpie-desktop-models.json").expanduser()
    best_ver = (6, 2)
    best_slug = DEFAULT_ASTRA_MODEL
    if catalog.is_file():
        try:
            data = json.loads(catalog.read_text(encoding="utf-8", errors="replace"))
            models = data.get("models", data) if isinstance(data, dict) else data
            for item in models if isinstance(models, list) else []:
                slug = str(item.get("slug", "")) if isinstance(item, dict) else ""
                m = re.fullmatch(r"gpt-(\d+)(?:\.(\d+))?-astra", slug)
                if m:
                    ver = (int(m.group(1)), int(m.group(2) or 0))
                    if ver > best_ver:
                        best_ver = ver
                        best_slug = slug
        except TimeoutError:
            raise
        except (OSError, ValueError):
            pass
    return best_slug


def _clean_env(*, keep_proxy: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = _augmented_path(env.get("PATH", ""))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    if not keep_proxy:
        for k in PROXY_ENV_KEYS:
            env.pop(k, None)
    return env


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git_changed_files(cwd: Path) -> list[str]:
    if not cwd.is_dir():
        return []
    try:
        res = subprocess.run(
            ["git", "-C", str(cwd), "status", "--porcelain"],
            capture_output=True,
            text=True,
            env=_clean_env(keep_proxy=True),
            timeout=10,
            check=False,
        )
        if res.returncode != 0:
            return []
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except (TimeoutError, subprocess.TimeoutExpired):
        return ["<git status timed out>"]
    except OSError:
        return []


def _extract_last_report_from_jsonl(events_path: Path) -> str:
    if not events_path.is_file():
        return ""
    last_text = ""
    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                for candidate in (
                    obj.get("text"),
                    obj.get("result"),
                    obj.get("content") if isinstance(obj.get("content"), str) else None,
                    (obj.get("message") or {}).get("content") if isinstance(obj.get("message"), dict) else None,
                ):
                    if isinstance(candidate, str) and len(candidate.strip()) > 20:
                        last_text = candidate.strip()
                msg = obj.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                    chunks = [
                        blk.get("text", "")
                        for blk in msg["content"]
                        if isinstance(blk, dict) and blk.get("type") == "text"
                    ]
                    joined = "\n".join(c for c in chunks if c).strip()
                    if len(joined) > 20:
                        last_text = joined
    except TimeoutError:
        raise
    except OSError:
        return ""
    return last_text


def cmd_doctor(args: argparse.Namespace) -> int:
    bins = _resolve_bins()
    cwd = Path(args.cwd).expanduser().resolve()
    codex_cfg = Path("~/.codex/config.toml").expanduser()
    configured_codex_model = None
    if codex_cfg.is_file():
        for ln in codex_cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r'^\s*model\s*=\s*"([^"]+)"', ln)
            if m:
                configured_codex_model = m.group(1)
                break
    report = {
        "bins": bins,
        "astra_pinned_model": _detect_default_astra_model(),
        "astra_default_effort": "max",
        "codex_config_default_model": configured_codex_model,
        "dsh_headless_profile_exists": Path("~/.dsh/profiles/headless").expanduser().is_dir(),
        "grok_config_exists": Path("~/.grok/config.toml").expanduser().is_file(),
        "cwd": str(cwd),
        "cwd_git_changes": len(_git_changed_files(cwd)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [k for k in ("codex", "dsh") if not bins.get(k)]
    return 1 if missing else 0


def cmd_astra(args: argparse.Namespace) -> int:
    bins = _resolve_bins()
    codex_bin = bins["codex"]
    if not codex_bin:
        sys.stderr.write("ERROR: `codex` CLI not found on PATH.\n")
        return 2

    cwd = Path(args.cwd).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file = out_file.with_suffix(out_file.suffix + ".stderr.log")
    meta_file = out_file.with_suffix(out_file.suffix + ".meta.json")

    model = args.model or _detect_default_astra_model()
    writable = bool(args.write or args.mode == "author")
    sandbox = "workspace-write" if writable else "read-only"
    suffix = "" if args.raw_prompt else ASTRA_REMINDERS.get(args.mode, "")
    prompt_text = prompt_file.read_text(encoding="utf-8").rstrip() + suffix + "\n"

    cmd: list[str] = [
        codex_bin,
        "exec",
        "--ephemeral",
        "-s",
        sandbox,
        "--skip-git-repo-check",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{args.effort}"',
        "-c",
        "notify=[]",
        "-c",
        "plugins={}",
        "-c",
        "mcp_servers={}",
        "-C",
        str(cwd),
        "-o",
        str(out_file),
    ]
    for extra in args.add_dir:
        cmd.extend(["--add-dir", str(Path(extra).expanduser().resolve())])
    cmd.append("-")

    t0 = time.time()
    _write_json_atomic(
        meta_file,
        {
            "kind": "astra",
            "mode": args.mode,
            "sandbox": sandbox,
            "model": model,
            "effort": args.effort,
            "cwd": str(cwd),
            "add_dirs": [str(Path(p).expanduser().resolve()) for p in args.add_dir],
            "out_file": str(out_file),
            "stderr_file": str(stderr_file),
            "state": "running",
            "started_at": t0,
        },
    )

    state = "failed"
    rc = 1
    effective_timeout = max(args.timeout, 1800) if args.effort in ("max", "ultra") and not args.allow_short_timeout else args.timeout
    try:
        with stderr_file.open("w", encoding="utf-8") as err_fh:
            for extra in args.add_dir:
                ep = Path(extra).expanduser().resolve()
                if ep.is_dir() and not (ep / ".git").exists() and sum(1 for _ in ep.iterdir()) > 40:
                    err_fh.write(
                        f"WARNING: --add-dir {ep} is a large non-repo parent folder; prefer mounting specific sibling repos.\n"
                    )
                    err_fh.flush()
            proc = subprocess.run(
                cmd,
                input=prompt_text,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=err_fh,
                cwd=str(cwd),
                env=_clean_env(keep_proxy=args.keep_proxy),
                timeout=effective_timeout,
                check=False,
            )
            rc = proc.returncode
            state = "completed" if (rc == 0 and out_file.is_file() and out_file.stat().st_size > 0) else "failed"
    except (TimeoutError, subprocess.TimeoutExpired):
        state = "timed_out"
        rc = 124
    finally:
        elapsed = round(time.time() - t0, 2)
        out_bytes = out_file.stat().st_size if out_file.is_file() else 0
        changed = _git_changed_files(cwd) if writable else []
        _write_json_atomic(
            meta_file,
            {
                "kind": "astra",
                "mode": args.mode,
                "sandbox": sandbox,
                "model": model,
                "effort": args.effort,
                "cwd": str(cwd),
                "out_file": str(out_file),
                "stderr_file": str(stderr_file),
                "state": state,
                "returncode": rc,
                "elapsed_s": elapsed,
                "out_bytes": out_bytes,
                "changed_files_count": len(changed),
            },
        )

    print(
        json.dumps(
            {
                "state": state,
                "sandbox": sandbox,
                "returncode": rc,
                "elapsed_s": elapsed,
                "out_file": str(out_file),
                "out_bytes": out_bytes,
            },
            ensure_ascii=False,
        )
    )
    return 0 if state == "completed" else (rc or 1)


def cmd_code(args: argparse.Namespace) -> int:
    bins = _resolve_bins()
    engine: str = args.engine
    engine_bin = bins.get(engine)
    if not engine_bin:
        sys.stderr.write(f"ERROR: `{engine}` CLI not found on PATH.\n")
        return 2

    cwd = Path(args.cwd).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_prompt_path = run_dir / "effective_prompt.md"
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    report_path = run_dir / "report.md"
    meta_path = run_dir / "meta.json"

    raw_prompt = prompt_file.read_text(encoding="utf-8").rstrip()
    effective_prompt = raw_prompt + ("" if args.raw_prompt else CODER_REMINDER) + "\n"
    effective_prompt_path.write_text(effective_prompt, encoding="utf-8")

    if engine == "dsh":
        cmd = [engine_bin, "--profile", args.dsh_profile, "--json", "-"]
        stdin_text: str | None = effective_prompt
    else:
        cmd = [
            engine_bin,
            "--prompt-file",
            str(effective_prompt_path),
            "--cwd",
            str(cwd),
            "--always-approve",
            "--no-plan",
            "--no-subagents",
            "--output-format",
            "streaming-json",
        ]
        if args.model:
            cmd.extend(["-m", args.model])
        if args.effort:
            cmd.extend(["--effort", args.effort])
        for extra in args.add_dir:
            cmd.extend(["--add-dir", str(Path(extra).expanduser().resolve())])
        stdin_text = None

    t0 = time.time()
    _write_json_atomic(
        meta_path,
        {
            "kind": "code",
            "engine": engine,
            "cwd": str(cwd),
            "run_dir": str(run_dir),
            "events_file": str(events_path),
            "stderr_file": str(stderr_path),
            "state": "running",
            "started_at": t0,
        },
    )

    state = "failed"
    rc = 1
    try:
        with events_path.open("w", encoding="utf-8") as out_fh, stderr_path.open("w", encoding="utf-8") as err_fh:
            proc = subprocess.run(
                cmd,
                input=stdin_text,
                text=True,
                stdout=out_fh,
                stderr=err_fh,
                cwd=str(cwd),
                env=_clean_env(keep_proxy=args.keep_proxy),
                timeout=args.timeout,
                check=False,
            )
            rc = proc.returncode
            state = "completed" if rc == 0 else "failed"
    except (TimeoutError, subprocess.TimeoutExpired):
        state = "timed_out"
        rc = 124
    finally:
        elapsed = round(time.time() - t0, 2)
        report_text = _extract_last_report_from_jsonl(events_path)
        if report_text:
            report_path.write_text(report_text + "\n", encoding="utf-8")
        changed = _git_changed_files(cwd)
        event_lines = sum(1 for _ in events_path.open("r", encoding="utf-8", errors="replace")) if events_path.is_file() else 0
        _write_json_atomic(
            meta_path,
            {
                "kind": "code",
                "engine": engine,
                "cwd": str(cwd),
                "run_dir": str(run_dir),
                "events_file": str(events_path),
                "stderr_file": str(stderr_path),
                "report_file": str(report_path) if report_path.is_file() else None,
                "state": state,
                "returncode": rc,
                "elapsed_s": elapsed,
                "event_lines": event_lines,
                "changed_files_count": len(changed),
                "changed_files_sample": changed[:30],
            },
        )

    print(
        json.dumps(
            {
                "engine": engine,
                "state": state,
                "returncode": rc,
                "elapsed_s": elapsed,
                "event_lines": event_lines,
                "changed_files_count": len(changed),
                "report_file": str(report_path) if report_path.is_file() else None,
            },
            ensure_ascii=False,
        )
    )
    return 0 if state == "completed" else (rc or 1)


def _summarize_meta(meta_file: Path) -> dict[str, Any]:
    data = json.loads(meta_file.read_text(encoding="utf-8", errors="replace"))
    if data.get("state") == "running" and isinstance(data.get("started_at"), (int, float)):
        data["live_elapsed_s"] = round(time.time() - float(data["started_at"]), 1)
        for log_key, count_key in (("events_file", "live_event_lines"), ("stderr_file", "live_stderr_lines")):
            p = data.get(log_key)
            if p and Path(p).is_file():
                with Path(p).open("r", encoding="utf-8", errors="replace") as fh:
                    data[count_key] = sum(1 for _ in fh)
        cwd = data.get("cwd")
        if cwd:
            data["live_changed_files_count"] = len(_git_changed_files(Path(cwd)))
    return data


def cmd_status(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    metas: list[Path] = []
    if target.is_file() and target.name.endswith("meta.json"):
        metas.append(target)
    elif target.is_dir():
        metas.extend(sorted(target.rglob("*meta.json")))
    summaries = [_summarize_meta(m) for m in metas]
    print(json.dumps({"count": len(summaries), "runs": summaries}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collaboration dispatcher for /joint-dev (Astra Max + DSH/Grok)")
    sub = ap.add_subparsers(dest="subcmd", required=True)

    p_doc = sub.add_parser("doctor", help="Preflight codex/astra, dsh, and grok CLIs")
    p_doc.add_argument("--cwd", default=".")
    p_doc.set_defaults(func=cmd_doctor)

    p_astra = sub.add_parser("astra", help="Run Codex Astra Max (read-only technical closure/review, or --write math authoring)")
    p_astra.add_argument(
        "--mode",
        choices=("tech-closure", "deep-review", "author", "consult"),
        default="consult",
        help="Role hint (default: consult; author enables workspace-write automatically)",
    )
    p_astra.add_argument("--write", action="store_true", help="Allow workspace-write for math/formal core authoring")
    p_astra.add_argument("--raw-prompt", action="store_true", help="Send prompt verbatim without appending role notes")
    p_astra.add_argument("--cwd", default=".")
    p_astra.add_argument("--add-dir", action="append", default=[])
    p_astra.add_argument("--prompt-file", required=True)
    p_astra.add_argument("--out", required=True)
    p_astra.add_argument("--model", default=None, help="Override model slug (default: gpt-6.2-astra)")
    p_astra.add_argument("--effort", default="max", choices=("high", "xhigh", "max", "ultra"))
    p_astra.add_argument("--timeout", type=int, default=2400)
    p_astra.add_argument("--allow-short-timeout", action="store_true", help="Allow <1800s timeout on max/ultra effort")
    p_astra.add_argument("--keep-proxy", action="store_true")
    p_astra.set_defaults(func=cmd_astra)

    p_code = sub.add_parser("code", help="Dispatch DSH or Grok coding pass in a worktree")
    p_code.add_argument("--engine", choices=("dsh", "grok"), default="dsh")
    p_code.add_argument("--raw-prompt", action="store_true", help="Send prompt verbatim without appending coder notes")
    p_code.add_argument("--cwd", required=True)
    p_code.add_argument("--add-dir", action="append", default=[])
    p_code.add_argument("--prompt-file", required=True)
    p_code.add_argument("--run-dir", required=True)
    p_code.add_argument("--dsh-profile", default="headless")
    p_code.add_argument("--model", default=None, help="Optional Grok model override")
    p_code.add_argument("--effort", default=None, help="Optional Grok reasoning effort override")
    p_code.add_argument("--timeout", type=int, default=1800)
    p_code.add_argument("--keep-proxy", action="store_true")
    p_code.set_defaults(func=cmd_code)

    p_stat = sub.add_parser("status", help="Summarize live/completed runs under a scratch directory")
    p_stat.add_argument("target", help="Run directory, scratch parent directory, or *meta.json file")
    p_stat.set_defaults(func=cmd_status)
    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
