#!/usr/bin/env python3
"""Cross-Platform Headless Multi-Engine Collaboration Helper for Antigravity `/joint-dev`.

[Role]
Lightweight plumbing CLI that removes environment, OS, and worktree/sandbox friction across:
- `Codex Astra Max` (`gpt-6.2-astra` or `--model magpie` at `effort="max"` by default; read-only or `--write`)
- `DSH` (`dsh --profile headless`, high-speed multi-file coding & refactoring on Windows/macOS/Linux)
- `Grok` (`~/.grok/bin/grok`, `500k`-context deep cross-file coding & analysis)
- Live compact status aggregation (`status`) for clean 30-minute Coordinator heartbeats.

[Invariants]
- Pure Python 3.11+ stdlib (`tomllib`, `subprocess`, `pathlib`); zero third-party deps, zero implicit git commits/pushes.
- Uses `os.pathsep` across platforms and windowless child spawning (`CREATE_NO_WINDOW`) on Windows.
- Prevents Windows Codex `CreateProcessWithLogonW failed: 1385` sandbox blindness while auditing read-only git cleanliness.
- Auto-mounts parent `.git` `commondir` via `--add-dir` when running inside linked Git worktrees.

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
import tomllib
from typing import Any

DEFAULT_ASTRA_MODEL = "gpt-6.2-astra"
DEFAULT_MAGPIE_ASTRA_MODEL = "group/magpie-gpt-6-astra"
SOCKS_PROXY_KEYS = ("all_proxy", "ALL_PROXY")
HTTP_PROXY_KEYS = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY")
SANDBOX_EXEC_ERROR_RE = re.compile(
    r"CreateProcessWithLogonW failed:\s*1385|Failed to create unified exec process",
    re.IGNORECASE,
)

READONLY_ENFORCEMENT_NOTE = (
    "\n- **STRICT READ-ONLY CONTRACT**: Do NOT create, edit, or delete any files or modify git state "
    "in the working tree or mounted `--add-dir` directories. Use shell/Python strictly for read-only "
    "source inspection and non-mutating verification."
)

ASTRA_REMINDERS: dict[str, str] = {
    "tech-closure": """

---
### Collaboration Notes (Technical Closure)
- Treat the product goals, usability, interaction surface, and macro system logic above as our agreed direction; focus your depth on closing the technical implementation cleanly.
- Inspect the real local repository, uncommitted worktree changes, local JSON/data artifacts, and mounted `--add-dir` reference sources thoroughly so nothing relies on guesswork or stale remote `main`.
- Favor structural unification, conservation laws, and deleting legacy/fallback baggage (including splitting oversized `> 800`-line legacy modules cleanly when they own a root cause) over heuristic scalar nudges or imaginary-enemy defensive checks.
""",
    "deep-review": """

---
### Collaboration Notes (Deep Systemic Review & Detailed Root-Cause Fix Blueprint)
- Take full time to trace real production entrypoints, call graphs, local worktree diffs, verification splits (`train` vs `heldout`, `L1`/`L2`/`L3` independence), and mounted `--add-dir` upstream sources end-to-end.
- **Do NOT compress fixes into 1–2 terse sentences tacked onto individual review bullets.** Unless `VERDICT: PASS`, structure your output into two explicit parts:
  1. `## Part 1 — Verified Findings & Counterexamples`: List each genuine `[P0 | P1 | SYSTEMIC | DOC-FACT] <file:line>` with the concrete reproduced failure mechanism / counterexample and tag which Root-Cause Group (`RC-1`, `RC-2`, ...) owns it.
  2. `## Part 2 — Detailed Systemic Fix Blueprint (By Root-Cause Group)`: Cluster related symptoms into a small number of unified structural designs (`RC-1`, `RC-2`, ...). For each group, specify the **complete, implementation-ready systemic solution**—unified data model / state invariants, exact algorithm & control-flow steps, function/module boundaries, what ad-hoc or legacy branches to delete, and resource/line-budget impact—so downstream engineers implement cohesive architecture rather than piecemeal `if` or constant band-aids.
- Skip stylistic nitpicks and imaginary-enemy defensive programming (e.g., internal `isinstance` spam, "just-in-case" fallback shims, or single-user local ACL/symlink paranoia).
""",
    "author": """

---
### Collaboration Notes (Direct Math / Core Authoring)
- Edit only the in-scope code/test files in the working tree (do not run `git commit`/`git push` or write user-facing `README`/`docs/` prose).
- Keep equations, invariants, and real-entrypoint wiring exact, structurally general (not hardcoded to single case/model IDs), and free of silent fallbacks.
""",
    "consult": "",
}

CODER_REMINDER = """

---
### Collaboration Notes (Scoped Worktree Slice & Anti-Spin Circuit Breaker)
- **Wave-scoped slice**: Your slice in this wave covers **code, configs, and tests only**—a dedicated `joint-docs-writer` agent updates `README*`/`CONTRIBUTING*`/`docs/**` once code and metrics stabilize. Keep doc files untouched unless explicitly listed in your prompt above, and note any user-facing contract deltas in your summary for the docs pass.
- Wire changes through the real production entrypoint, delete superseded legacy/silent-fallback branches cleanly, and keep logic structurally general rather than hardcoded to specific case IDs or magic-constant bypasses.
- **Anti-spin circuit breaker**: If a test or accuracy bound fails to converge after **2 principled attempts** due to a structural mismatch in an out-of-slice or legacy module, **stop tuning constants immediately**, leave clean state, and report the exact `file:line` + counterexample in your summary.
- Leave `git commit`/`git push` to the Coordinator. When using `ssh`/`scp`/`rsync` for remote server verification, pass `-o ControlMaster=no -o BatchMode=yes` so background muxes never hang stdout.
"""


def _subprocess_kwargs() -> dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)} if os.name == "nt" else {}


def _is_executable_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        return path.suffix.lower() in {".exe", ".cmd", ".bat", ".com"} if os.name == "nt" else os.access(path, os.X_OK)
    except TimeoutError:
        raise
    except OSError:
        return False


def _windows_candidate_dirs() -> list[str]:
    dirs: list[str] = []
    local_app = os.environ.get("LOCALAPPDATA", "")
    roaming_app = os.environ.get("APPDATA", "")
    if local_app:
        codex_bin_root = Path(local_app) / "OpenAI" / "Codex" / "bin"
        if codex_bin_root.is_dir():
            try:
                subdirs = sorted((p for p in codex_bin_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
                dirs.extend(str(p) for p in subdirs)
            except TimeoutError:
                raise
            except OSError:
                pass
        dirs.append(str(Path(local_app) / "Programs" / "DeepSeek Harness" / "resources" / "runtime" / "cli" / "bin"))
        dirs.append(str(Path(local_app) / "Programs" / "grok" / "bin"))
    if roaming_app:
        dirs.append(str(Path(roaming_app) / "npm"))
    dirs.extend([str(Path("~/.grok/bin").expanduser()), str(Path("~/.local/bin").expanduser())])
    return dirs


def _augmented_path(base_path: str | None = None, *, is_windows: bool | None = None) -> str:
    win = (os.name == "nt") if is_windows is None else is_windows
    sep = ";" if win else ":"
    prefix = (
        _windows_candidate_dirs()
        if win
        else ["/opt/homebrew/bin", "/usr/local/bin", str(Path("~/.grok/bin").expanduser()), str(Path("~/.local/bin").expanduser())]
    )
    raw_existing = (base_path if base_path is not None else os.environ.get("PATH", "")).split(sep)
    seen: set[str] = set()
    merged: list[str] = []
    for entry in [*prefix, *raw_existing]:
        cleaned = entry.strip()
        if not cleaned:
            continue
        norm = cleaned.lower() if win else cleaned
        if norm not in seen:
            seen.add(norm)
            merged.append(cleaned)
    return sep.join(merged)


def _read_codex_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_file = config_path or Path("~/.codex/config.toml").expanduser()
    if not cfg_file.is_file():
        return {}
    raw = ""
    try:
        raw = cfg_file.read_text(encoding="utf-8", errors="replace")
        return tomllib.loads(raw)
    except TimeoutError:
        raise
    except (OSError, ValueError):
        fallback: dict[str, Any] = {}
        for ln in raw.splitlines():
            m = re.match(r'^\s*(model|sandbox_mode)\s*=\s*"([^"]+)"', ln)
            if m:
                fallback[m.group(1)] = m.group(2)
        return fallback


def _find_bin(name: str, extra_candidates: tuple[str, ...] = (), *, search_path: str | None = None) -> str | None:
    eff_path = search_path if search_path is not None else _augmented_path()
    for cand in extra_candidates:
        if cand:
            p = Path(os.path.expandvars(cand)).expanduser()
            if _is_executable_file(p):
                return str(p)
    return shutil.which(name, path=eff_path)


def _resolve_bins(codex_cfg: dict[str, Any] | None = None) -> dict[str, str | None]:
    cfg = codex_cfg if codex_cfg is not None else _read_codex_config()
    eff_path = _augmented_path()
    cfg_codex_cli = str(((cfg.get("mcp_servers") or {}).get("node_repl") or {}).get("env", {}).get("CODEX_CLI_PATH", ""))

    if os.name == "nt":
        local_app = os.environ.get("LOCALAPPDATA", "")
        roaming_app = os.environ.get("APPDATA", "")
        win_codex_exes: list[str] = [cfg_codex_cli] if cfg_codex_cli else []
        if local_app:
            codex_root = Path(local_app) / "OpenAI" / "Codex" / "bin"
            if codex_root.is_dir():
                try:
                    exes = sorted(codex_root.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
                    win_codex_exes.extend(str(p) for p in exes)
                except TimeoutError:
                    raise
                except OSError:
                    pass
        if roaming_app:
            win_codex_exes.append(str(Path(roaming_app) / "npm" / "codex.cmd"))
        codex_cands = tuple(win_codex_exes)
        dsh_cands = (
            str(Path(local_app) / "Programs" / "DeepSeek Harness" / "resources" / "runtime" / "cli" / "bin" / "dsh.cmd") if local_app else "",
            str(Path("~/.local/bin/dsh.cmd").expanduser()),
            str(Path("~/.local/bin/dsh.exe").expanduser()),
        )
        grok_cands = (
            str(Path("~/.grok/bin/grok.exe").expanduser()),
            str(Path("~/.grok/bin/grok.cmd").expanduser()),
            str(Path(roaming_app) / "npm" / "grok.cmd") if roaming_app else "",
        )
    else:
        codex_cands = (
            "/opt/homebrew/bin/codex",
            "/usr/local/bin/codex",
            "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex",
        )
        dsh_cands = ("/opt/homebrew/bin/dsh", "~/.local/bin/dsh")
        grok_cands = ("~/.grok/bin/grok", "/opt/homebrew/bin/grok")

    return {
        "codex": _find_bin("codex", codex_cands, search_path=eff_path),
        "dsh": _find_bin("dsh", dsh_cands, search_path=eff_path),
        "grok": _find_bin("grok", grok_cands, search_path=eff_path),
    }


def _discover_astra_models(codex_dir: Path | None = None) -> dict[str, Any]:
    root = codex_dir or Path("~/.codex").expanduser()
    direct_best = ((6, 2), DEFAULT_ASTRA_MODEL)
    magpie_best: tuple[tuple[int, int], str] | None = None
    available: list[str] = []

    for fname in ("models_cache.json", "magpie-desktop-models.json"):
        fpath = root / fname
        if not fpath.is_file():
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8", errors="replace"))
            models = data.get("models", data) if isinstance(data, dict) else data
            for item in models if isinstance(models, list) else []:
                slug = str(item.get("slug", "")) if isinstance(item, dict) else ""
                if not slug or "astra" not in slug.lower():
                    continue
                if slug not in available:
                    available.append(slug)
                m_direct = re.fullmatch(r"gpt-(\d+)(?:\.(\d+))?-astra", slug)
                if m_direct:
                    ver = (int(m_direct.group(1)), int(m_direct.group(2) or 0))
                    if ver >= direct_best[0]:
                        direct_best = (ver, slug)
                m_magpie = re.fullmatch(r"group/magpie-gpt-(\d+)(?:\.(\d+))?-astra", slug)
                if m_magpie:
                    ver = (int(m_magpie.group(1)), int(m_magpie.group(2) or 0))
                    if magpie_best is None or ver >= magpie_best[0]:
                        magpie_best = (ver, slug)
        except TimeoutError:
            raise
        except (OSError, ValueError):
            pass

    return {
        "default_direct": direct_best[1],
        "default_magpie": magpie_best[1] if magpie_best else DEFAULT_MAGPIE_ASTRA_MODEL,
        "available": available,
    }


def _resolve_astra_model(requested: str | None, codex_dir: Path | None = None) -> str:
    discovered = _discover_astra_models(codex_dir)
    if not requested:
        return str(discovered["default_direct"])
    return str(discovered["default_magpie"]) if requested.strip().lower() == "magpie" else requested.strip()


def _resolve_astra_sandbox(
    *, writable: bool, explicit_sandbox: str | None, codex_cfg: dict[str, Any], is_windows: bool | None = None
) -> tuple[str, bool]:
    """Select Codex sandbox mode; avoid Windows Win32 1385 child-exec rejection via git-audited danger-full-access."""
    if explicit_sandbox:
        return explicit_sandbox, (not writable and explicit_sandbox == "danger-full-access")
    win = (os.name == "nt") if is_windows is None else is_windows
    if win:
        return "danger-full-access", (not writable)
    if str(codex_cfg.get("sandbox_mode", "")) == "danger-full-access" and writable:
        return "danger-full-access", False
    return ("workspace-write" if writable else "read-only"), False


def _clean_env(*, keep_proxy: bool = False, codex_cfg: dict[str, Any] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = _augmented_path(env.get("PATH", ""))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["NO_PROXY"] = "127.0.0.1,localhost,::1"
    env["no_proxy"] = "127.0.0.1,localhost,::1"

    cfg_env_set = ((codex_cfg or {}).get("shell_environment_policy") or {}).get("set")
    if isinstance(cfg_env_set, dict):
        for k, v in cfg_env_set.items():
            if isinstance(k, str) and isinstance(v, str) and k not in ("NO_PROXY", "no_proxy"):
                env.setdefault(k, v)

    if not keep_proxy:
        for k in SOCKS_PROXY_KEYS:
            env.pop(k, None)
        if not isinstance(cfg_env_set, dict):
            for k in HTTP_PROXY_KEYS:
                env.pop(k, None)
    return env


def _linked_worktree_commondirs(paths: list[Path]) -> list[str]:
    """Detect linked Git worktrees and return parent `.git` commondirs needed for `--add-dir`."""
    extras: list[str] = []
    seen: set[str] = set()
    for base in paths:
        dot_git = base / ".git"
        try:
            if not dot_git.is_file():
                continue
            lines = dot_git.read_text(encoding="utf-8", errors="replace").splitlines()
            if not lines or not lines[0].startswith("gitdir:"):
                continue
            raw_gitdir = lines[0].split("gitdir:", 1)[1].strip()
            gitdir = (base / raw_gitdir).resolve() if not Path(raw_gitdir).is_absolute() else Path(raw_gitdir).resolve()
            commondir_file = gitdir / "commondir"
            common_git = (gitdir / commondir_file.read_text(encoding="utf-8", errors="replace").strip()).resolve() if commondir_file.is_file() else gitdir
            for candidate in (common_git, common_git.parent if common_git.name == ".git" else common_git):
                s = str(candidate)
                key = s.lower() if os.name == "nt" else s
                if candidate.exists() and key not in seen:
                    seen.add(key)
                    extras.append(s)
        except TimeoutError:
            raise
        except OSError:
            continue
    return extras


def _merge_add_dirs(cwd: Path, explicit_add_dirs: list[str]) -> list[str]:
    resolved_explicit = [Path(p).expanduser().resolve() for p in explicit_add_dirs]
    auto_dirs = _linked_worktree_commondirs([cwd, *resolved_explicit])
    merged: list[str] = []
    seen: set[str] = {str(cwd).lower() if os.name == "nt" else str(cwd)}
    for item in [*(str(p) for p in resolved_explicit), *auto_dirs]:
        key = item.lower() if os.name == "nt" else item
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
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
            **_subprocess_kwargs(),
        )
        return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()] if res.returncode == 0 else []
    except (TimeoutError, subprocess.TimeoutExpired):
        return ["<git status timed out>"]
    except OSError:
        return []


def _count_sandbox_exec_errors(stderr_path: Path) -> int:
    if not stderr_path.is_file():
        return 0
    try:
        with stderr_path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for ln in fh if SANDBOX_EXEC_ERROR_RE.search(ln))
    except TimeoutError:
        raise
    except OSError:
        return 0


def _extract_last_report_from_jsonl(events_path: Path) -> str:
    if not events_path.is_file():
        return ""
    last_text = ""
    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line.startswith("{"):
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
                    joined = "\n".join(blk.get("text", "") for blk in msg["content"] if isinstance(blk, dict) and blk.get("type") == "text").strip()
                    if len(joined) > 20:
                        last_text = joined
    except TimeoutError:
        raise
    except OSError:
        return ""
    return last_text


def cmd_doctor(args: argparse.Namespace) -> int:
    codex_cfg = _read_codex_config()
    bins = _resolve_bins(codex_cfg)
    cwd = Path(args.cwd).expanduser().resolve()
    models_info = _discover_astra_models()
    eff_sandbox_ro, ro_audit = _resolve_astra_sandbox(writable=False, explicit_sandbox=None, codex_cfg=codex_cfg)
    report = {
        "platform": sys.platform,
        "python_executable": sys.executable,
        "bins": bins,
        "astra_pinned_model": models_info["default_direct"],
        "astra_magpie_model": models_info["default_magpie"],
        "astra_available_models": models_info["available"],
        "astra_default_effort": "max",
        "astra_effective_readonly_sandbox": eff_sandbox_ro,
        "astra_readonly_git_audit": ro_audit,
        "codex_config_default_model": codex_cfg.get("model"),
        "codex_config_sandbox_mode": codex_cfg.get("sandbox_mode"),
        "dsh_headless_profile_exists": Path("~/.dsh/profiles/headless").expanduser().is_dir(),
        "grok_config_exists": Path("~/.grok/config.toml").expanduser().is_file(),
        "cwd": str(cwd),
        "cwd_linked_worktree_add_dirs": _linked_worktree_commondirs([cwd]),
        "cwd_git_changes": len(_git_changed_files(cwd)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(not bins.get(k) for k in ("codex", "dsh")) else 0


def cmd_astra(args: argparse.Namespace) -> int:
    codex_cfg = _read_codex_config()
    codex_bin = _resolve_bins(codex_cfg)["codex"]
    if not codex_bin:
        sys.stderr.write("ERROR: `codex` CLI not found on PATH or standard install locations.\n")
        return 2

    cwd = Path(args.cwd).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file = out_file.with_suffix(out_file.suffix + ".stderr.log")
    meta_file = out_file.with_suffix(out_file.suffix + ".meta.json")

    model = _resolve_astra_model(args.model)
    writable = bool(args.write or args.mode == "author")
    sandbox, audit_readonly = _resolve_astra_sandbox(writable=writable, explicit_sandbox=args.sandbox, codex_cfg=codex_cfg)
    add_dirs = _merge_add_dirs(cwd, args.add_dir)

    suffix = "" if args.raw_prompt else ASTRA_REMINDERS.get(args.mode, "")
    if audit_readonly and not args.raw_prompt:
        suffix += READONLY_ENFORCEMENT_NOTE
    prompt_text = prompt_file.read_text(encoding="utf-8").rstrip() + suffix + "\n"

    cmd: list[str] = [
        codex_bin, "exec", "--ephemeral", "-s", sandbox, "--skip-git-repo-check",
        "-m", model,
        "-c", f'model_reasoning_effort="{args.effort}"',
        "-c", "notify=[]", "-c", "plugins={}", "-c", "mcp_servers={}",
        "-c", "features.responses_websockets=false", "-c", "features.responses_websockets_v2=false",
        "-C", str(cwd), "-o", str(out_file),
    ]
    for extra in add_dirs:
        cmd.extend(["--add-dir", extra])
    cmd.append("-")

    pre_changed = set(_git_changed_files(cwd)) if audit_readonly else set()
    t0 = time.time()
    base_meta = {
        "kind": "astra", "mode": args.mode, "sandbox": sandbox, "readonly_audited": audit_readonly,
        "model": model, "effort": args.effort, "cwd": str(cwd), "add_dirs": add_dirs,
        "out_file": str(out_file), "stderr_file": str(stderr_file),
    }
    _write_json_atomic(meta_file, {**base_meta, "state": "running", "started_at": t0})

    state, rc = "failed", 1
    effective_timeout = max(args.timeout, 1800) if args.effort in ("max", "ultra") and not args.allow_short_timeout else args.timeout
    try:
        with stderr_file.open("w", encoding="utf-8") as err_fh:
            for extra in args.add_dir:
                ep = Path(extra).expanduser().resolve()
                if ep.is_dir() and not (ep / ".git").exists() and sum(1 for _ in ep.iterdir()) > 40:
                    err_fh.write(f"WARNING: --add-dir {ep} is a large non-repo parent folder; prefer specific sibling repos.\n")
                    err_fh.flush()
            proc = subprocess.run(
                cmd, input=prompt_text, text=True, stdout=subprocess.DEVNULL, stderr=err_fh,
                cwd=str(cwd), env=_clean_env(keep_proxy=args.keep_proxy, codex_cfg=codex_cfg),
                timeout=effective_timeout, check=False, **_subprocess_kwargs(),
            )
            rc = proc.returncode
            state = "completed" if (rc == 0 and out_file.is_file() and out_file.stat().st_size > 0) else "failed"
    except (TimeoutError, subprocess.TimeoutExpired):
        state, rc = "timed_out", 124
    finally:
        elapsed = round(time.time() - t0, 2)
        out_bytes = out_file.stat().st_size if out_file.is_file() else 0
        post_changed = _git_changed_files(cwd) if (writable or audit_readonly) else []
        unexpected_writes = sorted(set(post_changed) - pre_changed) if audit_readonly else []
        sandbox_exec_errors = _count_sandbox_exec_errors(stderr_file)
        warning: str | None = None
        if sandbox_exec_errors > 0:
            warning = (
                f"Detected {sandbox_exec_errors} sandbox child-exec rejection(s) (e.g. Win32 1385) in stderr; "
                "Astra could not read local files. Re-run with --sandbox danger-full-access."
            )
            if state == "completed":
                state = "degraded_blind_sandbox"
        elif unexpected_writes:
            warning = f"Read-only Astra run modified {len(unexpected_writes)} worktree path(s): {unexpected_writes[:5]}"

        _write_json_atomic(
            meta_file,
            {
                **base_meta, "state": state, "returncode": rc, "elapsed_s": elapsed, "out_bytes": out_bytes,
                "sandbox_exec_errors": sandbox_exec_errors,
                "changed_files_count": len(post_changed) if writable else len(unexpected_writes),
                "unexpected_changed_files": unexpected_writes, "warning": warning,
            },
        )

    summary_payload: dict[str, Any] = {
        "state": state, "model": model, "sandbox": sandbox, "returncode": rc,
        "elapsed_s": elapsed, "out_file": str(out_file), "out_bytes": out_bytes,
        "sandbox_exec_errors": sandbox_exec_errors,
    }
    if unexpected_writes:
        summary_payload["unexpected_changed_files"] = unexpected_writes
    if warning:
        summary_payload["warning"] = warning
    print(json.dumps(summary_payload, ensure_ascii=False))
    return 0 if state == "completed" else (rc or 1)


def _build_code_command(
    *, engine: str, engine_bin: str, cwd: Path, effective_prompt_path: Path,
    effective_prompt: str, add_dirs: list[str], args: argparse.Namespace,
) -> tuple[list[str], str | None]:
    if engine == "dsh":
        cmd = [engine_bin, "--profile", args.dsh_profile, "--approval-policy", "never"]
        if args.dsh_permission_mode:
            cmd.extend(["--permission-mode", args.dsh_permission_mode])
        for extra in add_dirs:
            cmd.extend(["--add-dir", extra])
        cmd.extend(["--json", "-"])
        return cmd, effective_prompt

    cmd = [
        engine_bin, "--prompt-file", str(effective_prompt_path), "--cwd", str(cwd),
        "--always-approve", "--no-plan", "--no-subagents", "--output-format", "streaming-json",
    ]
    if args.model:
        cmd.extend(["-m", args.model])
    if args.effort:
        cmd.extend(["--effort", args.effort])
    for extra in add_dirs:
        cmd.extend(["--add-dir", extra])
    return cmd, None


def cmd_code(args: argparse.Namespace) -> int:
    codex_cfg = _read_codex_config()
    engine: str = args.engine
    engine_bin = _resolve_bins(codex_cfg).get(engine)
    if not engine_bin:
        sys.stderr.write(f"ERROR: `{engine}` CLI not found on PATH or standard install locations.\n")
        return 2

    cwd = Path(args.cwd).expanduser().resolve()
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_prompt_path = run_dir / "effective_prompt.md"
    events_path, stderr_path = run_dir / "events.jsonl", run_dir / "stderr.log"
    report_path, meta_path = run_dir / "report.md", run_dir / "meta.json"

    effective_prompt = prompt_file.read_text(encoding="utf-8").rstrip() + ("" if args.raw_prompt else CODER_REMINDER) + "\n"
    effective_prompt_path.write_text(effective_prompt, encoding="utf-8")

    add_dirs = _merge_add_dirs(cwd, args.add_dir)
    cmd, stdin_text = _build_code_command(
        engine=engine, engine_bin=engine_bin, cwd=cwd,
        effective_prompt_path=effective_prompt_path, effective_prompt=effective_prompt,
        add_dirs=add_dirs, args=args,
    )

    t0 = time.time()
    base_meta = {
        "kind": "code", "engine": engine, "cwd": str(cwd), "add_dirs": add_dirs,
        "run_dir": str(run_dir), "events_file": str(events_path), "stderr_file": str(stderr_path),
    }
    _write_json_atomic(meta_path, {**base_meta, "state": "running", "started_at": t0})

    state, rc = "failed", 1
    try:
        with events_path.open("w", encoding="utf-8") as out_fh, stderr_path.open("w", encoding="utf-8") as err_fh:
            proc = subprocess.run(
                cmd, input=stdin_text, text=True, stdout=out_fh, stderr=err_fh,
                cwd=str(cwd), env=_clean_env(keep_proxy=args.keep_proxy, codex_cfg=codex_cfg),
                timeout=args.timeout, check=False, **_subprocess_kwargs(),
            )
            rc = proc.returncode
            state = "completed" if rc == 0 else "failed"
    except (TimeoutError, subprocess.TimeoutExpired):
        state, rc = "timed_out", 124
    finally:
        elapsed = round(time.time() - t0, 2)
        report_text = _extract_last_report_from_jsonl(events_path)
        if report_text:
            report_path.write_text(report_text + "\n", encoding="utf-8")
        changed = _git_changed_files(cwd)
        event_lines = 0
        if events_path.is_file():
            with events_path.open("r", encoding="utf-8", errors="replace") as ev_fh:
                event_lines = sum(1 for _ in ev_fh)
        _write_json_atomic(
            meta_path,
            {
                **base_meta, "report_file": str(report_path) if report_path.is_file() else None,
                "state": state, "returncode": rc, "elapsed_s": elapsed, "event_lines": event_lines,
                "changed_files_count": len(changed), "changed_files_sample": changed[:30],
            },
        )

    print(
        json.dumps(
            {
                "engine": engine, "state": state, "returncode": rc, "elapsed_s": elapsed,
                "event_lines": event_lines, "changed_files_count": len(changed),
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
    print(json.dumps({"count": len(metas), "runs": [_summarize_meta(m) for m in metas]}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collaboration dispatcher for /joint-dev (Astra Max + DSH/Grok)")
    sub = ap.add_subparsers(dest="subcmd", required=True)

    p_doc = sub.add_parser("doctor", help="Preflight codex/astra, dsh, grok, sandbox & worktree readiness")
    p_doc.add_argument("--cwd", default=".")
    p_doc.set_defaults(func=cmd_doctor)

    p_astra = sub.add_parser("astra", help="Run Codex Astra Max (read-only technical closure/review, or --write math authoring)")
    p_astra.add_argument("--mode", choices=("tech-closure", "deep-review", "author", "consult"), default="consult")
    p_astra.add_argument("--write", action="store_true", help="Allow workspace edits for math/formal core authoring")
    p_astra.add_argument("--raw-prompt", action="store_true", help="Send prompt verbatim without appending role notes")
    p_astra.add_argument("--cwd", default=".")
    p_astra.add_argument("--add-dir", action="append", default=[])
    p_astra.add_argument("--prompt-file", required=True)
    p_astra.add_argument("--out", required=True)
    p_astra.add_argument("--model", default=None, help="Model slug or 'magpie' shorthand (default: newest gpt-*-astra)")
    p_astra.add_argument("--effort", default="max", choices=("high", "xhigh", "max", "ultra"))
    p_astra.add_argument("--sandbox", choices=("read-only", "workspace-write", "danger-full-access"), default=None)
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
    p_code.add_argument("--dsh-permission-mode", default=None, help="Optional DSH permission preset override")
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
