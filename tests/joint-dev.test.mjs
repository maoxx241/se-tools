import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PLUGIN_DIR = path.join(REPO_ROOT, "plugins", "joint-dev");
const DISPATCH_SCRIPT = path.join(PLUGIN_DIR, "skills", "joint-dev", "scripts", "joint_dispatch.py");

function resolvePythonRunner() {
  for (const [bin, prefixArgs] of [
    ["python3", []],
    ["python", []],
    ["uv", ["run", "--no-project", "python"]],
  ]) {
    const probe = spawnSync(bin, [...prefixArgs, "-c", "import sys, tomllib; print(sys.version_info[:2])"], {
      encoding: "utf8",
      windowsHide: true,
    });
    if (probe.status === 0 && probe.stdout.includes("(3,")) {
      return { bin, prefixArgs };
    }
  }
  throw new Error("No Python 3.11+ interpreter found for joint_dispatch.py tests");
}

test("joint-dev plugin bundle manifest, LF endings, and frontmatter contracts", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(PLUGIN_DIR, "plugin.json"), "utf8"));
  assert.equal(manifest.name, "joint-dev");
  assert.equal(manifest.version, "2.1.0");
  assert.equal(manifest.logo, "assets/logo.svg");
  assert.ok(fs.existsSync(path.join(PLUGIN_DIR, manifest.logo)));
  assert.ok(Array.isArray(manifest.suggestedPrompts) && manifest.suggestedPrompts.length <= 3);

  const requiredFiles = [
    "plugin.json",
    "README.md",
    "skills/joint-dev/SKILL.md",
    "skills/joint-dev/references/playbooks-and-guardrails.md",
    "skills/joint-dev/scripts/joint_dispatch.py",
    "agents/joint-design-architect.md",
    "agents/joint-coder-runner.md",
    "agents/joint-review-gatekeeper.md",
    "agents/joint-docs-writer.md",
  ];
  for (const rel of requiredFiles) {
    const full = path.join(PLUGIN_DIR, rel);
    assert.ok(fs.existsSync(full), `Missing ${rel}`);
    const content = fs.readFileSync(full, "utf8");
    assert.ok(!content.includes("\r\n"), `CRLF detected in ${rel}`);
  }

  const dispatchLines = fs.readFileSync(DISPATCH_SCRIPT, "utf8").split("\n").length;
  assert.ok(dispatchLines < 800, `joint_dispatch.py should stay < 800 lines, got ${dispatchLines}`);
});

test("joint_dispatch.py cross-platform PATH, Windows 1385 sandbox guard, Magpie model, DSH args & worktree commondir", () => {
  const py = resolvePythonRunner();
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "joint-dev-test-"));
  try {
    const pyCheck = `
import argparse, json, sys
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("joint_dispatch", r"""${DISPATCH_SCRIPT}""")
jd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jd)

tmp = Path(r"""${tmpRoot}""")

# 1. Windows vs POSIX PATH separator integrity (never split or join Windows drive colons with ':')
win_path = jd._augmented_path(r"C:\\Tools\\bin;D:\\Python", is_windows=True)
assert ";" in win_path and r"C:\\Tools\\bin" in win_path, win_path
posix_path = jd._augmented_path("/usr/bin:/bin", is_windows=False)
assert "/opt/homebrew/bin:/usr/local/bin" in posix_path, posix_path

# 2. models_cache.json discovery + 'magpie' shorthand
codex_dir = tmp / ".codex"
codex_dir.mkdir(parents=True, exist_ok=True)
(codex_dir / "models_cache.json").write_text(json.dumps({
    "models": [
        {"slug": "gpt-6-astra"},
        {"slug": "gpt-6.2-astra"},
        {"slug": "group/magpie-gpt-6-astra"},
        {"slug": "group/magpie-gpt-6.2-astra"},
    ]
}), encoding="utf-8")
assert jd._resolve_astra_model(None, codex_dir) == "gpt-6.2-astra"
assert jd._resolve_astra_model("magpie", codex_dir) == "group/magpie-gpt-6.2-astra"

# 3. Windows Codex 1385 sandbox avoidance + read-only git audit vs POSIX read-only sandbox
sb_win_ro, audit_win_ro = jd._resolve_astra_sandbox(writable=False, explicit_sandbox=None, codex_cfg={}, is_windows=True)
assert (sb_win_ro, audit_win_ro) == ("danger-full-access", True)
sb_posix_ro, audit_posix_ro = jd._resolve_astra_sandbox(writable=False, explicit_sandbox=None, codex_cfg={}, is_windows=False)
assert (sb_posix_ro, audit_posix_ro) == ("read-only", False)
sb_posix_rw, audit_posix_rw = jd._resolve_astra_sandbox(writable=True, explicit_sandbox=None, codex_cfg={}, is_windows=False)
assert (sb_posix_rw, audit_posix_rw) == ("workspace-write", False)

# 4. Linked Git worktree commondir auto-detection
base_git = tmp / "repo" / ".git"
wt_gitdir = base_git / "worktrees" / "repo-feat"
wt_gitdir.mkdir(parents=True, exist_ok=True)
(wt_gitdir / "commondir").write_text("../..\\n", encoding="utf-8")
wt_dir = tmp / "worktrees" / "repo-feat"
wt_dir.mkdir(parents=True, exist_ok=True)
(wt_dir / ".git").write_text(f"gitdir: {wt_gitdir}\\n", encoding="utf-8")
extra_dirs = jd._linked_worktree_commondirs([wt_dir])
assert str(base_git.resolve()) in extra_dirs, extra_dirs

# 5. DSH headless command forwards --approval-policy never and --add-dir
ns = argparse.Namespace(dsh_profile="headless", dsh_permission_mode="workspace-write", model=None, effort=None)
cmd, stdin_txt = jd._build_code_command(
    engine="dsh",
    engine_bin="/mock/dsh",
    cwd=wt_dir,
    effective_prompt_path=tmp / "p.md",
    effective_prompt="hello",
    add_dirs=[str(base_git.resolve())],
    args=ns,
)
assert cmd == [
    "/mock/dsh",
    "--profile",
    "headless",
    "--approval-policy",
    "never",
    "--permission-mode",
    "workspace-write",
    "--add-dir",
    str(base_git.resolve()),
    "--json",
    "-",
], cmd
assert stdin_txt == "hello"

# 6. Stderr Win32 1385 blind-sandbox detector
err_log = tmp / "sample.stderr.log"
err_log.write_text("ok\\nFailed to create unified exec process: CreateProcessWithLogonW failed: 1385\\n", encoding="utf-8")
assert jd._count_sandbox_exec_errors(err_log) == 1
print("ALL_OK")
`;
    const res = spawnSync(py.bin, [...py.prefixArgs, "-c", pyCheck], {
      encoding: "utf8",
      env: {
        ...process.env,
        HOME: tmpRoot,
        USERPROFILE: tmpRoot,
        APPDATA: path.join(tmpRoot, "AppData", "Roaming"),
        LOCALAPPDATA: path.join(tmpRoot, "AppData", "Local"),
      },
      windowsHide: true,
    });
    assert.equal(res.status, 0, `Python test failed:\nSTDOUT: ${res.stdout}\nSTDERR: ${res.stderr}`);
    assert.match(res.stdout, /ALL_OK/);
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
});
