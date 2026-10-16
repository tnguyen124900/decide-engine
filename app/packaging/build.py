"""Build the DECIDE desktop app with PyInstaller (run on the target OS).

    python packaging/build.py

Output: dist/"DECIDE" (folder) or dist/"DECIDE.app" on macOS.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the app/ folder
REPO = ROOT.parent  # repository root (engine in src/)
PKG = ROOT / "packaging"
BUILD = ROOT / "build_desktop"
PAYLOAD = BUILD / "app_payload"

EXCLUDE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.bak*", ".git", ".venv", "tests", "_to_delete",
    "_export_*", ".claude_scratch", "secrets.toml", "build_desktop", "dist", "packaging", "*.egg-info", ".gitignore", ".pytest_cache",
)
DEMO_MODELS = ["tree", "markov", "psm", "microsim", "nma", "manual"]
# Collect all data/submodules for packages that load files dynamically.
COLLECT_ALL = [
    "streamlit", "streamlit_flow", "streamlit_pdf_viewer", "altair", "vl_convert",
    "reportlab", "openpyxl", "webview",
]
COPY_META = ["streamlit", "pandas", "numpy", "scipy", "altair", "openpyxl", "reportlab"]


def stage_payload() -> None:
    """app/ plus the engine package (../src), its priors and the sample models."""
    if BUILD.exists():
        shutil.rmtree(BUILD)
    shutil.copytree(ROOT, PAYLOAD, ignore=EXCLUDE)
    for name in ("src", "priors", "samples"):
        shutil.copytree(REPO / name, PAYLOAD / name, ignore=EXCLUDE)
    # The budget-impact modules of the engine are not part of the released app.
    for module in (PAYLOAD / "src" / "heor_tree_engine").glob("bia*.py"):
        module.unlink()
    for name in ("models", "autosave"):  # ship empty user folders
        shutil.rmtree(PAYLOAD / name, ignore_errors=True)
        (PAYLOAD / name).mkdir()
    # The released app shows only these model types (uses the app's staging filter).
    (PAYLOAD / "STAGING_MODELS").write_text("\n".join(DEMO_MODELS) + "\n", encoding="utf-8")


def imported_modules() -> list[str]:
    local = {p.stem for p in PAYLOAD.glob("*.py")} | {p.name for p in PAYLOAD.iterdir() if p.is_dir()}
    local |= {p.name for p in (PAYLOAD / "src").iterdir()} if (PAYLOAD / "src").exists() else set()
    found: set[str] = set()
    for path in PAYLOAD.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return sorted(m for m in found if m not in local)


def main() -> None:
    stage_payload()
    hidden = imported_modules()
    sep = ";" if sys.platform.startswith("win") else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--name", "DECIDE", "--windowed",
        "--distpath", str(ROOT / "dist"), "--workpath", str(BUILD / "work"),
        "--specpath", str(BUILD),
        "--add-data", f"{PAYLOAD}{sep}app_payload",
    ]
    icon = PKG / ("DECIDE.icns" if sys.platform == "darwin" else "DECIDE.ico")
    if icon.exists():
        cmd += ["--icon", str(icon)]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for pkg in COPY_META:
        cmd += ["--copy-metadata", pkg]
    for mod in hidden:
        cmd += ["--hidden-import", mod]
    cmd.append(str(PKG / "launcher.py"))
    print("hidden imports:", len(hidden))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
