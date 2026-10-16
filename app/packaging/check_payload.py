"""Fail the release build if private-only material ended up in the app payload.

    python packaging/check_payload.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAYLOAD = Path(__file__).resolve().parent.parent / "build_desktop" / "app_payload"
BAD_NAMES = re.compile(r"(^|/)(optional_editors|bia_templates)(/|$)|(^|/)((dtm_)?bia[a-z_]*|xl_eval|xlsx_patch)\.py$")
BAD_TEXT = re.compile(r"\bBIA\b|budget.impact|CADTH|PBAC|CDA-AMC|Lynda|gmail|DECIDE Lab|DECIDE-(?![a-z])|\bNICE\b(?! (DSU|TSD))", re.I)
OK_NICE = re.compile(r"NICE (DSU|TSD)", re.I)


def main() -> int:
    if not PAYLOAD.is_dir():
        print(f"payload not found: {PAYLOAD}")
        return 2
    problems: list[str] = []
    for path in sorted(PAYLOAD.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PAYLOAD).as_posix()
        if BAD_NAMES.search(rel):
            problems.append(f"file should not be bundled: {rel}")
            continue
        if path.suffix.lower() not in {".py", ".json", ".md", ".txt", ".toml", ".yml", ".yaml", ".html", ".js", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            cleaned = OK_NICE.sub("", line)
            if BAD_TEXT.search(cleaned):
                problems.append(f"{rel}:{line_no}: {line.strip()[:100]}")
                break
    for item in problems:
        print(item)
    print("payload check:", "FAILED" if problems else "clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
