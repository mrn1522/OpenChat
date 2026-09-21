#!/usr/bin/env python3
"""Generate stylized GitHub release notes ("patch notes") for a release tag.

Collects commits between the previous semver tag and the given tag and groups
them by conventional-commit-style prefix (feat:, fix:, ...) into emoji-headed
sections. Merged-PR numbers in subjects are linked to the repo's PR pages.

Usage:
    python tools/gen_release_notes.py v0.3.0 [--out patch-notes.md]

Requires a full git history (checkout with fetch-depth: 0) and, for PR links,
either the GITHUB_REPOSITORY env var (set by GitHub Actions) or a github.com
remote named "origin".
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
SUBJECT_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?:\([^)]*\))?!?:\s*(?P<summary>.+?)\s*(?:\(#(?P<pr>\d+)\))?$")

SECTIONS = [
    ("feat", "✨ New Features"),
    ("fix", "🐛 Bug Fixes"),
    ("perf", "⚡ Performance"),
    ("refactor", "🧹 Refactoring"),
    ("docs", "📚 Documentation"),
    ("ci", "🔧 CI & Tooling"),
    ("build", "🔧 CI & Tooling"),
    ("chore", "🔧 CI & Tooling"),
    ("test", "🔧 CI & Tooling"),
    ("style", "🔧 CI & Tooling"),
]
FALLBACK_SECTION = "📦 Other Changes"


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def repo_slug() -> str:
    if slug := os.environ.get("GITHUB_REPOSITORY"):
        return slug
    url = git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", url)
    if not match:
        sys.exit("Cannot determine repo slug; set GITHUB_REPOSITORY=owner/repo")
    return match.group(1)


def semver_tags() -> list[tuple[tuple[int, int, int], str]]:
    tags = []
    for tag in git("tag", "--list", "v*").splitlines():
        if m := TAG_RE.match(tag):
            tags.append(((int(m[1]), int(m[2]), int(m[3])), tag))
    tags.sort(reverse=True)
    return tags


def previous_tag(tag: str, tags: list[tuple[tuple[int, int, int], str]]) -> str | None:
    m = TAG_RE.match(tag)
    if not m:
        sys.exit(f"Tag {tag!r} is not a strict semver tag (vX.Y.Z)")
    current = (int(m[1]), int(m[2]), int(m[3]))
    lower = [t for v, t in tags if v < current]
    return lower[0] if lower else None


def render(tag: str, slug: str, prev: str | None) -> str:
    rng = f"{prev}..{tag}" if prev else tag
    subjects = git("log", rng, "--pretty=%s").splitlines()

    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for raw in subjects:
        subject = raw.strip()
        if not subject or subject.startswith(("Merge ", "release:")):
            continue
        section = FALLBACK_SECTION
        summary = subject
        pr = None
        if m := SUBJECT_RE.match(subject):
            kind = m["type"].lower()
            summary = m["summary"].strip()
            pr = m["pr"]
            for prefix, title in SECTIONS:
                if kind == prefix:
                    section = title
                    break
        summary = summary[0].upper() + summary[1:] if summary else summary
        if pr:
            summary += f" ([#{pr}](https://github.com/{slug}/pull/{pr}))"
        if section not in grouped:
            grouped[section] = []
            order.append(section)
        grouped[section].append(summary)

    lines = [f"## OpenChat {tag} — Patch Notes", ""]
    section_order = [t for _, t in SECTIONS]
    ordered = sorted(order, key=lambda s: section_order.index(s) if s in section_order else len(section_order))
    if not ordered:
        lines.append("Maintenance release — internal changes only.")
    for section in ordered:
        lines += [f"### {section}", ""]
        lines += [f"- {item}" for item in grouped[section]]
        lines.append("")
    if prev:
        lines += ["---", f"**Full changelog:** https://github.com/{slug}/compare/{prev}...{tag}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="release tag, e.g. v0.3.0")
    parser.add_argument("--out", help="write notes to this file instead of stdout")
    args = parser.parse_args()

    notes = render(args.tag, repo_slug(), previous_tag(args.tag, semver_tags()))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(notes)
    else:
        sys.stdout.write(notes)


if __name__ == "__main__":
    main()
