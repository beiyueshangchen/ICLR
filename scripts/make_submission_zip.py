"""Build the code-submission archive.

The archive holds the code, the tests and the Junyi subset the example runs on;
run artefacts and caches are skipped.  See ``docs/datasets.md`` for what the
dataset subset contains.

Usage::

    python scripts/make_submission_zip.py                # -> grace_submission.zip
    python scripts/make_submission_zip.py -o /tmp/sub.zip
"""

from __future__ import annotations

import argparse
import os
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Paths (relative to the repository root) that are never part of a submission.
EXCLUDED_PREFIXES = (
    ".chat/",
    ".pytest_cache/",
    "runs/",
    "__pycache__/",
    ".git/",
)
#: Files that are never part of the submission (kept here in case the optional
#: raw Junyi log is downloaded again).
EXCLUDED_NAMES = (
    "junyi_ProblemLog_original.csv",
    "skill_builder_data_corrected.csv",
)
EXCLUDED_SUFFIXES = (".pyc", ".pt", ".pth", ".ckpt")
EXCLUDED_EXTENSIONS = (".zip",)


def _is_excluded(relpath: str) -> bool:
    if any(relpath.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    if any(part == "__pycache__" for part in relpath.split("/")):
        return True
    if os.path.basename(relpath) in EXCLUDED_NAMES:
        return True
    return relpath.endswith(EXCLUDED_SUFFIXES + EXCLUDED_EXTENSIONS)


def iter_files(root: str = REPO_ROOT):
    """Yield every file that belongs in the submission, relative to ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = _rel(dirpath, root)
        dirnames[:] = [d for d in dirnames if not _is_excluded(f"{rel_dir}/{d}/".lstrip("./"))]
        for filename in filenames:
            relpath = _rel(os.path.join(dirpath, filename), root)
            if not _is_excluded(relpath):
                yield relpath


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="grace_submission.zip")
    parser.add_argument("--root", default=REPO_ROOT, help="repository root to package")
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    files = sorted(iter_files(args.root))

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relpath in files:
            archive.write(os.path.join(args.root, relpath), arcname=f"grace/{relpath}")

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"{output}: {len(files)} files, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
