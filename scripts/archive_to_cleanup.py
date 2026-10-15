"""Move untracked files out of the repository into the cleanup archive, with a manifest.

    python scripts/archive_to_cleanup.py [--no-hash] PATH [PATH ...]

Each PATH (file or directory, relative to the repository root) is moved to the
same relative path under ``$PEARLSIM_CLEANUP`` (default
``/global/scratch/users/ikolaja/cleanup/PEARLSim_v1``). Before moving, one row
per file is appended to ``MANIFEST.tsv`` there: path, size, sha256 (files up to
1 GB; larger files record ``skipped``), mtime, and the archive date. After
moving, the hashes are verified. ``--no-hash`` records size and mtime only, for
large data directories: the move is a same-filesystem rename, which cannot
change file contents. Tracked files are refused: they are
recoverable from git (tag ``v1-as-run``) and are removed with ``git rm``.
See docs/REMOVED.md.
"""

import datetime
import hashlib
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST = os.environ.get("PEARLSIM_CLEANUP", "/global/scratch/users/ikolaja/cleanup/PEARLSim_v1")
HASH_LIMIT = 1 << 30


def sha256(path, enabled=True):
    if not enabled or os.path.getsize(path) > HASH_LIMIT:
        return "skipped"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(path):
    if os.path.isfile(path):
        yield path
        return
    for root, _, names in os.walk(path):
        for name in sorted(names):
            yield os.path.join(root, name)


def is_tracked(rel):
    result = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", rel],
                            capture_output=True, text=True)
    return result.returncode == 0


def main(paths, hashing=True):
    os.makedirs(DEST, exist_ok=True)
    if os.stat(REPO).st_dev != os.stat(DEST).st_dev:
        print("warning: archive is on a different filesystem; mv will copy data", file=sys.stderr)
    manifest = os.path.join(DEST, "MANIFEST.tsv")
    new_manifest = not os.path.exists(manifest)
    today = datetime.date.today().isoformat()
    with open(manifest, "a") as m:
        if new_manifest:
            m.write("path\tbytes\tsha256\tmtime\tarchived\n")
        for rel in paths:
            rel = os.path.normpath(rel)
            src = os.path.join(REPO, rel)
            if not os.path.exists(src):
                print(f"skip (missing): {rel}")
                continue
            if os.path.isfile(src) and is_tracked(rel):
                sys.exit(f"refusing to archive tracked file {rel}; use git rm")
            rows = []
            for f in files_under(src):
                st = os.stat(f)
                mtime = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
                rows.append((os.path.relpath(f, REPO), st.st_size, sha256(f, hashing), mtime))
            dst = os.path.join(DEST, rel)
            if os.path.exists(dst):
                sys.exit(f"destination exists, not overwriting: {dst}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            for path, size, digest, mtime in rows:
                moved = os.path.join(DEST, path)
                if digest != "skipped" and sha256(moved) != digest:
                    sys.exit(f"hash mismatch after move: {path}")
                m.write(f"{path}\t{size}\t{digest}\t{mtime}\t{today}\n")
            print(f"archived {rel} ({len(rows)} files)")


if __name__ == "__main__":
    args = sys.argv[1:]
    no_hash = "--no-hash" in args
    main([a for a in args if a != "--no-hash"], hashing=not no_hash)
