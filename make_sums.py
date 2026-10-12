#!/usr/bin/env python3
"""Write results/SHA256SUMS for every file under results/ (except itself). Run after analyze.py."""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
lines = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name != "SHA256SUMS":
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}")
(root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(lines)} files hashed into {root / 'SHA256SUMS'}")
