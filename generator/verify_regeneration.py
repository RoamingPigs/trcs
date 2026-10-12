"""Compare two TRCS provenance manifests by file name and SHA-256. Exit 1 on any difference."""
import json
import sys

a, b = (json.load(open(p))["files"] for p in sys.argv[1:3])
A = {f["filename"]: f["sha256"] for f in a}
B = {f["filename"]: f["sha256"] for f in b}
bad = sorted(k for k in A.keys() | B.keys() if A.get(k) != B.get(k))
print(f"{len(A)} files in the first manifest, {len(B)} in the second, {len(bad)} differ")
for k in bad[:20]:
    print("  differs:", k)
sys.exit(1 if bad else 0)
