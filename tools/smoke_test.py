#!/usr/bin/env python3
"""Check an OCR result against the reference output of the sample page.

  python tools/smoke_test.py RESULT.json [--max-cer 0.01]

RESULT.json is what `VNOCR.exe samples/sample_page.png -o RESULT.json` (or `python -m vnocr ...`)
writes. The reference (samples/sample_page.expected.txt) was produced by the same engine on Linux;
a different CPU or the NPU (FP16) may change a few characters, so a small CER is allowed.
Exit code 1 when the CER is above the limit.
"""
import argparse
import json
import sys
from pathlib import Path


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--expected", default=str(Path(__file__).resolve().parents[1] / "samples/sample_page.expected.txt"))
    ap.add_argument("--max-cer", type=float, default=0.01)
    args = ap.parse_args()
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    got = " ".join(p["text"] for p in data["pages"])
    ref = Path(args.expected).read_text(encoding="utf-8").strip()
    cer = levenshtein(ref, got) / max(len(ref), 1)
    print(f"placement: {data.get('placement')}")
    print(f"characters: {len(ref)}  CER vs reference: {cer:.3%}  (limit {args.max_cer:.1%})")
    for p in data["pages"]:
        print(f"page {p['page']}: {p['lines']} lines, seconds {p['seconds']}")
    sys.exit(0 if cer <= args.max_cer else 1)


if __name__ == "__main__":
    main()
