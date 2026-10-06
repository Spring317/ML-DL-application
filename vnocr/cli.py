"""Command line: python -m vnocr INPUT [-o OUTPUT.md] [--pages 1-3,5] [--cpu]"""

import argparse
import json
import sys
import time
from pathlib import Path

from .engine import OcrEngine, default_model_dir
from .gui import parse_pages
from .logutil import make_logger


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vnocr", description="OCR a Vietnamese PDF or image.")
    ap.add_argument("input", nargs="?", help="PDF or image; without it the desktop window opens")
    ap.add_argument("-o", "--output", help="Write Markdown (.md), plain text (.txt) or JSON (.json)")
    ap.add_argument("--pages", default="", help="e.g. 1-3,5 (PDF only)")
    ap.add_argument("--cpu", action="store_true", help="Do not use the NPU")
    ap.add_argument("--models", help="Model folder (default: ./models next to the app)")
    args = ap.parse_args(argv)
    if not args.input:
        from .gui import main as gui_main
        gui_main()
        return

    log = make_logger()
    engine = OcrEngine(args.models or default_model_dir(), use_npu=not args.cpu, log=log)
    t0 = time.perf_counter()
    results = engine.ocr_document(args.input, pages=parse_pages(args.pages),
                                  progress=lambda r: print(f"page {r.page}: {r.lines} lines, {r.tables} tables, "
                                                           f"{r.seconds['total']:.1f}s", file=sys.stderr))
    elapsed = time.perf_counter() - t0
    log(f"{len(results)} page(s) in {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s per page)")

    out = Path(args.output) if args.output else None
    if out and out.suffix.lower() == ".json":
        data = {"placement": engine.placement, "pages": [r.__dict__ for r in results]}
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    markdown = not (out and out.suffix.lower() == ".txt")
    doc = "\n".join(f"## Page {r.page}\n\n{r.markdown if markdown else r.text}\n" for r in results)
    if out:
        out.write_text(doc, encoding="utf-8")
    else:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding="utf-8")
            print(doc)


if __name__ == "__main__":
    main()
