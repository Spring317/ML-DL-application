"""Run the app engine on the 53 evaluation pages (same 312-DPI renders as the research runs)
and write predictions in the evaluate_all.py format.

  python app/tests/validate_on_benchmark.py --out reports/app-engine-cpu [--npu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from vnocr import cv_compat as cv2  # noqa: E402
from vnocr.engine import OcrEngine  # noqa: E402

DOCS = [("báo_cáo_đo_kiểm", 10), ("hợp_đồng_nguyên_tắc", 6), ("nghiệm_thu_hạ_tầng", 7), ("vanban_masked", 30)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--npu", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "pages").mkdir(parents=True, exist_ok=True)
    engine = OcrEngine(ROOT / "app/models", use_npu=args.npu)
    totals = {}
    with open(out / "predictions.jsonl", "w") as f:
        for doc, n in DOCS:
            for p in range(1, n + 1):
                img = cv2.imread_rgb(ROOT / f"reports/easyocr_vi/renders/{doc}/page_{p:03d}.png")[:, :, ::-1].copy()
                r = engine.ocr_page(img, p)
                for k, v in r.seconds.items():
                    totals[k] = totals.get(k, 0) + v
                f.write(json.dumps({"document": doc, "page": p, "text": r.text, "markdown": r.markdown,
                                    "lines": r.lines, "seconds": r.seconds}, ensure_ascii=False) + "\n")
                f.flush()
                (out / "pages" / f"{doc}_page_{p:03d}.md").write_text(r.markdown + "\n")
                print(f"{doc[:14]:14s} p{p:02d} lines={r.lines} tables={r.tables} "
                      + " ".join(f"{k}={v:.1f}" for k, v in r.seconds.items()), flush=True)
    (out / "summary.json").write_text(json.dumps({"placement": engine.placement, "seconds_total": totals,
                                                  "pages": sum(n for _, n in DOCS)}, indent=2))
    print("done", totals)


if __name__ == "__main__":
    main()
