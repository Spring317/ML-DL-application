"""Self-test: is the NPU used, and how fast is one page?

  python -m vnocr.selftest [DOCUMENT]        (VNOCR.exe --selftest in the installed app)

Prints and logs the ONNX Runtime build, available execution providers, where each model was
placed (NPU = QNN/Hexagon, CPU = fallback) and per-stage times of one page on NPU and CPU.
Without DOCUMENT the bundled sample page is used.
"""

import sys
import time
from pathlib import Path

import onnxruntime as ort

from .engine import OcrEngine, default_model_dir
from .logutil import make_logger


def sample_document():
    for cand in (default_model_dir().parent / "samples" / "sample_page.png",
                 default_model_dir() / "sample_page.png"):
        if cand.exists():
            return cand
    return None


def run(document=None, log=None):
    log = log or make_logger()
    lines = []

    def out(msg):
        lines.append(msg)
        log(msg)

    out(f"onnxruntime {ort.__version__}; providers: {ort.get_available_providers()}")
    document = document or sample_document()
    for use_npu in (True, False):
        t0 = time.perf_counter()
        engine = OcrEngine(default_model_dir(), use_npu=use_npu, log=log)
        out(f"{'NPU' if use_npu else 'CPU-only'} run: models loaded in {time.perf_counter() - t0:.1f}s, "
            f"placement {engine.placement}")
        if document:
            for n, img in engine.load_document(document, pages=[1]):
                engine.ocr_page(img, n)                      # warm-up
                r = engine.ocr_page(img, n)
                out(f"  one page: {r.lines} lines, " + ", ".join(f"{k} {v:.2f}" for k, v in r.seconds.items()))
                out("  text: " + r.text[:160])
    out(f"log file: {log.path}")
    return "\n".join(lines)


def main():
    run(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
