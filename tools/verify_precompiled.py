#!/usr/bin/env python3
"""Verify the AI Hub pre-compiled NPU models against ONNX Runtime CPU, using finished inference jobs.

For each model: download the job's inputs and device outputs, run the source ONNX on the CPU with
the same inputs, compare (outputs are matched by position; AI Hub renames them), then decode the
sample page with each model's device outputs replayed and compare the text with the CPU text.

  python app/tools/verify_precompiled.py
"""
import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import qai_hub as hub  # noqa: E402

from vnocr import cv_compat as cv2  # noqa: E402
from vnocr.engine import OcrEngine  # noqa: E402

JOBS = {"craft_768": ("jg9o0mwwg", "jp4en1v8g"), "vietocr_cnn": ("jp1o2je85", "jpx094y3p"),
        "vietocr_encoder_b8": ("jgd6n3orp", "jgn10n3kp")}


class Replay:
    def __init__(self, sess, outs, feeds):
        self.sess, self.outs, self.feeds, self.k, self.max_input_diff = sess, outs, feeds, 0, 0.0

    def run(self, names, feeds, *a):
        for key, v in feeds.items():   # the replayed device output must belong to the same input
            self.max_input_diff = max(self.max_input_diff, float(np.abs(np.asarray(v) - self.feeds[self.k][key]).max()))
        out = self.outs[self.k]
        self.k += 1
        return out


def main():
    img = cv2.imread_rgb(APP / "samples/sample_page.png")[:, :, ::-1].copy()
    cpu = OcrEngine(APP / "models", use_npu=False, log=lambda m: None)
    cpu_text = cpu.ocr_page(img, 1).text
    report = {}
    for name, (cj, ij) in JOBS.items():
        job = hub.get_job(ij)
        st = job.get_status()
        r = {"compile_job": cj, "inference_job": ij, "device": job.device.name, "status": st.code}
        inp = job.inputs.download()
        data = job.download_output_data()
        keys = sorted(data, key=lambda k: int(k.split("_")[-1]))           # output_0, output_1, ...
        n = len(data[keys[0]])
        feeds = [{k: np.asarray(inp[k][i]) for k in inp} for i in range(n)]
        dev = [[np.asarray(data[k][i]) for k in keys] for i in range(n)]
        sess = cpu.sessions[name]
        corr, maxdiff, scale = [], 0.0, 0.0
        for f, d in zip(feeds, dev):
            ref = sess.run(None, f)
            for a, b in zip(ref, d):
                corr.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
                maxdiff = max(maxdiff, float(np.abs(a - b).max()))
                scale = max(scale, float(np.abs(a).max()))
        r.update(calls=n, min_correlation=min(corr), max_abs_diff=maxdiff, output_scale=scale,
                 identical_to_cpu=bool(maxdiff == 0))
        eng = OcrEngine(APP / "models", use_npu=False, log=lambda m: None)
        rep = Replay(eng.sessions[name], dev, feeds)
        eng.sessions[name] = rep
        text = eng.ocr_page(img, 1).text
        r.update(replay_inputs_match=rep.max_input_diff < 1e-5, page_text_identical=text == cpu_text)
        if text != cpu_text:
            import difflib
            r["text_diff"] = [d for d in difflib.ndiff(cpu_text.split(), text.split()) if d[0] in "+-"][:20]
        report[name] = r
        print(name, r, flush=True)
    (APP / "models/npu/precompile_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
