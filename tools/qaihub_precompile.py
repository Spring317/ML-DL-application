#!/usr/bin/env python3
"""Pre-compile the NPU models for Snapdragon X Plus on Qualcomm AI Hub and verify them on the device.

For each NPU model (CRAFT, VietOCR CNN, transformer encoder):
  1. compile job: ONNX -> precompiled QNN ONNX (EPContext wrapping a QNN context binary), FP16,
     QAIRT 2.50 (the version onnxruntime-qnn 2.6 is built against);
  2. inference job: the compiled model on the device, fed the exact inputs the app produces for
     samples/sample_page.png;
  3. comparison with ONNX Runtime CPU, and the page text decoded from the device outputs.
The compiled models are saved to app/models/npu/ and loaded by the app before the source ONNX.

  python app/tools/qaihub_precompile.py
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import qai_hub as hub  # noqa: E402

from vnocr import cv_compat as cv2  # noqa: E402
from vnocr.engine import NPU_MODELS, OcrEngine  # noqa: E402

DEVICE = hub.Device("Snapdragon X Plus 8-Core CRD")
OPTIONS = "--target_runtime precompiled_qnn_onnx --qairt_version 2.50 --quantize_full_type float16"
OUT = APP / "models" / "npu"
REPORT = APP / "models" / "npu" / "precompile_report.json"


class Recorder:
    """Wraps an ORT session: records inputs, or replays given outputs in call order."""

    def __init__(self, sess, replay=None):
        self.sess, self.inputs, self.replay, self.k = sess, [], replay, 0

    def run(self, names, feeds, *a):
        self.inputs.append({k: np.array(v) for k, v in feeds.items()})
        if self.replay is None:
            return self.sess.run(names, feeds, *a)
        out = self.replay[self.k]
        self.k += 1
        return out


def page_text(engine, img):
    return engine.ocr_page(img, 1).text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = cv2.imread_rgb(APP / "samples/sample_page.png")[:, :, ::-1].copy()
    engine = OcrEngine(APP / "models", use_npu=False, log=lambda m: None)
    rec = {n: Recorder(engine.sessions[n]) for n in NPU_MODELS}
    engine.sessions.update(rec)
    cpu_text = page_text(engine, img)
    report = {"device": DEVICE.name, "options": OPTIONS, "models": {}}

    compiled, jobs = {}, {}
    for name in NPU_MODELS:
        cj = hub.submit_compile_job(model=str(APP / "models" / f"{name}.onnx"), device=DEVICE, options=OPTIONS,
                                    name=f"vnocr_{name}_precompiled_qnn")
        print(name, "compile job", cj.job_id, flush=True)
        compiled[name] = cj
    for name, cj in compiled.items():
        tm = cj.get_target_model()
        if tm is None:
            st = cj.get_status()
            report["models"][name] = {"compile_job": cj.job_id, "error": st.message}
            print(name, "COMPILE FAILED", st.message, flush=True)
            continue
        inputs = rec[name].inputs
        names = list(inputs[0])
        ds = hub.upload_dataset({k: [x[k] for x in inputs] for k in names}, name=f"vnocr_{name}_sample_inputs")
        jobs[name] = (tm, hub.submit_inference_job(model=tm, device=DEVICE, inputs=ds, name=f"vnocr_{name}_precompiled_check"))
        report["models"][name] = {"compile_job": cj.job_id, "model_id": tm.model_id, "inference_job": jobs[name][1].job_id,
                                  "calls": len(inputs)}
        print(name, "inference job", jobs[name][1].job_id, f"({len(inputs)} inputs)", flush=True)

    device_out = {}
    for name, (tm, ij) in jobs.items():
        st = ij.wait()
        r = report["models"][name]
        if not st.success:
            r["error"] = st.message
            print(name, "INFERENCE FAILED", st.message, flush=True)
            continue
        data = ij.download_output_data()
        keys = sorted(data, key=lambda k: int(k.split("_")[-1]))  # AI Hub renames outputs to output_<i>
        outs = [[np.asarray(data[k][i]) for k in keys] for i in range(r["calls"])]
        device_out[name] = outs
        # numeric agreement with ONNX Runtime CPU on the same inputs (outputs matched by position)
        corr = []
        for feeds, o in zip(rec[name].inputs, outs):
            for ref, v in zip(engine.sessions[name].sess.run(None, feeds), o):
                corr.append(float(np.corrcoef(ref.ravel(), v.ravel())[0, 1]))
        r["min_correlation_vs_cpu"] = min(corr)
        # save the compiled model (a directory or archive with model.onnx + context binary)
        dst = OUT / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        got = Path(tm.download(str(dst / "download")))
        if got.suffix == ".zip" or zipfile.is_zipfile(got):
            with zipfile.ZipFile(got) as z:
                z.extractall(dst)
            got.unlink()
        r["files"] = sorted(str(p.relative_to(OUT)) for p in dst.rglob("*") if p.is_file())
        print(name, r, flush=True)

    if len(device_out) == len(NPU_MODELS):
        # text with device outputs for every NPU stage. Replaying CRAFT changes the boxes and so the
        # later inputs, so stages are replayed one at a time from the CPU recording.
        for name in NPU_MODELS:
            eng = OcrEngine(APP / "models", use_npu=False, log=lambda m: None)
            eng.sessions[name] = Recorder(eng.sessions[name], replay=device_out[name])
            text = page_text(eng, img)
            report["models"][name]["page_text_identical_to_cpu"] = text == cpu_text
            report["models"][name]["page_text_char_diff"] = sum(a != b for a, b in zip(text, cpu_text)) + abs(len(text) - len(cpu_text))
            print(name, "page text identical to CPU:", text == cpu_text, flush=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print("report", REPORT)


if __name__ == "__main__":
    main()
