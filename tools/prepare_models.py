#!/usr/bin/env python3
"""Collect the ONNX models the app needs into app/models (run on the development machine).

  craft_768.onnx          CRAFT text detector, input 1x3x768x768          -> NPU
  vietocr_cnn.onnx        VietOCR CNN, line crop 1x3x32x512               -> NPU
  vietocr_encoder_b8.onnx transformer encoder, 8 lines + padding mask      -> NPU
  vietocr_cross_kv_b8.onnx cross-attention keys/values from encoder memory -> CPU
  vietocr_decoder_kv_b8.onnx greedy decoder step with KV cache             -> CPU
  vocab.txt               VietOCR character set of the fine-tuned weights

The NPU models are the same files that were validated on Snapdragon X Plus through AI Hub
(reports/approach1_qaihub.json); the decoder step gives identical text to VietOCR's greedy
decoder on all 2,363 benchmark lines (reports/static-kv-cpu). Requires torch + vietocr.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from vietocr.tool.config import Cfg  # noqa: E402
from vietocr.tool.predictor import Predictor  # noqa: E402
from vietocr_static import EncoderKV, SRC_LEN  # noqa: E402


class CrossKV(torch.nn.Module):
    """Second half of EncoderKV: per-layer cross-attention K/V from the encoder memory."""

    def __init__(self, lt):
        super().__init__()
        self.kv = EncoderKV(lt)

    def forward(self, memory):
        from vietocr_static import _heads
        C = memory.shape[-1]
        ks, vs = [], []
        for layer in self.kv.dec.layers:
            a = layer.multihead_attn
            w, b = a.in_proj_weight, a.in_proj_bias
            ks.append(_heads(torch.nn.functional.linear(memory, w[C:2 * C], b[C:2 * C]), a.num_heads))
            vs.append(_heads(torch.nn.functional.linear(memory, w[2 * C:], b[2 * C:]), a.num_heads))
        return torch.stack(ks), torch.stack(vs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="/mnt/ssd-0/ocr_work/vietocr_ft/step2000.pth")
    ap.add_argument("--vocab", default="/mnt/ssd-0/ocr_work/vietocr_ft/vocab_step2000.txt")
    ap.add_argument("--out", default=str(ROOT / "app/models"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    copies = {
        "craft_768.onnx": ROOT / "models/onnx_models/craft_detector_static.onnx",
        "vietocr_cnn.onnx": ROOT / "artifacts/vietocr_npu/vietocr_cnn_ft2000_512.onnx",
        "vietocr_encoder_b8.onnx": ROOT / "artifacts/vietocr_npu/vietocr_encoder_ft2000_b8.onnx",
        "vietocr_decoder_kv_b8.onnx": ROOT / "artifacts/vietocr_npu/vietocr_decoder_kv_ft2000_b8.onnx",
        "vocab.txt": Path(args.vocab),
    }
    for name, src in copies.items():
        shutil.copyfile(src, out / name)
        print("copied", name)

    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = "cpu"
    config["vocab"] = Path(args.vocab).read_text()
    config["weights"] = args.weights
    lt = Predictor(config).model.eval().transformer
    mod = CrossKV(lt).eval()
    memory = torch.randn(8, SRC_LEN, 256)
    with torch.no_grad():
        ref = mod(memory)[0].numpy()
    path = out / "vietocr_cross_kv_b8.onnx"
    torch.onnx.export(mod, (memory,), str(path), input_names=["memory"], output_names=["cross_k", "cross_v"],
                      opset_version=17, dynamo=False)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"memory": memory.numpy()})[0]
    print(f"exported {path.name}: max abs diff vs PyTorch {np.abs(got - ref).max():.2e}")


if __name__ == "__main__":
    main()
