"""OCR engine: CRAFT + VietOCR on ONNX Runtime, NPU (QNN) where it helps and CPU elsewhere.

Placement (measured on Snapdragon X Plus through Qualcomm AI Hub):
  CRAFT detector, 768x768 tiles        NPU   41.8 ms per tile
  VietOCR CNN, 32x512 line crops       NPU   18.3 ms per line
  transformer encoder, 8 lines         NPU   32.5 ms per batch
  cross-attention K/V, 8 lines         CPU   (small matmuls)
  greedy decoder step with KV cache    CPU   4.7 ms per step (faster than the NPU's 7.4 ms)
The NPU only runs fixed shapes: pages are cut into overlapping tiles, line crops are padded
to 512 px with a mask so padding is ignored, and long lines are split at word gaps.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import cv_compat as cv2
from .craft_post import adjust_coordinates, get_det_boxes
from .layout import (RECOGNIZER_MAX_WIDTH, assemble_page, build_page_layout, crop_line, detect_tables,
                     drop_stray_boxes, group_words_into_lines, line_crop_tensors, remove_colored_marks)
from .textpost import post_process

RENDER_DPI = 312          # the pipeline was tuned and evaluated on 312-DPI page renders
CRAFT_TILE = 768
CRAFT_OVERLAP = 160
CRAFT_PAGE_SCALE = 0.45
SRC_LEN = 256
CACHE_LEN = 128
BATCH = 8
NEG = -1e4
SOS, EOS = 1, 2
NPU_MODELS = ("craft_768", "vietocr_cnn", "vietocr_encoder_b8")
CPU_MODELS = ("vietocr_cross_kv_b8", "vietocr_decoder_kv_b8")


@dataclass
class PageResult:
    page: int
    text: str
    markdown: str
    lines: int
    tables: int
    seconds: dict = field(default_factory=dict)


class Vocab:
    def __init__(self, chars):
        self.i2c = {i + 4: c for i, c in enumerate(chars)}

    def decode(self, ids):
        out = []
        for i in ids[1:] if ids and ids[0] == SOS else ids:
            if i == EOS:
                break
            out.append(self.i2c.get(i, ""))
        return "".join(out)


_QNN = {}


def _qnn_backend(log):
    """Find the QNN execution provider once.

    onnxruntime-qnn >= 2.0 is a plugin EP: its library is registered with ONNX Runtime and the
    NPU is selected as an EP device. Older onnxruntime-qnn builds (1.x) have it built in."""
    if "mode" in _QNN:
        return _QNN
    _QNN["mode"] = None
    try:
        import onnxruntime_qnn as qnn_ep
        try:
            ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
        except Exception as e:  # already registered in this process
            if "already" not in str(e).lower():
                raise
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
        npu = [d for d in devices if "NPU" in str(getattr(d.device, "type", "")).upper()] or devices
        if npu:
            _QNN.update(mode="plugin", devices=npu, htp=qnn_ep.get_qnn_htp_path())
    except ImportError:
        if "QNNExecutionProvider" in ort.get_available_providers():
            _QNN["mode"] = "builtin"
    except Exception as e:
        log(f"QNN plugin registration failed: {e}")
    log(f"QNN execution provider: {_QNN['mode'] or 'not available'}")
    return _QNN


def _qnn_session(path, cache_dir, log):
    """Session on the Hexagon NPU via the QNN execution provider (FP16), or None.

    The compiled context is cached so later starts skip graph compilation. CPU fallback
    inside the session is disabled: a model runs fully on the NPU or is loaded on the CPU."""
    qnn = _qnn_backend(log)
    if not qnn["mode"]:
        return None
    ctx = Path(cache_dir) / f"{Path(path).stem}_qnn_ctx.onnx"
    base = {"backend_path": qnn.get("htp", "QnnHtp.dll")}
    # 1) model pre-compiled for Snapdragon X Plus on Qualcomm AI Hub (models/npu/<name>/*.onnx),
    # 2) context compiled on this laptop at an earlier start, 3) compile the source ONNX now.
    precompiled = sorted((Path(path).parent / "npu" / Path(path).stem).rglob("*.onnx"))
    attempts = [(str(precompiled[0]), False)] if precompiled else []
    attempts += [(str(ctx), False)] if ctx.exists() else [(str(path), True)]
    for (source, compile_now), opts in [(a, o) for a in attempts for o in
                                        ({**base, "enable_htp_fp16_precision": "1", "htp_performance_mode": "burst"}, base)]:
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if compile_now:
            so.add_session_config_entry("ep.context_enable", "1")
            so.add_session_config_entry("ep.context_file_path", str(ctx))
        try:
            if qnn["mode"] == "plugin":
                so.add_provider_for_devices(qnn["devices"], opts)
                sess = ort.InferenceSession(source, sess_options=so)
            else:
                sess = ort.InferenceSession(source, sess_options=so, providers=["QNNExecutionProvider"],
                                            provider_options=[opts])
            if "QNNExecutionProvider" in sess.get_providers():
                log(f"{Path(path).name}: NPU via {Path(source).name}")
                return sess
            log(f"{Path(path).name}: QNN session created but not using the NPU ({sess.get_providers()})")
        except Exception as e:  # unsupported option set or graph: try the next, then the CPU
            log(f"QNN session for {Path(path).name} failed ({sorted(opts)}): {str(e).splitlines()[0][:200]}")
    return None


def _cpu_session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class OcrEngine:
    def __init__(self, model_dir, use_npu=True, cache_dir=None, log=print):
        self.log = log
        model_dir = Path(model_dir)
        if cache_dir is None:   # writable per-user folder (the model folder may be read-only)
            base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
            cache_dir = Path(base) / "vnocr" / "qnn_cache"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = Vocab((model_dir / "vocab.txt").read_text(encoding="utf-8"))
        self.sessions, self.placement = {}, {}
        t0 = time.perf_counter()
        for name in NPU_MODELS + CPU_MODELS:
            path = model_dir / f"{name}.onnx"
            sess = _qnn_session(path, cache_dir, log) if (use_npu and name in NPU_MODELS) else None
            self.placement[name] = "NPU" if sess is not None else "CPU"
            self.sessions[name] = sess or _cpu_session(path)
        self.load_seconds = time.perf_counter() - t0
        log("Model placement: " + ", ".join(f"{k}={v}" for k, v in self.placement.items())
            + f" (loaded in {self.load_seconds:.1f}s)")

    # ------------------------------------------------------------------ input

    @staticmethod
    def load_document(path, dpi=RENDER_DPI, pages=None):
        """Yield (page_number, BGR uint8 image). PDFs are rendered at ``dpi``; images are used as is."""
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            import pymupdf
            with pymupdf.open(path) as pdf:
                numbers = pages or range(1, len(pdf) + 1)
                for n in numbers:
                    pix = pdf[n - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                    yield n, np.ascontiguousarray(rgb[:, :, ::-1])
        else:
            yield 1, np.ascontiguousarray(cv2.imread_rgb(path)[:, :, ::-1])

    # ------------------------------------------------------------------ detection

    def _detect_lines(self, img):
        scaled = cv2.resize(img, None, fx=CRAFT_PAGE_SCALE, fy=CRAFT_PAGE_SCALE, interpolation=cv2.INTER_AREA)
        sh, sw = scaled.shape[:2]
        step = CRAFT_TILE - CRAFT_OVERLAP

        def starts(n):
            return [0] if n <= CRAFT_TILE else list(range(0, n - CRAFT_TILE, step)) + [n - CRAFT_TILE]

        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        page = np.zeros(((sh + 1) // 2, (sw + 1) // 2, 2), dtype=np.float32)
        craft = self.sessions["craft_768"]
        for y in starts(sh):
            for x in starts(sw):
                canvas = np.zeros((CRAFT_TILE, CRAFT_TILE, 3), dtype=np.uint8)
                part = scaled[y:y + CRAFT_TILE, x:x + CRAFT_TILE]
                canvas[:part.shape[0], :part.shape[1]] = part
                tensor = ((canvas.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
                m = craft.run(None, {"input_image": tensor})[0][0]   # outputs by position: AI Hub renames them
                y2, x2 = y // 2, x // 2
                hh, ww = min(m.shape[0], page.shape[0] - y2), min(m.shape[1], page.shape[1] - x2)
                page[y2:y2 + hh, x2:x2 + ww] = np.maximum(page[y2:y2 + hh, x2:x2 + ww], m[:hh, :ww])
        boxes = get_det_boxes(page[:, :, 0], page[:, :, 1], text_threshold=0.7, link_threshold=0.4, low_text=0.4)
        if len(boxes) == 0:
            return []
        quads = adjust_coordinates(boxes, 1.0 / CRAFT_PAGE_SCALE, 1.0 / CRAFT_PAGE_SCALE)
        lines = [b for b in group_words_into_lines(quads) if b["w"] >= 5 and b["h"] >= 5]
        return drop_stray_boxes(lines)

    # ------------------------------------------------------------------ recognition

    def _recognize(self, crops, widths):
        """Line crops -> text. CNN per crop, then batches of 8 lines sorted by width (shorter batches
        stop earlier; every line is decoded independently, so the order does not change the text)."""
        cnn = self.sessions["vietocr_cnn"]
        feats = [cnn.run(None, {"line_image": c})[0] for c in crops]
        order = sorted(range(len(crops)), key=lambda i: -widths[i])
        texts = [""] * len(crops)
        steps = 0
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            n = len(idx)
            idx_p = idx + [idx[0]] * (BATCH - n)
            f = np.concatenate([feats[i] for i in idx_p], axis=1).astype(np.float32)
            mask = np.full((BATCH, SRC_LEN), NEG, np.float32)
            for row, i in enumerate(idx_p):
                mask[row, :min(max(widths[i] // 2, 4), SRC_LEN)] = 0.0
            memory = self.sessions["vietocr_encoder_b8"].run(None, {"features": f, "src_mask": mask})[0]
            ck, cv = self.sessions["vietocr_cross_kv_b8"].run(None, {"memory": memory.astype(np.float32)})
            seqs, k_steps = self._greedy(ck, cv, mask, n)
            steps += k_steps
            for i, seq in zip(idx, seqs):
                texts[i] = self.vocab.decode(seq).strip()
        return texts, steps

    def _greedy(self, ck, cv, mask, n):
        dec = self.sessions["vietocr_decoder_kv_b8"]
        nl, _, h, _, d = ck.shape
        pk = np.zeros((nl, BATCH, h, CACHE_LEN, d), np.float32)
        pv = np.zeros_like(pk)
        token = np.full((BATCH, 1), SOS, np.int32)
        seqs = [[SOS] for _ in range(BATCH)]
        done = np.zeros(BATCH, bool)
        done[n:] = True
        t = 0
        for t in range(CACHE_LEN):
            onehot = np.zeros((1, CACHE_LEN), np.float32)
            onehot[0, t] = 1.0
            smask = np.full((1, CACHE_LEN), NEG, np.float32)
            smask[0, :t + 1] = 0.0
            nxt, nk, nv = dec.run(None, {"token": token, "pos_onehot": onehot, "self_mask": smask, "past_k": pk,
                                         "past_v": pv, "cross_k": ck, "cross_v": cv, "src_mask": mask})
            pk[:, :, :, t] = nk[:, :, :, 0]
            pv[:, :, :, t] = nv[:, :, :, 0]
            for i in np.where(~done)[0]:
                seqs[i].append(int(nxt[i]))
                if nxt[i] == EOS:
                    done[i] = True
            token = nxt.reshape(BATCH, 1).astype(np.int32)
            if done.all():
                break
        return seqs[:n], t + 1

    # ------------------------------------------------------------------ page

    def ocr_page(self, img_bgr, page_number=1, remove_stamps=True):
        sec = {}
        t = time.perf_counter()
        img = remove_colored_marks(img_bgr) if remove_stamps else img_bgr
        boxes = self._detect_lines(img)
        sec["detect"] = time.perf_counter() - t

        t = time.perf_counter()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tables = detect_tables(gray)
        layout = build_page_layout(boxes, tables, gray.shape[1], gray)
        crops, owner, widths = [], [], []
        for ui, unit in enumerate(layout.units):
            b = unit["box"]
            limits = None
            if unit["kind"] == "cell":
                tb = layout.tables[unit["table"]]
                limits = (tb.cols[unit["col"]].at(b["yc"]) + 6, tb.cols[unit["col"] + 1].at(b["yc"]) - 6)
            crop = crop_line(gray, b, x_limits=limits)
            for tensor, content_w in line_crop_tensors(crop, RECOGNIZER_MAX_WIDTH, "normalized"):
                crops.append(tensor)
                owner.append(ui)
                widths.append(content_w)
        sec["layout"] = time.perf_counter() - t

        t = time.perf_counter()
        texts, steps = self._recognize(crops, widths) if crops else ([], 0)
        sec["recognize"] = time.perf_counter() - t

        pieces = {}
        for ui, text in zip(owner, texts):
            pieces.setdefault(ui, []).append(text)
        unit_texts = [post_process(" ".join(pieces.get(ui, []))) for ui in range(len(layout.units))]
        plain, markdown = assemble_page(layout, unit_texts)
        sec["total"] = sum(sec.values())
        return PageResult(page=page_number, text=plain, markdown=markdown, lines=len(crops),
                          tables=len(layout.tables), seconds={**sec, "decoder_steps": steps})

    def ocr_document(self, path, pages=None, progress=None):
        results = []
        for n, img in self.load_document(path, pages=pages):
            r = self.ocr_page(img, n)
            results.append(r)
            if progress:
                progress(r)
        return results


def default_model_dir():
    env = os.environ.get("VNOCR_MODELS")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "models", here / "models", Path.cwd() / "models"):
        if (cand / "vocab.txt").exists():
            return cand
    return here.parent / "models"
