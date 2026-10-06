# VNOCR: Vietnamese document OCR for Windows on Snapdragon

Open a PDF or an image of a Vietnamese document and get its text, with tables rebuilt as
Markdown. It runs fully offline on a Snapdragon X Plus / X Elite laptop: the heavy models
run on the Hexagon NPU through ONNX Runtime's QNN execution provider, and the rest runs on
the CPU.

## What runs where

| Stage | Model and fixed input | Runs on | Time on X Plus (AI Hub profile) |
|---|---|---|---|
| Text detection | CRAFT, 768×768 tiles (6 per A4 page, 160 px overlap) | NPU | 41.8 ms per tile |
| Line recognition, CNN | VietOCR VGG, 1×3×32×512 line crops | NPU | 18.3 ms per line |
| Transformer encoder | 8 lines + padding mask | NPU | 32.5 ms per 8 lines |
| Cross-attention K/V | small matmuls on the encoder output | CPU | – |
| Greedy decoder (KV cache) | one character per step, 8 lines | CPU | 4.7 ms per step |
| Layout and tables | ruling-line table detection, reading order | CPU | – |

The NPU only accepts fixed shapes:
- Pages are cut into overlapping tiles.
- Each line crop is padded to 512 px, and a mask makes attention ignore the padding.
- Long lines are split at word gaps.

The decoder runs on the CPU because it is faster there than on the NPU (4.7 ms against 7.4 ms per step).

**Accuracy and speed.** These are the evaluation results of this pipeline on 53 test pages,
with frozen criteria:
- Content CER 2.29% and WER 4.51%. CRAFT, the CNN and the encoder ran on the Snapdragon X Plus
  NPU and the decoder on its CPU, through AI Hub.
- About 2.4 s per page for the models, excluding layout. This is the sum of the per-model
  device profiles.

See `reports/evaluation_v2.md` in the project.

## This package contains no .exe yet: build it once on Windows ARM64

This folder holds the app's source code, the models and the build scripts. **It contains no `.exe`.**
The `.exe` and the installer have to be built **on Windows 11 ARM64**: PyInstaller bundles the Python
and native libraries of the machine it runs on, so it cannot produce a Windows ARM program from
Linux or from x64 Windows. You only need to build once; after that, anyone can use the installer
without Python.

### Option A: build on a Snapdragon laptop (recommended, about 10–15 minutes)

1. Install **Python 3.12 for ARM64** from python.org ("Windows installer (ARM64)").
   The x64 Python will not work, because the NPU runtime only ships ARM64 packages.
2. Unzip this package, open PowerShell in the `VNOCR` folder and run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File build_installer.ps1
   ```

3. The script:
   1. creates `.venv` and installs the libraries (onnxruntime-qnn, numpy, scipy, pillow, pymupdf, pyinstaller);
   2. bundles everything into `dist\VNOCR\` with PyInstaller;
   3. OCRs the sample page with the built `VNOCR.exe` and stops with an error if the text does not match the reference;
   4. builds the installer with Inno Setup, downloading Inno Setup if it is missing.
4. You get:
   - `dist\VNOCR\VNOCR.exe`: the app, which runs without Python. Copy the whole `dist\VNOCR` folder.
   - `dist\VNOCR-Setup-0.1.0-arm64.exe`: the installer to give to other users.
5. Check the NPU: run `dist\VNOCR\VNOCR.exe --selftest`, or **VNOCR self-test** from the Start menu
   after installing. Each model should show `NPU`.

### Option B: build on GitHub (no ARM machine needed)

1. Create a GitHub repository and push this folder to it. Plain git is enough: every model file
   is under GitHub's 100 MB limit, so no Git LFS is needed. From a terminal with the GitHub CLI:

   ```bash
   gh auth login                      # one-time code, can be confirmed from a phone
   git init -b main && git add . && git commit -m "VNOCR 0.1.0"
   gh repo create vnocr --private --source . --push
   gh workflow run build-windows-arm64
   gh run watch                       # follow the build
   gh run download --name VNOCR-Setup-arm64   # fetch the installer
   ```
2. Run the **build-windows-arm64** workflow (`gh workflow run`, or the Actions tab on github.com).
3. On a Windows 11 ARM64 machine, GitHub then:
   - builds the installer;
   - installs it silently;
   - checks the installed app's OCR output on the sample page.
4. Download `VNOCR-Setup-0.1.0-arm64.exe` with `gh run download`, or from the workflow run's **Artifacts**.

GitHub's ARM machines have no NPU, so this test covers the CPU path only. Run the self-test on a
Snapdragon laptop to check the NPU.

### Option C: run from source without building (developers)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1   # creates .venv and installs the libraries
run.bat                                              # opens the desktop app
```

## Installing the built installer (end users)

1. Run `VNOCR-Setup-0.1.0-arm64.exe`, click through, and start **VNOCR** from the Start menu.
   - No Python, admin rights or extra downloads are needed.
   - Choose "all users" in the installer if you want it in Program Files.
2. **VNOCR self-test (NPU check)** in the Start menu shows whether the models run on the NPU.
3. To uninstall: Settings > Apps, or the **Uninstall VNOCR** shortcut.

## Use

- **Desktop app:** start **VNOCR** from the Start menu (installed), `dist\VNOCR\VNOCR.exe` (built), or `run.bat` (from source).
  1. Open a document.
  2. Optionally enter pages, e.g. `1-3,5`.
  3. Click **Run OCR**.

  Results appear page by page. Save them as Markdown (tables kept) or plain text.
- **Command line** (from source; with the built app use `VNOCR.exe` in place of `.venv\Scripts\python -m vnocr`):

  ```powershell
  .venv\Scripts\python -m vnocr scan.pdf -o result.md        # Markdown
  .venv\Scripts\python -m vnocr scan.pdf --pages 1-2 -o r.txt
  .venv\Scripts\python -m vnocr photo.jpg -o r.json          # JSON with per-stage timings
  .venv\Scripts\python -m vnocr scan.pdf --cpu               # without the NPU
  ```

- **Logs:** everything is written to `%LOCALAPPDATA%\vnocr\vnocr.log`.
- **Check that the NPU is used:**

  ```powershell
  .venv\Scripts\python -m vnocr.selftest sample.pdf
  ```

  This prints the execution providers, where each model was placed (`NPU` or `CPU`) and the
  time of one page, both on the NPU and on the CPU only.

**Pre-compiled NPU models.** `models/npu/` holds CRAFT, the CNN and the encoder compiled for the Snapdragon X Plus
on Qualcomm AI Hub, as QNN FP16 context binaries (QAIRT 2.50, the version onnxruntime-qnn 2.6 targets).
- They were run on an X Plus with the sample page's inputs: correlation with the CPU 0.99997 or higher, and identical page
  text. See `models/npu/precompile_report.json`.
- The app loads these first, so there is no compile step on the laptop.
- If the laptop's QNN runtime cannot load them, it compiles the source ONNX instead, once, and caches the result in
  `%LOCALAPPDATA%\vnocr\qnn_cache`. Failing that, it uses the CPU.
- A model that cannot run fully on the NPU is loaded on the CPU instead. The selftest shows which.
- CPU fallback inside a single model is disabled, so a model runs either fully on the NPU or fully on the CPU, never split.

**Input.**
- **PDFs** are rendered at 312 DPI, the resolution the pipeline was tuned on.
- **Images** are used as they are. Scans at about 300 DPI work best.
- **Stamps and signatures** in red or blue are painted out before detection.

## Files

| Path | Purpose |
|---|---|
| `vnocr/engine.py` | Pipeline: rendering, tiling, NPU/CPU sessions, batching, greedy decoding |
| `vnocr/layout.py` | Table detection, line grouping, crops (from `scripts/table_layout.py`) |
| `vnocr/cv_compat.py` | The OpenCV functions the pipeline needs, in numpy/scipy (no OpenCV wheels exist for Windows ARM64) |
| `vnocr/craft_post.py` | CRAFT score maps to word boxes (from EasyOCR, MIT licence) |
| `vnocr/gui.py`, `vnocr/cli.py` | Desktop window and command line |
| `models/` | Source ONNX models and vocabulary (`tools/prepare_models.py`); `models/npu/` holds the AI Hub pre-compiled QNN binaries (`tools/qaihub_precompile.py`, checked by `tools/verify_precompiled.py`) |
| `tests/` | Comparison with OpenCV and the 53-page validation (development machine) |
| `build_installer.ps1`, `installer/vnocr.iss` | Build the bundled app and the ARM64 installer |
| `.github/workflows/build-windows-arm64.yml` | Build and test the installer on GitHub's Windows ARM64 runner |
| `samples/` | Sample page and its reference OCR output (used by the build checks and the self-test) |
| `tools/smoke_test.py` | Compare an OCR result with the reference |

## Limitations

- Tested end to end on Linux with ONNX Runtime CPU: 53 evaluation pages scored 2.29% CER / 4.50% WER,
  the same as the research pipeline. **Not yet run on a Windows ARM laptop.**
  - The NPU models are the files that were validated on Snapdragon X Plus through Qualcomm AI Hub.
  - The QNN provider options and context caching follow the ONNX Runtime documentation.
  - Report any `QNN session ... failed` message from the selftest.
- Layout runs on the CPU in Python. It takes a few seconds per page and is not included in the 2.4 s model time.
- Tuned on printed Vietnamese administrative documents. Handwriting is not supported.
