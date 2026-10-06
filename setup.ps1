# One-time setup on Windows 11 ARM64 (Snapdragon X Plus / X Elite).
# Needs the ARM64 build of Python 3.11-3.13 from python.org ("Windows installer (ARM64)").
$ErrorActionPreference = "Stop"
$py = (Get-Command py -ErrorAction SilentlyContinue)
if ($py) { $python = "py -3.12-arm64" } else { $python = "python" }
$arch = & cmd /c "$python -c ""import platform; print(platform.machine())"""
if ($arch -ne "ARM64") {
    Write-Error "Python reports '$arch'. Install the ARM64 Python from python.org: the NPU runtime (onnxruntime-qnn) only ships ARM64 wheels."
}
& cmd /c "$python -m venv .venv"
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-win-arm64.txt
Write-Host "Setup done. Start the app with run.bat, or: .venv\Scripts\python -m vnocr document.pdf -o result.md"
