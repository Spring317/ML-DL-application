# Build VNOCR-Setup-<version>-arm64.exe on a Windows 11 ARM64 machine.
#   powershell -ExecutionPolicy Bypass -File build_installer.ps1
# Needs: ARM64 Python 3.12 (python.org) and internet access. Inno Setup is downloaded if missing.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Python environment (ARM64 only)
if (-not (Test-Path .venv)) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12-arm64 -m venv .venv }
    else { & python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"
$arch = & $py -c "import platform; print(platform.machine())"
if ($arch -ne "ARM64") { throw "Python is '$arch'; install the ARM64 Python from python.org." }
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements-win-arm64.txt pyinstaller
$version = & $py -c "import vnocr; print(vnocr.__version__)"

# 2. Standalone folder dist\VNOCR (Python, libraries, models, sample)
& $py -m PyInstaller --noconfirm --clean --windowed --name VNOCR `
    --collect-all onnxruntime --collect-all onnxruntime_qnn --collect-all pymupdf `
    --collect-submodules scipy --hidden-import tkinter `
    --add-data "models;models" --add-data "samples;samples" launcher.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# 3. Quick check of the built exe on the sample page (CPU and NPU if present)
$result = Join-Path $env:TEMP "vnocr_build_check.json"
Start-Process -Wait -FilePath "dist\VNOCR\VNOCR.exe" -ArgumentList "samples\sample_page.png", "-o", "`"$result`""
& $py tools\smoke_test.py $result --max-cer 0.02
if ($LASTEXITCODE -ne 0) { throw "Built exe gives wrong OCR output" }

# 4. Installer
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
    Invoke-WebRequest "https://jrsoftware.org/download.php/is.exe" -OutFile "$env:TEMP\innosetup.exe"
    Start-Process -Wait "$env:TEMP\innosetup.exe" -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-"
}
& $iscc "/DAppVersion=$version" installer\vnocr.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Write-Host "Installer: dist\VNOCR-Setup-$version-arm64.exe"
