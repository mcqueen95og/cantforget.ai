# ---------------------------------------------------------------------------
# cantforget.ai — one-shot installer for Windows (PowerShell 5.1+)
#
#   Open PowerShell in the project folder and run:
#     powershell -ExecutionPolicy Bypass -File scripts\install.ps1
#
# Does everything: checks Python, creates a private virtual environment,
# installs every library, and downloads the AI model files. Internet is
# needed once, for the downloads; the app itself then runs fully offline.
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

# Always run from the project root even if invoked from elsewhere.
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir
$ModelsDir  = Join-Path $ProjectDir "data\models"
$SpeakerDir = Join-Path $ModelsDir "speaker"
$VenvDir    = Join-Path $ProjectDir ".venv"
$VenvPy     = Join-Path $VenvDir "Scripts\python.exe"

function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Info($msg) { Write-Host "  ->  $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "  [!]  $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "cantforget.ai - installer (Windows)" -ForegroundColor White
Write-Host ""

# --- 1. Python -------------------------------------------------------------
Write-Host "1/4  Checking Python" -ForegroundColor White
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Host "Python was not found."
    Write-Host "Install it from https://www.python.org/downloads/ (tick 'Add to PATH'),"
    Write-Host "then re-open PowerShell and run this script again."
    Read-Host "Press Enter to close"
    exit 1
}
$pyCmd = $py.Source
$verOutput = & $pyCmd -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
$parts = $verOutput.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Warn "Python $verOutput found - 3.10 or newer is required."
    Write-Host "Please upgrade Python and re-run this script."
    Read-Host "Press Enter to close"
    exit 1
}
Ok "Python $verOutput found"

# --- 2. Virtual environment -------------------------------------------------
Write-Host "2/4  Creating a private Python environment (.venv)" -ForegroundColor White
if (-not (Test-Path $VenvPy)) {
    & $pyCmd -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
    Ok "Environment created at $VenvDir"
} else {
    Ok "Environment already exists - reusing it"
}

# --- 3. Libraries ------------------------------------------------------------
Write-Host "3/4  Installing the Python libraries (this can take a few minutes)" -ForegroundColor White
& $VenvPy -m pip install --upgrade pip --quiet
& $VenvPy -m pip install -r (Join-Path $ProjectDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Library installation failed." }
Ok "Libraries installed (FastAPI, OpenCV, Whisper, sherpa-onnx, ...)"

# --- 4. AI model files --------------------------------------------------------
Write-Host "4/4  Downloading the AI model files" -ForegroundColor White
New-Item -ItemType Directory -Force -Path $SpeakerDir | Out-Null

function Fetch($url, $dest, $name) {
    if (Test-Path $dest) {
        if ((Get-Item $dest).Length -gt 0) { Ok "$name already present"; return }
    }
    Info "downloading $name ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile "$dest.part" -UseBasicParsing
    Move-Item -Force "$dest.part" $dest
    Ok "$name saved"
}

Fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
      (Join-Path $ModelsDir "face_detection_yunet.onnx") `
      "face detector (YuNet)"

Fetch "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" `
      (Join-Path $ModelsDir "face_recognition_sface_2021dec.onnx") `
      "face recogniser (SFace)"

Fetch "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" `
      (Join-Path $ModelsDir "silero_vad.onnx") `
      "voice detector (Silero VAD)"

Fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" `
      (Join-Path $SpeakerDir "eres2net.onnx") `
      "voice recogniser (ERes2Net)"

}

# The speech-to-text model (Whisper "tiny") downloads itself on first use.
Write-Host ""
Write-Host "Install complete [OK]" -ForegroundColor Green
Write-Host ""
Write-Host "To start the app:"
Write-Host "    cd $ProjectDir"
Write-Host "    .\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"
Write-Host ""
Write-Host "Then open  http://127.0.0.1:8000/  in your browser."
Write-Host ""
Read-Host "Press Enter to close"
