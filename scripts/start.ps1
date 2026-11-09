$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$noCuda = Join-Path $root ".venv\.yata-no-cuda"

function Get-YataLanguage {
    $dataDir = if ($env:YATA_DATA_DIR) { $env:YATA_DATA_DIR } elseif ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "Yata" } else { $null }
    if ($dataDir) {
        try {
            $saved = (Get-Content -LiteralPath (Join-Path $dataDir "setup.json") -Raw -Encoding UTF8 -ErrorAction Stop | ConvertFrom-Json).locale
            if ($saved -in @("ja", "en")) { return $saved }
        } catch { }
    }
    $system = (Get-UICulture).TwoLetterISOLanguageName
    if ($system -in @("ja", "en")) { return $system }
    return "ja"
}
$language = Get-YataLanguage
function T([string]$ja, [string]$en) { if ($language -eq "en") { return $en } else { return $ja } }

function Stop-WithMessage([string]$text) {
    Write-Host ""
    Write-Host $text -ForegroundColor Red
    if ($env:YATA_ASSUME_YES -ne "1") { [void](Read-Host (T "Enter キーを押すと閉じます" "Press Enter to close")) }
    exit 1
}

function Confirm-Step([string]$question) {
    if ($env:YATA_ASSUME_YES -eq "1") {
        Write-Host "$question [Y/n] y"
        return $true
    }
    $answer = Read-Host "$question [Y/n]"
    return -not ($answer -match "^\s*[nN]")
}

function Find-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe",
                             "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe")) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

Write-Host (T "Yata を起動します。" "Starting Yata.")
Write-Host ""

Write-Host (T "[1/3] 準備ツール uv を確認しています..." "[1/3] Checking for the setup tool uv...")
$uv = Find-Uv
if (-not $uv) {
    Write-Host (T "Yata は、Python と必要な部品を入れるのに uv（Astral 社の無料ツール）を使います。この PC にはまだありません。" `
        "Yata uses uv (a free tool from Astral) to install Python and the parts it needs. This PC does not have it yet.")
    if (-not (Confirm-Step (T "uv を今インストールしますか？（インターネットから astral.sh の公式インストーラを実行します）" `
            "Install uv now? (This runs the official installer from astral.sh over the internet.)"))) {
        Stop-WithMessage (T "uv がないため起動できません。https://docs.astral.sh/uv/ の手順で uv を入れてから、もう一度起動してください。" `
            "Yata cannot start without uv. Install uv as described at https://docs.astral.sh/uv/, then start Yata again.")
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = Find-Uv
    if (-not $uv) {
        Stop-WithMessage (T "uv をインストールできませんでした。インターネットにつながっているか確認して、もう一度起動してください。" `
            "Could not install uv. Check the internet connection and start Yata again.")
    }
}

# Speaker labels (sherpa-onnx) are always included: small, and the setup page then needs no extra step.
$extras = @("--extra", "speakers")
$remember = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $bundled = Test-Path -LiteralPath (Join-Path $root ".venv\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll")
    # The CUDA Toolkit (or anything else on PATH) may already provide cuBLAS.
    $system = @($env:PATH.Split(";") | Where-Object {
        try { $_ -and [System.IO.File]::Exists((Join-Path $_ "cublas64_12.dll")) } catch { $false }
    })
    if ($bundled) {
        $extras += @("--extra", "cuda")
    } elseif ($system.Count -eq 0 -and -not (Test-Path -LiteralPath $noCuda)) {
        Write-Host ""
        Write-Host (T "NVIDIA の GPU が見つかりました。字幕を GPU で速く動かすには、NVIDIA の部品（cuBLAS など、約 600 MB）が要ります。" `
            "An NVIDIA GPU was found. Fast captions on the GPU need NVIDIA's libraries (cuBLAS and others, about 600 MB).")
        Write-Host (T "入れない場合も、字幕は CPU で動きます（少し遅く、話している途中の字幕は出ません）。" `
            "Without them, captions run on the CPU (a little slower, and no captions while people speak).")
        if (Confirm-Step (T "GPU 用の部品も入れますか？" "Install the GPU libraries too?")) {
            $extras += @("--extra", "cuda")
        } else {
            $remember = $true
        }
    }
}

Write-Host ""
Write-Host (T "[2/3] Python と必要な部品を確認しています（初回は数分かかります）..." "[2/3] Checking Python and the parts Yata needs (a few minutes the first time)...")
# --inexact keeps what is already in .venv (for example the optional Qwen3-ASR packages); --frozen installs
# exactly the versions in uv.lock.
$sync = @("sync", "--frozen", "--inexact", "--no-dev") + $extras
if (-not (Test-Path -LiteralPath $venvPython)) { $sync += @("--python", "3.12") }
& $uv @sync
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage (T "部品を入れられませんでした。インターネットにつながっているか確認して、もう一度起動してください。" `
        "Could not install the parts. Check the internet connection and start Yata again.")
}
if ($remember) { New-Item -ItemType File -Force -Path $noCuda | Out-Null }

Write-Host ""
Write-Host (T "[3/3] Yata を起動します。準備ができると、ブラウザでページが開きます。" "[3/3] Starting Yata. The page opens in the browser when it is ready.")
Write-Host (T "      ページは Google Chrome か Microsoft Edge で使ってください（タブの音声を共有するため）。" `
    "      Use the page in Google Chrome or Microsoft Edge (to share a tab's audio).")
Write-Host (T "      終わるときは、このウィンドウで Ctrl+C を押すか、このウィンドウを閉じます。" `
    "      To stop, press Ctrl+C in this window or close it.")
Write-Host ""
& $venvPython -B -m yata --open
if ($LASTEXITCODE -ne 0) {
    Stop-WithMessage (T "Yata を起動できませんでした。上のメッセージを確認してください。" "Yata could not start. See the messages above.")
}
