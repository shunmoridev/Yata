#!/bin/sh
set -u
cd "$(dirname "$0")" || exit 1
ROOT=$(pwd)

data_dir() {
  if [ -n "${YATA_DATA_DIR:-}" ]; then
    printf '%s\n' "$YATA_DATA_DIR"
  elif [ "$(uname -s)" = "Darwin" ]; then
    printf '%s\n' "$HOME/Library/Application Support/Yata"
  else
    printf '%s\n' "${XDG_DATA_HOME:-$HOME/.local/share}/Yata"
  fi
}
LANG_CODE=$(sed -n 's/.*"locale": *"\([a-z]*\)".*/\1/p' "$(data_dir)/setup.json" 2>/dev/null | head -n 1)
case "$LANG_CODE" in
  ja|en) ;;
  *)
    LANG_CODE=$(printf '%s' "${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}" | cut -c1-2)
    if [ -z "$LANG_CODE" ] && [ "$(uname -s)" = "Darwin" ]; then
      LANG_CODE=$(defaults read -g AppleLocale 2>/dev/null | cut -c1-2)
    fi
    case "$LANG_CODE" in ja|en) ;; *) LANG_CODE=ja ;; esac
    ;;
esac

msg() {
  if [ "$LANG_CODE" = "en" ]; then printf '%s' "$2"; else printf '%s' "$1"; fi
}

fail() {
  printf '\n%s\n' "$*" >&2
  exit 1
}

ask() {
  if [ "${YATA_ASSUME_YES:-}" = "1" ]; then
    printf '%s [Y/n] y\n' "$1"
    return 0
  fi
  printf '%s [Y/n] ' "$1"
  read -r answer || answer=""
  case "$answer" in
    [nN]*) return 1 ;;
    *) return 0 ;;
  esac
}

find_uv() {
  for candidate in "$(command -v uv 2>/dev/null)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

system_cuda() {
  command -v ldconfig >/dev/null 2>&1 || return 1
  ldconfig -p 2>/dev/null | grep -q 'libcublas\.so\.12' && ldconfig -p 2>/dev/null | grep -q 'libcudnn\.so\.9'
}

OS=$(uname -s)
case "$OS" in
  Darwin)
    # The packages Yata uses (onnxruntime, PyAV) are built for Apple silicon and macOS 14 or later.
    [ "$(uname -m)" = "arm64" ] || fail "$(msg "この Mac（Intel）では Yata の部品が動きません。Apple シリコン（M1 以降）の Mac で使ってください。" \
      "Yata's parts do not run on this Mac (Intel). Use a Mac with Apple silicon (M1 or later).")"
    MAJOR=$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)
    [ "${MAJOR:-0}" -ge 14 ] 2>/dev/null || fail "$(msg "macOS 14（Sonoma）以降が必要です。今の macOS:" "Yata needs macOS 14 (Sonoma) or later. This Mac has macOS") $(sw_vers -productVersion 2>/dev/null)"
    ;;
  Linux) ;;
  *) fail "$(msg "この OS（$OS）には対応していません。Windows・macOS・Linux で使えます。" "This OS ($OS) is not supported. Yata runs on Windows, macOS and Linux.")" ;;
esac

msg "Yata を起動します。" "Starting Yata."; echo
echo ""

msg "[1/3] 準備ツール uv を確認しています..." "[1/3] Checking for the setup tool uv..."; echo
UV=$(find_uv) || UV=""
if [ -z "$UV" ]; then
  msg "Yata は、Python と必要な部品を入れるのに uv（Astral 社の無料ツール）を使います。この PC にはまだありません。" \
    "Yata uses uv (a free tool from Astral) to install Python and the parts it needs. This computer does not have it yet."; echo
  ask "$(msg "uv を今インストールしますか？（インターネットから astral.sh の公式インストーラを実行します）" \
    "Install uv now? (This runs the official installer from astral.sh over the internet.)")" ||
    fail "$(msg "uv がないため起動できません。https://docs.astral.sh/uv/ の手順で uv を入れてから、もう一度起動してください。" \
      "Yata cannot start without uv. Install uv as described at https://docs.astral.sh/uv/, then start Yata again.")"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    fail "$(msg "uv を入れるには curl か wget が必要です（Ubuntu なら: sudo apt install curl）。入れてから、もう一度起動してください。" \
      "Installing uv needs curl or wget (on Ubuntu: sudo apt install curl). Install one, then start Yata again.")"
  fi
  UV=$(find_uv) || fail "$(msg "uv をインストールできませんでした。インターネットにつながっているか確認して、もう一度起動してください。" \
    "Could not install uv. Check the internet connection and start Yata again.")"
fi

# Speaker labels (sherpa-onnx) are always included: small, and the setup page then needs no extra step.
EXTRAS="--extra speakers"
REMEMBER=""
if [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  if ls "$ROOT"/.venv/lib/python3*/site-packages/nvidia/cublas/lib/libcublas.so.12 >/dev/null 2>&1; then
    EXTRAS="$EXTRAS --extra cuda"
  elif ! system_cuda && [ ! -f "$ROOT/.venv/.yata-no-cuda" ]; then
    echo ""
    msg "NVIDIA の GPU が見つかりました。字幕を GPU で速く動かすには、NVIDIA の部品（cuBLAS と cuDNN、約 1.3 GB）が要ります。" \
      "An NVIDIA GPU was found. Fast captions on the GPU need NVIDIA's libraries (cuBLAS and cuDNN, about 1.3 GB)."; echo
    msg "入れない場合も、字幕は CPU で動きます（少し遅く、話している途中の字幕は出ません）。" \
      "Without them, captions run on the CPU (a little slower, and no captions while people speak)."; echo
    if ask "$(msg "GPU 用の部品も入れますか？" "Install the GPU libraries too?")"; then
      EXTRAS="$EXTRAS --extra cuda"
    else
      REMEMBER=1
    fi
  fi
fi

echo ""
msg "[2/3] Python と必要な部品を確認しています（初回は数分かかります）..." "[2/3] Checking Python and the parts Yata needs (a few minutes the first time)..."; echo
# --inexact keeps what is already in .venv (for example the optional Qwen3-ASR packages); --frozen installs
# exactly the versions in uv.lock.
PYTHON_ARG=""
[ -x "$ROOT/.venv/bin/python" ] || PYTHON_ARG="--python 3.12"
# shellcheck disable=SC2086  # EXTRAS and PYTHON_ARG are lists of words
"$UV" sync --frozen --inexact --no-dev $EXTRAS $PYTHON_ARG ||
  fail "$(msg "部品を入れられませんでした。インターネットにつながっているか確認して、もう一度起動してください。" \
    "Could not install the parts. Check the internet connection and start Yata again.")"
[ -n "$REMEMBER" ] && : > "$ROOT/.venv/.yata-no-cuda"

# The CUDA libraries from the cuda extra: Linux finds them through LD_LIBRARY_PATH, read when Python starts.
for dir in "$ROOT"/.venv/lib/python3*/site-packages/nvidia/cublas/lib "$ROOT"/.venv/lib/python3*/site-packages/nvidia/cudnn/lib; do
  [ -d "$dir" ] && LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
[ -n "${LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH

echo ""
msg "[3/3] Yata を起動します。準備ができると、ブラウザでページが開きます。" "[3/3] Starting Yata. The page opens in the browser when it is ready."; echo
msg "      ページは Google Chrome か Microsoft Edge で使ってください（タブの音声を共有するため）。" \
  "      Use the page in Google Chrome or Microsoft Edge (to share a tab's audio)."; echo
msg "      終わるときは、このウィンドウで Ctrl+C を押します。" "      To stop, press Ctrl+C in this window."; echo
echo ""
exec "$ROOT/.venv/bin/python" -B -m yata --open
