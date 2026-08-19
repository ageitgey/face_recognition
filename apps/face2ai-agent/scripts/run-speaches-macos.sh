#!/usr/bin/env bash
# Run speaches (OpenAI-compatible STT + TTS: faster-whisper, Kokoro, Piper) natively on macOS
# without Docker. Verified on Apple Silicon (M1, macOS 26) on 2026-08-17.
#
# Why these knobs:
#   - speaches pins uv ~=0.10 -> we run it through `uvx uv@0.10` so your global uv is untouched.
#   - piper-tts (>=1.3) macOS wheels look for espeak-ng data at a build-machine path; espeak-ng
#     reads ESPEAK_DATA_PATH as the *parent* of `espeak-ng-data` -> `brew install espeak-ng`.
#   - onnxruntime's CoreML provider crashes the server on longer Piper sentences -> exclude it.
set -euo pipefail

SPEACHES_DIR="${SPEACHES_DIR:-$HOME/.cache/face2ai/speaches}"
# Own Hugging Face cache: speaches scans every repo in the cache and trips over unrelated ones
# (e.g. LiveKit's turn-detector) -> keep its models separate. speaches-cli downloads into it via the server.
SPEACHES_HF_CACHE="${SPEACHES_HF_CACHE:-$HOME/.cache/face2ai/hf-hub}"
PORT="${SPEACHES_PORT:-8000}"
ESPEAK_SHARE="${ESPEAK_SHARE:-$(brew --prefix 2>/dev/null || echo /opt/homebrew)/share}"

if [ ! -f "$ESPEAK_SHARE/espeak-ng-data/phontab" ]; then
  echo "espeak-ng data not found under $ESPEAK_SHARE — run: brew install espeak-ng" >&2
  exit 1
fi
if [ ! -d "$SPEACHES_DIR/.git" ]; then
  git clone --depth 1 https://github.com/speaches-ai/speaches.git "$SPEACHES_DIR"
fi
cd "$SPEACHES_DIR"
uvx 'uv@0.10' sync
uvx 'uv@0.10' pip install 'piper-tts>=1.7.0' >/dev/null

echo "speaches on http://127.0.0.1:$PORT — models: uvx speaches-cli model download <id>"
echo "  STT: Systran/faster-whisper-small        TTS de: speaches-ai/piper-de_DE-thorsten-medium   TTS en: speaches-ai/Kokoro-82M-v1.0-ONNX"
mkdir -p "$SPEACHES_HF_CACHE"
HF_HUB_CACHE="$SPEACHES_HF_CACHE" \
ESPEAK_DATA_PATH="$ESPEAK_SHARE" \
UNSTABLE_ORT_OPTS__EXCLUDE_PROVIDERS='["TensorrtExecutionProvider","CoreMLExecutionProvider"]' \
exec uvx 'uv@0.10' run --no-sync uvicorn --factory --host 127.0.0.1 --port "$PORT" speaches.main:create_app
