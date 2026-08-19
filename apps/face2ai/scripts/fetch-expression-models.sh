#!/usr/bin/env bash
# Fetch BOTH expression model assets up front so Face2AI never touches the network at startup:
#  1. MediaPipe Face Landmarker (3.7 MB) → $FACE2AI_EXPRESSION_MODELS_DIR
#     (default: $FACE2AI_DATA_DIR/models, i.e. ~/.face2ai/models — same derivation as Settings.from_env).
#  2. EmotiEffLib enet_b0_8_va_mtl.onnx (16 MB) → ~/.emotiefflib/ — the exact cache path emotiefflib 1.1.1
#     resolves (utils.get_model_path_onnx: os.path.expanduser("~")/.emotiefflib/<model>.onnx); the adapter
#     refuses to construct the recognizer when this file is missing instead of letting emotiefflib download it.
# Idempotent: existing files are kept. Each download is fail-fast (-f) and atomic (.part, then mv).
set -euo pipefail

fetch() {  # fetch <url> <target-path>
  local url="$1" target="$2"
  mkdir -p "$(dirname "$target")"
  if [ ! -f "$target" ]; then
    curl -fsSL -o "$target.part" "$url" && mv "$target.part" "$target"
  fi
  ls -la "$target"
}

DIR="${FACE2AI_EXPRESSION_MODELS_DIR:-${FACE2AI_DATA_DIR:-$HOME/.face2ai}/models}"
fetch "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
      "$DIR/face_landmarker.task"

EMOTIEFF_MODEL="enet_b0_8_va_mtl"
fetch "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/models/affectnet_emotions/onnx/${EMOTIEFF_MODEL}.onnx?raw=true" \
      "$HOME/.emotiefflib/${EMOTIEFF_MODEL}.onnx"
