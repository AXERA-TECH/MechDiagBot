#!/usr/bin/env bash
# 下载 sherpa-onnx 中文 KWS 唤醒模型到 models/。
# 用法：./scripts/download_kws_model.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
MODEL_DIR="${KWS_MODEL_DIR:-$ROOT/models/$MODEL_NAME}"
URL="${KWS_MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$MODEL_NAME.tar.bz2}"

required=(
  tokens.txt
  encoder-epoch-12-avg-2-chunk-16-left-64.onnx
  decoder-epoch-12-avg-2-chunk-16-left-64.onnx
  joiner-epoch-12-avg-2-chunk-16-left-64.onnx
)

missing=0
for name in "${required[@]}"; do
  if [[ ! -f "$MODEL_DIR/$name" ]]; then
    missing=1
    break
  fi
done
if [[ "$missing" -eq 0 ]]; then
  echo "模型已存在：$MODEL_DIR"
  exit 0
fi

command -v curl >/dev/null 2>&1 || {
  echo "缺少 curl，无法下载模型" >&2
  exit 1
}
command -v tar >/dev/null 2>&1 || {
  echo "缺少 tar，无法解压模型" >&2
  exit 1
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "下载 $URL"
curl -L --fail --retry 3 -o "$tmp/model.tar.bz2" "$URL"
tar -xjf "$tmp/model.tar.bz2" -C "$tmp"

mkdir -p "$MODEL_DIR"
for name in "${required[@]}"; do
  cp "$tmp/$MODEL_NAME/$name" "$MODEL_DIR/$name"
done
cp "$tmp/$MODEL_NAME/README.md" "$MODEL_DIR/README.md" 2>/dev/null || true

echo "完成：$MODEL_DIR"
