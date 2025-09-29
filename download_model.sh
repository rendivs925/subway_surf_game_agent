#!/usr/bin/env bash
set -e

mkdir -p models

echo "Downloading YOLOv8n ONNX model..."
wget -O models/subway_surfers.onnx \
  https://huggingface.co/unity/inference-engine-yolo/resolve/ed7f4daf9263d0d31be1d60b9d67c8baea721d60/yolov8n.onnx

if [ -f models/subway_surfers.onnx ]; then
  echo "✅ Download succeeded: models/subway_surfers.onnx"
  file models/subway_surfers.onnx
else
  echo "❌ Download failed!"
  exit 1
fi
