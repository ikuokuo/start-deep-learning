#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import os
import time
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

# Initialize model
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",   # HF repo or local path
    backend="transformers",                # or "vllm" for high-throughput inference
    # Inference/generation controls (applied across backends)
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

# Load image
image = Image.open("data/dog.jpg").convert("RGB")
print(f"Image size: {image.size[0]} x {image.size[1]}")

categories = ["dog", "car", "bicycle", "motorcycle",]

# Object Detection
t0 = time.time()
results = rex.inference(
    images=image,
    task="detection",
    categories=categories,
)
t1 = time.time()
time_infer = (t1 - t0) * 1000
print(f"Infer time: {time_infer:.3f} ms")

result = results[0]
if not result.get("success"):
    print("Infer failed:", result.get("raw_output"))
    raise SystemExit(1)

import json
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

# Visualize
os.makedirs("result", exist_ok=True)
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=20,
    draw_width=5,
    show_labels=True,
)
vis.save("result/dog.jpg")
