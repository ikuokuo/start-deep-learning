#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import os
import time
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

# Initialize model
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni-AWQ",
    backend="vllm",
    quantization="awq",
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
    # vLLM params adjusted as below to reduce HBM usage
    max_pixels=1280*28*28,      # 降低输入尺寸，减少显存占用
    max_model_len=2048,         # 降低上下文长度，减少显存占用
    max_num_seqs=1,             # 单序列处理，最省显存
    enforce_eager=True,         # 禁用CUDA graph，牺牲速度换显存
    gpu_memory_utilization=0.6, # 控制显存使用率，避免OOM
    limit_mm_per_prompt={"image": 1, "video": 0},
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
