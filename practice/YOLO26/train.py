#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
from ultralytics import YOLO

# 加载模型，Nano 版
model = YOLO("yolo26n.pt")

# 训练模型，用 COCO8 数据集测试
#   结果在 runs/detect/train/weights/best.pt
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# 检测图像
results = model("data/dog.jpg")
results[0].show()
