#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
from ultralytics import YOLO

# 加载模型，Nano 版
model = YOLO("yolo26n.pt")
# model = YOLO("runs/detect/train/weights/best.pt")

# 检测图像
results = model.predict("data/dog.jpg")
results[0].show()

# 保存结果
results[0].save("result/dog.jpg")
