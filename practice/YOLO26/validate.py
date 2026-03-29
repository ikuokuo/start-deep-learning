#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
from ultralytics import YOLO

# 加载模型，Nano 版
model = YOLO("yolo26n.pt")

# 验证精度，用 COCO128 数据集
# metrics = model.val(data="coco.yaml")
metrics = model.val(data="coco128.yaml")
print(f"mAP50-95 = {metrics.box.map}")
