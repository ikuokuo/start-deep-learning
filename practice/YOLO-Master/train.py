#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
from ultralytics import YOLO

# 加载模型，从 YAML 构建（从零开始）
# model = YOLO('cfg/models/master/v0/det/yolo-master-n.yaml')
# 加载模型，Nano 版（全量微调）
model = YOLO("yolo_master_n.pt")

# 训练模型，用 COCO8 数据集测试
#   结果在 runs/detect/train10/weights/best.pt
results = model.train(
    data='coco8.yaml',  # data='coco.yaml',
    epochs=100,         # epochs=600,
    batch=8,            # batch=256,
    imgsz=640,
    device="0",         # device="0,1,2,3",  # 如果使用多 GPU
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1
)

# 检测图像
results = model("data/dog.jpg")
results[0].show()
