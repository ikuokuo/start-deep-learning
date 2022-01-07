# [RVM][]

[RVM]: https://github.com/PeterL1n/RobustVideoMatting

## Install

### Env

- [Anaconda](https://www.anaconda.com/products/individual)
- [PyTorch](https://pytorch.org/get-started/locally/)

```bash
conda create -n torch python=3.9 -y
conda activate torch

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
```

### RVM

```bash
git clone https://github.com/PeterL1n/RobustVideoMatting.git
```

## Inference using PyTorch

- model: [rvm_mobilenetv3.pth](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth)
- video: [spiderman.mp4](https://drive.google.com/drive/folders/1VFnWwuu-YXDKG-N6vcjK_nL7YZMFapMU?usp=sharing)

```bash
cd RobustVideoMatting

# pip install -r requirements_inference.txt
pip install av pims

# using official inference code
python inference.py \
--variant mobilenetv3 \
--checkpoint "rvm_mobilenetv3.pth" \
--device cuda \
--input-source "spiderman.mp4" \
--output-type video \
--output-composition "spiderman_com.mp4" \
--output-video-mbps 4 \
--seq-chunk 1
```

<!--
python inference.py \
--variant mobilenetv3 \
--checkpoint "rvm_mobilenetv3.pth" \
--device cuda \
--input-source "input" \
--output-type png_sequence \
--output-composition "output" \
--seq-chunk 1
-->

## Inference using ONNX

- model: [rvm_mobilenetv3_fp16.onnx](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.onnx)
- image: [input.jpg](input.jpg)

```bash
cd start-deep-learning/practice/RVM

# Requirements
#  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
pip install onnxruntime-gpu=1.10

# Inference / ONNX
#  https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md#onnx
python rvm_onnx_infer.py --model "rvm_mobilenetv3_fp16.onnx" --input-image "input.jpg" --show
```

Input:

![](input.jpg)

Output:

![](input_com.png)
