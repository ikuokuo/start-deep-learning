#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os
import sys
from enum import Enum, auto
from typing import Optional, Tuple

import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Precision(Enum):
    FLOAT16 = auto()
    FLOAT32 = auto()


def _infer(model_path: str,
           input_image: str,
           input_resize: Optional[Tuple[int, int]] = None,
           output_image: Optional[str] = None,
           precision: Precision = Precision.FLOAT16,
           do_show: bool = True) -> None:
    # Read input
    with Image.open(input_image) as img:
        img.load()
    if input_resize is not None:
        img = img.resize(input_resize, Image.ANTIALIAS)

    if do_show:
        plt.imshow(img)
        plt.show()

    src_type = np.float16
    if precision == Precision.FLOAT32:
        src_type = np.float32

    # HWC [0,255] > BCHW [0,1]
    src = np.array(img)
    src = np.moveaxis(src, -1, 0).astype(src_type)
    src = src[np.newaxis, :] / 255.

    # Load model
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    # Create an io binding
    io = sess.io_binding()

    # Create tensors on CUDA
    rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=src_type), 'cuda') ] * 4
    downsample_ratio_value = auto_downsample_ratio(img.height, img.width)
    downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([downsample_ratio_value], dtype=np.float32), 'cuda')

    # Set output binding
    for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
        io.bind_output(name, 'cuda')

    # Inference
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()

    # Only transfer `fgr` and `pha` to CPU.
    fgr = fgr.numpy()
    pha = pha.numpy()

    com = np.where(pha > 0, fgr, pha)
    com = np.concatenate([com, pha], axis=1) # + alpha

    # BCHW [0,1] > HWC [0,255]
    img = np.squeeze(com, axis=0)
    img = np.moveaxis(img, 0, -1) * 255
    img = img.astype(np.uint8)

    if do_show:
        plt.imshow(img)
        plt.show()

    # Save output
    if output_image:
        img = Image.fromarray(img)
        img.save(output_image)
        print(f'save to {output_image}')


def auto_downsample_ratio(h, w):
    return min(512 / max(h, w), 1)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input-image', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--output-image', type=str)
    parser.add_argument('--precision', type=str,
        default=Precision.FLOAT16.name.lower(),
        choices=tuple(p.name.lower() for p in Precision))
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f'onnx model not found: {args.model}')
    if not os.path.isfile(args.input_image):
        sys.exit(f'input image not found: {args.input_image}')
    if not args.output_image:
        root, _ = os.path.splitext(args.input_image)
        args.output_image = f'{root}_com.png'

    args.precision = Precision[args.precision.upper()]

    print('Args')
    print(f'  model: {args.model}')
    print(f'  input_image: {args.input_image}')
    print(f'  input_resize: {args.input_resize}')
    print(f'  output_image: {args.output_image}')
    print(f'  precision: {args.precision}')
    print(f'  show: {args.show}')

    return args


def _main():
    args = _parse_args()
    _infer(
        model_path=args.model,
        input_image=args.input_image,
        input_resize=args.input_resize,
        output_image=args.output_image,
        precision=args.precision,
        do_show=args.show
    )


if __name__ == '__main__':
    _main()
