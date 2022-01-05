#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os
import typing

import torch
from torchvision import utils

from model import *


device = "cuda"
models_dir = "models"


class StyleGANOptions(typing.NamedTuple):
    num: int
    seed: int
    ext: str
    output_dir: str


class StyleGAN:

    def __init__(self) -> None:
        self._latent_dim = 512
        self._init_generator()

    def _init_generator(self):
        original_generator = Generator(1024, self._latent_dim, 8, 2).to(device)
        ckpt = torch.load(f"{models_dir}/stylegan2-ffhq-config-f.pt",
            map_location=lambda storage, loc: storage)
        original_generator.load_state_dict(ckpt["g_ema"], strict=False)

        self._original_generator = original_generator
        self._mean_latent = original_generator.mean_latent(10000)

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, options: StyleGANOptions) -> None:
        if not options.output_dir:
            return None

        n_sample = options.num
        seed = options.seed

        torch.manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(n_sample, self._latent_dim, device=device)
            original_sample = self._original_generator([z], truncation=0.7, truncation_latent=self._mean_latent)

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))
            return img

        # print(original_sample.shape)
        for i, sample in enumerate(original_sample):
            print(f"{i:3d} {sample.shape}")

            save_path = os.path.join(options.output_dir, f"{i}{options.ext}")
            img = norm_ip(sample, -1, 1)
            utils.save_image(img, save_path)

            print(f"    > {save_path}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="the device name: %(default)s")

    parser.add_argument("-n", "--num", default=5, type=int,
        help="the number of output images: %(default)s")
    parser.add_argument("-s", "--seed", default=3000, type=int,
        help="the seed for random: %(default)s")
    parser.add_argument("-e", "--ext", default=".jpeg",
        help="the extension of output images: %(default)s")
    parser.add_argument("-o", "--output_dir", default="input",
        help="the output directory: %(default)s")

    args = parser.parse_args()

    global device
    device = args.device

    if args.output_dir:
        os.makedirs(args.output_dir, mode=0o774, exist_ok=True)

    print("Args")
    print(f"  device: {args.device}")
    print(f"  num: {args.num}")
    print(f"  seed: {args.seed}")
    print(f"  ext: {args.ext}")
    print(f"  output_dir: {args.output_dir}")

    return args


def _main():
    args = _parse_args()

    stylegan = StyleGAN()

    print("stylegan ...")
    stylegan(StyleGANOptions(
        num=args.num,
        seed=args.seed,
        ext=args.ext,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    _main()
