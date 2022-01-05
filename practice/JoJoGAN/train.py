#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
import os
import sys
from copy import deepcopy
import typing

import torch
from torchvision import utils

from tqdm import tqdm
import lpips
import wandb

from model import *
from util import *


device = "cuda"

from e4e_projection import projection
# from stylize_projection import StylizeProjection
# projection = StylizeProjection(device=device)


class StyleImage(typing.NamedTuple):
    path: str
    aligned: torch.Tensor
    latent: torch.Tensor

    aligned2target = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _load_style_image(img_path: str, aligned_dir: str, latent_dir: str) -> StyleImage:
    assert os.path.exists(img_path), f"{img_path} does not exist!"

    name, _ = os.path.splitext(os.path.basename(img_path))

    # crop and align the face
    style_aligned_path = os.path.join(aligned_dir, f'{name}.png')
    if not os.path.exists(style_aligned_path):
        style_aligned = align_face(img_path)
        style_aligned.save(style_aligned_path)
    else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')

    # GAN invert
    style_code_path = os.path.join(latent_dir, f'{name}.pt')
    if not os.path.exists(style_code_path):
        latent = projection(style_aligned, style_code_path, device)
    else:
        latent = torch.load(style_code_path)['latent']

    return StyleImage(
        path=img_path,
        aligned=style_aligned,
        latent=latent,
    )


def load_style_images(style_images: typing.List[str],
        style_images_aligned_dir: str, inversion_codes_dir: str) \
        -> typing.Tuple[torch.Tensor, torch.Tensor]:
    targets = []
    latents = []

    for style_path in style_images:
        style_image = _load_style_image(style_path, style_images_aligned_dir, inversion_codes_dir)
        targets.append(StyleImage.aligned2target(style_image.aligned).to(device))
        latents.append(style_image.latent.to(device))

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)

    return targets, latents


def load_test_input(filepath: str, test_temp_dir: str) -> typing.Optional[StyleImage]:
    if not os.path.exists(filepath):
        return None
    return _load_style_image(filepath, test_temp_dir, test_temp_dir)


def load_original_generator(net_path, latent_dim) -> Generator:
    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load(net_path, map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    return original_generator


class TrainOptions(typing.NamedTuple):
    original_generator: Generator
    alpha: float
    preserve_color: bool
    num_iter: int
    latent_dim: int
    use_wandb: bool
    log_interval: int
    test_input: StyleImage
    save_path: str


def train(targets: torch.Tensor, latents: torch.Tensor, options: TrainOptions) -> None:
    target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
    display_image(target_im, title='Style References')

    original_generator = options.original_generator
    alpha = options.alpha
    preserve_color = options.preserve_color
    num_iter = options.num_iter
    latent_dim = options.latent_dim
    use_wandb = options.use_wandb
    log_interval = options.log_interval
    test_input = options.test_input
    save_path = options.save_path

    if use_wandb:
        wandb.init(project="JoJoGAN")
        config = wandb.config
        config.num_iter = num_iter
        config.preserve_color = preserve_color
        wandb.log(
            {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
            step=0)

    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    generator = deepcopy(original_generator)

    g_optim = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    if preserve_color:
        id_swap = [7,9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))

    for idx in tqdm(range(num_iter)):
        # if preserve_color:
        #     random_alpha = 0
        # else:
        #     random_alpha = np.random.uniform(alpha, 1)
        mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim])
            .to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = alpha*latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)
        loss = lpips_fn(F.interpolate(img, size=(256,256), mode='area'),
            F.interpolate(targets, size=(256,256), mode='area')).mean()

        if use_wandb:
            wandb.log({"loss": loss}, step=idx)
            if idx % log_interval == 0 and test_input:
                generator.eval()
                my_sample = generator(test_input.latent.unsqueeze(0), input_is_latent=True)
                generator.train()
                my_sample = transforms.ToPILImage()(utils.make_grid(my_sample, normalize=True, range=(-1, 1)))
                wandb.log(
                    {"Current stylization": [wandb.Image(my_sample)]},
                    step=idx)

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    torch.save({"g": generator.state_dict()}, save_path)


def _parse_args() -> argparse.Namespace:
    def float_range(mini, maxi):
        def float_range_checker(arg):
            try:
                f = float(arg)
            except ValueError as e:
                raise argparse.ArgumentTypeError("must be a floating point number") from e
            if f < mini or f > maxi:
                raise argparse.ArgumentTypeError(f"must be in range [{mini}，{maxi}]")
            return f
        return float_range_checker

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="the device name: %(default)s")

    parser.add_argument("-n", "--name", default="unnamed",
        help="the style name: %(default)s")
    parser.add_argument("-i", "--images", default=[],
        action="extend", nargs="+",
        help="the style images: %(default)s")
    parser.add_argument("-o", "--outdir", default="output",
        help="the output directory of model and temporary files: %(default)s")

    parser.add_argument("--test", default="test_input/iu.jpeg",
        help="test input image for visualizing: %(default)s")
    parser.add_argument("--pretrained", default="models/stylegan2-ffhq-config-f.pt",
        help="pretrained network for finetuning: %(default)s")

    parser.add_argument("--alpha", default=1.0, type=float_range(0, 1),
        help="alpha controls the strength of the style: %(default)s")
    parser.add_argument("--preserve_color", action="store_true",
        help="tries to preserve color of original image by limiting family of allowable transformations: %(default)s")
    parser.add_argument("--num_iter", default=200, type=int,
        help="number of finetuning steps: %(default)s. "
             "Different style reference may require different iterations. Try 200~500 iterations.")
    parser.add_argument("--latent_dim", default=512, type=int,
        help="latent dim: %(default)s")
    parser.add_argument("--use_wandb", action="store_true",
        help="log training on wandb and interval for image logging: %(default)s")
    parser.add_argument("--log_interval", default=50, type=int,
        help="log interval: %(default)s")

    args = parser.parse_args()

    global device
    device = args.device

    if not args.images:
        sys.exit("images not given, could not train")
    if args.outdir:
        os.makedirs(args.outdir, mode=0o774, exist_ok=True)
    # if not os.path.isfile(args.test):
    #     sys.exit(f"test image not found: {args.test}")
    if not os.path.isfile(args.pretrained):
        sys.exit(f"pretrained network not found: {args.pretrained}")

    print("Args")
    print(f"  device: {args.device}")
    print(f"  name: {args.name}")
    print(f"  images: {args.images}")
    print(f"  outdir: {args.outdir}")
    print(f"  test: {args.test}")
    print(f"  pretrained: {args.pretrained}")
    print(f"  alpha: {args.alpha}")
    print(f"  preserve_color: {args.preserve_color}")
    print(f"  num_iter: {args.num_iter}")
    print(f"  latent_dim: {args.latent_dim}")
    print(f"  use_wandb: {args.use_wandb}")
    print(f"  log_interval: {args.log_interval}")

    return args


def _main():
    args = _parse_args()

    save_path = os.path.join(args.outdir, f"{args.name}.pt")
    if args.preserve_color:
        save_path = os.path.join(args.outdir, f"{args.name}_preserve_color.pt")
    if os.path.exists(save_path):
        s = input("save path already exists, override it? [Y/n] ")
        s = s.lower()
        if len(s) == 0 or s == "y":
            pass
        else:
            sys.exit()

    def outdir_sub(*p):
        sub = os.path.join(args.outdir, *p)
        os.makedirs(sub, exist_ok=True)
        return sub

    print("load_style_images ...")
    style_images_aligned_dir = outdir_sub(args.name, "style_images_aligned")
    inversion_codes_dir = outdir_sub(args.name, "inversion_codes")
    targets, latents = load_style_images(args.images, style_images_aligned_dir, inversion_codes_dir)
    print("load_style_images done")

    print("load_test_input ...")
    test_temp_dir = outdir_sub(args.name)
    test_input = load_test_input(args.test, test_temp_dir)
    print("load_test_input done")

    global projection
    del projection

    print("load_original_generator ...")
    original_generator = load_original_generator(args.pretrained, args.latent_dim)
    print("load_original_generator done")

    print("train ...")
    train(targets, latents, TrainOptions(
        original_generator=original_generator,
        alpha=1-args.alpha,
        preserve_color=args.preserve_color,
        num_iter=args.num_iter,
        latent_dim=args.latent_dim,
        use_wandb=args.use_wandb,
        log_interval=args.log_interval,
        test_input=test_input,
        save_path=save_path,
    ))
    print("train done")
    print(f"  save to: {save_path}")


if __name__ == "__main__":
    _main()
