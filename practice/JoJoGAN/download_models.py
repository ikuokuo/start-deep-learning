#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring
import argparse
from pathlib import Path

from gdown import download as drive_download


drive_ids = {
    "stylegan2-ffhq-config-f.pt": "1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "dlibshape_predictor_68_face_landmarks.dat": "11BDmNKS1zxSZxkgsEvQoKgFd8J264jKp",
    "e4e_ffhq_encode.pt": "1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "restyle_psp_ffhq_encode.pt": "1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "arcane_caitlyn.pt": "1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "arcane_caitlyn_preserve_color.pt": "1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "arcane_jinx_preserve_color.pt": "1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "arcane_jinx.pt": "1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "arcane_multi_preserve_color.pt": "1enJgrC08NpWpx2XGBmLt1laimjpGCyfl",
    "arcane_multi.pt": "15V9s09sgaw-zhKp116VHigf5FowAy43f",
    "disney.pt": "1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "disney_preserve_color.pt": "1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "jojo.pt": "13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "jojo_preserve_color.pt": "1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "jojo_yasuho.pt": "1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "jojo_yasuho_preserve_color.pt": "1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "supergirl.pt": "1L0y9IYgzLNzB-33xTpXpecsKU-t9DpVC",
    "supergirl_preserve_color.pt": "1VmKGuvThWHym7YuayXxjv0fSn32lfDpE",
    "art.pt": "1a0QDEHwXQ6hE_FcYEyNMuv5r5UnRQLKT",
}


def drive_path(drive_id: str) -> str:
    return f"https://drive.google.com/uc?id={drive_id}"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="models",
        help="the download directory: %(default)s")

    args = parser.parse_args()

    print("Args")
    print(f"  dir: {args.dir}")

    return args


def _main():
    args = _parse_args()

    print()
    models_dir = Path(args.dir)
    print(f"models_dir: {models_dir}")
    models_dir.mkdir(mode=0o774, parents=True, exist_ok=True)
    print()

    for i, (name, drive_id) in enumerate(drive_ids.items()):
        drive_url = drive_path(drive_id)
        model_path = models_dir / name

        print(f"{i:2d} {name}")
        print(f"  << {drive_url}")
        print(f"  >> {model_path} {'[exists]' if model_path.exists() else ''}")

        if model_path.exists():
            continue

        drive_download(drive_url, str(model_path), quiet=False)


if __name__ == "__main__":
    _main()
