"""Falcon OCR — 精简推理脚本。

用法
----
    # 全页 OCR (低延迟, 文本不密集时推荐)
    python ocr.py --image document.png

    # 版面分析 + 区域级 OCR (密集文本/复杂版面推荐)
    python ocr.py --image document.png --task ocr_layout

    # 公式识别 (输出 LaTeX)
    python ocr.py --image formula.png --category formula

    # 表格识别 (输出 HTML)
    python ocr.py --image table.png --category table

    # 极限省显存：全开 (RTX 2060 等 6GB GPU)
    python ocr.py --image document.png \
        --dtype bfloat16 --max-image-size 512 --max-seq-length 4096 \
        --n-pages 32 --page-size 128 \
        --flex-attn-safe --no-cudagraph

省显存参数
----------
    --flex-attn-safe    小显存 GPU (A40, RTX 3090/4090, L40) 避免 FlexAttention OOM
    --dtype bfloat16    模型显存减半 (默认 float32)
    --max-seq-length    降低序列长度 (默认 8192)
    --max-image-size    降低图片分辨率 (默认 1024)
    --n-pages           KV-cache 页数, 降低可大幅省显存 (默认 1536)
    --page-size         每页 token 数 (默认 128)
    --no-cudagraph      不捕获 CUDA graph, 省显存 (推理略慢)
"""

from pathlib import Path
from typing import Literal

import torch
import tyro

from falcon_perception import OCR_MODEL_ID, cuda_timed, load_and_prepare_model, setup_torch_config
from falcon_perception.data import ImageProcessor, load_image

setup_torch_config()


def visualize_layout(image, detections, save_path, max_dim=1024):
    """保存版面检测可视化结果。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    from PIL import Image as PILImage
    w_img, h_img = image.size
    scale = max_dim / max(w_img, h_img)
    fig_w, fig_h = w_img * scale / 100, h_img * scale / 100

    categories = sorted(set(d["category"] for d in detections))
    palette = list(mcolors.TABLEAU_COLORS.values())
    cat_to_color = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image)

    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det["bbox"]
        w, h = x1 - x0, y1 - y0
        color = cat_to_color[det["category"]]
        rgb = mcolors.to_rgb(color)

        rect = patches.Rectangle((x0, y0), w, h, linewidth=1.5, edgecolor=color, facecolor=(*rgb, 0.3))
        ax.add_patch(rect)
        label = f'{i}:{det["category"]}:{int(det["score"] * 100)}'
        ax.text(x0 - 2, y0 - 4, label, fontsize=9,
                color="white", backgroundcolor=color, fontweight="bold",
                va="bottom", ha="left")

    ax.set_axis_off()
    ax.set_title("Layout Detection", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Layout vis : {save_path}")


@torch.inference_mode()
def main(
    image: str,
    task: Literal["ocr_plain", "ocr_layout"] = "ocr_plain",
    category: Literal["text", "formula", "table"] = "text",
    out_dir: str = "./outputs/",
    # 模型
    hf_model_id: str = OCR_MODEL_ID,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: Literal["bfloat16", "float32", "float"] = "float32",
    # 省显存
    flex_attn_safe: bool = False,
    max_seq_length: int = 8192,
    max_image_size: int = 1024,
    n_pages: int = 1536,
    page_size: int = 128,
    # 推理
    compile: bool = True,
    cudagraph: bool = True,
):
    """Falcon OCR 推理：输入图像 → 识别文本/公式/表格。"""
    # -- flex-attn-safe: 小 block 避免 Triton OOM
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1} if flex_attn_safe else None

    # -- 加载模型
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )

    # -- 加载图片
    pil_image = load_image(image).convert("RGB")
    w, h = pil_image.size
    print(f"Image    : {w} x {h}")
    print(f"Task     : {task}")
    print(f"Category : {category}")

    # -- 构建引擎
    image_processor = ImageProcessor(patch_size=16, merge_size=1)

    from falcon_perception.paged_ocr_inference import OCRInferenceEngine

    engine = OCRInferenceEngine(
        model, tokenizer, image_processor,
        max_seq_length=max_seq_length,
        n_pages=n_pages,
        page_size=page_size,
        capture_cudagraph=cudagraph,
        kernel_options=kernel_options,
    )

    # -- Warmup
    print("Warmup ...")
    with cuda_timed(reset_peak_memory=False) as t:
        engine.generate_plain([pil_image], use_tqdm=False)
    print(f"Warmup done in {t.elapsed:.1f}s")

    # -- 推理
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Running inference ...")

    if task == "ocr_layout":
        with cuda_timed() as t:
            results = engine.generate_with_layout(images=[pil_image], use_tqdm=True)

        elements = results[0]
        full_text = "\n".join(e["text"] for e in elements if e["text"])

        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}")
        print(f"  Infer time : {t.elapsed * 1000:.0f} ms")
        print(f"  Regions    : {len(elements)}")
        print(f"{'=' * 60}")
        for i, elem in enumerate(elements):
            preview = (elem["text"][:80] + "...") if len(elem["text"]) > 80 else elem["text"]
            print(f"  [{i}] {elem['category']}  score={elem['score']:.3f}  {preview}")
        print(f"\n{full_text}")

        # -- 保存版面检测可视化
        visualize_layout(pil_image, elements, str(out_path / "layout.jpg"))

    else:  # ocr_plain
        with cuda_timed() as t:
            texts = engine.generate_plain(images=[pil_image], category=category, use_tqdm=True)

        text = texts[0] if texts else ""

        print(f"\n{'=' * 60}")
        print("Results")
        print(f"{'=' * 60}")
        print(f"  Infer time : {t.elapsed * 1000:.0f} ms")
        print(f"{'=' * 60}")
        print(text)

    print(f"\nOutput : {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
