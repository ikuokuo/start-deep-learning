"""Falcon Perception — 精简推理脚本。

用法
----
    # 基本推理
    python perception.py --image photo.jpg --query "the red car"

    # 只要 bbox，跳过 upsampler
    python perception.py --image photo.jpg --query "all objects" --task detection

    # 降低分辨率 + 序列长度，省显存
    python perception.py --image photo.jpg --query "cat" --max-image-size 512 --max-seq-length 4096

    # bf16 + flex-attn-safe，小显存 GPU 可用
    python perception.py --image photo.jpg --query "dog" --flex-attn-safe --dtype bfloat16

    # 极限省显存：全开
    python perception.py --image photo.jpg --query "person" \
        --task detection --dtype bfloat16 --max-image-size 256 --max-seq-length 2048 \
        --flex-attn-safe --no-cudagraph

省显存参数
----------
    --flex-attn-safe    小显存 GPU (A40, RTX 3090/4090, L40) 避免 FlexAttention OOM
    --dtype bfloat16    模型显存减半 (默认 float32)
    --max-seq-length    降低序列长度 (默认 8192)
    --max-image-size    降低图片分辨率 (默认 1024)
    --task detection    跳过 HR upsampler, 只输出 bbox
    --no-cudagraph      不捕获 CUDA graph, 省显存 (推理略慢)
"""

import time
from pathlib import Path
from typing import Literal

import torch
import tyro

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    cuda_timed,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import ImageProcessor, load_image

setup_torch_config()


@torch.inference_mode()
def main(
    image: str,
    query: str = "all objects",
    task: Literal["segmentation", "detection"] = "segmentation",
    out_dir: str = "./outputs/",
    # 模型
    hf_model_id: str = PERCEPTION_MODEL_ID,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: Literal["bfloat16", "float32", "float"] = "float32",
    # 省显存
    flex_attn_safe: bool = False,
    max_seq_length: int = 8192,
    min_image_size: int = 256,
    max_image_size: int = 1024,
    # 推理
    compile: bool = True,
    cudagraph: bool = True,
):
    """Falcon Perception 推理：输入图像 + 自然语言 query → 检测/分割结果。"""
    # -- flex-attn-safe: 小 block 避免 Triton OOM
    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1} if flex_attn_safe else {}

    # -- 加载模型
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )

    if task == "segmentation" and not model_args.do_segmentation:
        print("模型不支持 segmentation, 回退到 detection。")
        task = "detection"

    # -- 加载图片
    pil_image = load_image(image).convert("RGB")
    w, h = pil_image.size
    print(f"Image : {w} x {h}")
    print(f"Task  : {task}")
    print(f"Query : {query!r}")

    # -- 构建引擎
    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]

    from falcon_perception.paged_inference import (
        PagedInferenceEngine,
        SamplingParams,
        Sequence,
    )

    engine = PagedInferenceEngine(
        model, tokenizer, image_processor,
        max_batch_size=2,
        max_seq_length=max_seq_length,
        n_pages=128,
        page_size=128,
        prefill_length_limit=max_seq_length,
        enable_hr_cache=False,
        capture_cudagraph=cudagraph,
        kernel_options=kernel_options or None,
    )

    # -- 构建序列
    prompt = build_prompt_for_task(query, task)
    sampling_params = SamplingParams(stop_token_ids=stop_token_ids)

    def _make_sequence():
        return [Sequence(
            text=prompt,
            image=pil_image,
            min_image_size=min_image_size,
            max_image_size=max_image_size,
            task=task,
        )]

    # -- Warmup (吸收 torch.compile JIT + CUDA graph capture)
    print("Warmup ...")
    with cuda_timed(reset_peak_memory=False) as t:
        engine.generate(_make_sequence(), sampling_params=sampling_params, use_tqdm=False, print_stats=False)
    print(f"Warmup done in {t.elapsed:.1f}s")

    # -- 推理
    print("Running inference ...")
    sequences = _make_sequence()
    t0 = time.time()
    engine.generate(sequences, sampling_params=sampling_params, use_tqdm=True, print_stats=True)
    time_infer = (time.time() - t0) * 1000

    # -- 输出结果
    from falcon_perception.visualization_utils import pair_bbox_entries, render_paged_inference_outputs

    seq = sequences[0]
    aux = seq.output_aux
    decoded = tokenizer.decode(seq.output_ids)

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"  Infer time : {time_infer:.0f} ms")
    print(f"  Prefill    : {seq.stats.prefill_ms:.0f} ms")
    print(f"  Decode     : {seq.stats.decode_wall_ms:.0f} ms ({seq.stats.decode_steps} steps)")
    print(f"  Finalize   : {seq.stats.finalize_ms:.0f} ms")
    if task == "segmentation":
        print(f"  Masks      : {len(aux.masks_rle)}")
    else:
        print(f"  Boxes      : {len(pair_bbox_entries(aux.bboxes_raw))}")
    print(f"  Decoded    : {decoded!r}")

    # -- 保存可视化
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    render_paged_inference_outputs(sequences, image_processor, output_dir=out_dir, task=task)
    print(f"\nOutdir : {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
