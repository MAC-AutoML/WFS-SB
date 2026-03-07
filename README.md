<div align="center">

# 🌊 WFS-SB

### Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding

<p>
  Open-source implementation of <strong>WFS-SB</strong>, a training-free frame selection framework for long-video understanding with LVLMs.
</p>

<p>
  <a href="https://arxiv.org/abs/2603.00512">
    <img src="https://img.shields.io/badge/ArXiv-2603.00512-b31b1b.svg" alt="ArXiv">
  </a>
  <img src="https://img.shields.io/badge/Task-Long_Video_Understanding-2563eb.svg" alt="Task">
  <img src="https://img.shields.io/badge/Method-Training--Free-16a34a.svg" alt="Method">
  <img src="https://img.shields.io/badge/Benchmarks-VideoMME%20%7C%20MLVU%20%7C%20LVB-f59e0b.svg" alt="Benchmarks">
</p>

<p>
  <a href="https://arxiv.org/abs/2603.00512">Paper</a> ·
  <a href="#highlights">Highlights</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#full-pipeline-workflow">Full Pipeline Workflow</a> ·
  <a href="#project-structure">Project Structure</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

Long videos contain heavy frame redundancy, while Large Vision-Language Models (LVLMs) operate under limited context budgets. Most query-aware frame selection methods focus only on frame relevance, which often yields fragmented visual evidence and ignores the video's narrative structure.

**WFS-SB** addresses this issue by detecting **semantic boundaries** in the query-frame similarity signal. It first uses **wavelet-based multi-resolution analysis** to suppress high-frequency noise, then identifies boundary points that divide a video into coherent clips. Based on these clips, WFS-SB allocates the frame budget adaptively and selects frames with **Maximal Marginal Relevance (MMR)** to preserve both relevance and diversity.

<p align="center">
  <img src="assert/method_v2_01.png" alt="WFS-SB framework" width="96%" />
</p>

## Highlights

- 🌟 **Training-free pipeline** that plugs into long-video LVLM inference without extra model training.
- 🌊 **Wavelet-based denoising** helps recover robust semantic change signals from noisy query-frame similarities.
- 🧩 **Two-stage selection strategy** combines clip-level budget allocation with within-clip MMR sampling.
- 📈 **Strong reported gains** over prior frame selection strategies on three long-video benchmarks.

## News

- [2026-02-21] 🎉 Our paper was accepted to **CVPR 2026**.

## Repository Contents

- 🎞️ `preprocess/`: frame sampling, feature extraction, and query-frame similarity scoring.
- 🧠 `wfs/`: the unified WFS pipeline for VideoMME, LongVideoBench, and MLVU.
- 📁 `datasets/`: annotation files and reproduction keyframe JSONs.
- 🩹 `lmms-eval-diff/`: `lmms-eval` patch artifacts and integration notes.

## Quick Start

### 1. Environment

This repository provides the WFS code and the patch artifacts for `lmms-eval`. If you do not already have a compatible `lmms-eval` checkout under the repository root, prepare it first and then install the environment.

```bash
# Prepare a compatible lmms-eval checkout
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
git checkout bb1ebe76e7a942386c25c4664f902e0e59e8a401
git apply ../lmms-eval-diff/lmms_eval_wfs.patch
cd ..

# Create and activate the environment
conda create -n wfs python=3.10 -y
conda activate wfs

# Install dependencies
pip install -e ./lmms-eval
pip install -r requirements.txt
```

For FlashAttention 2, install a wheel that matches your local `Python`, `PyTorch`, and `CUDA` versions. For the environment in `requirements.txt`, choose a wheel built for `Python 3.10` and `Torch 2.6`.

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0/flash_attn-2.6.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

See the official FlashAttention releases page for a matching build: https://github.com/Dao-AILab/flash-attention/releases

If you prefer to rebuild `lmms-eval` from upstream and apply the patch manually, see `lmms-eval-diff/README.md`.

### 2. Dataset Preparation

This repository includes **annotation files** and **reproduction keyframe JSONs**, but it does **not** include the raw benchmark videos. Please download the videos from the official dataset sources and place them in the expected directories.

Organize the datasets as follows:

```text
datasets/
├── videomme/
│   ├── data/                                  # Put VideoMME .mp4 files here
│   ├── videomme_json_file.json
│   └── keyframe_dir/
│       ├── reproduce_videomme_f8.json
│       ├── reproduce_videomme_f16.json
│       └── reproduce_videomme_f32.json
├── longvideobench/
│   ├── videos/                                # Put LongVideoBench .mp4 files here
│   ├── lvb_val.json
│   └── keyframe_dir/
│       ├── reproduce_lvb_f8.json
│       ├── reproduce_lvb_f16.json
│       └── reproduce_lvb_f32.json
└── mlvu/
    ├── video/                                 # Put MLVU .mp4 files here
    ├── mlvu_dev.json
    └── keyframe_dir/
        ├── reproduce_mlvu_f8.json
        ├── reproduce_mlvu_f16.json
        └── reproduce_mlvu_f32.json
```

If your local paths differ, update `configs/dataset_paths.example.yaml` accordingly.

### 3. Quick Inference

After the raw videos are in place, you can directly reproduce inference results with the provided keyframe JSON files.

**Uniform baseline**

```bash
export QWEN_CKPT=Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model qwen2_5_vl \
  --tasks videomme \
  --model_args max_num_frames=16,pretrained=${QWEN_CKPT},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
  --batch_size 1 \
  --output_path ./results/videomme/uni
```

**WFS reproduction JSONs**

```bash
export QWEN_CKPT=Qwen/Qwen2.5-VL-7B-Instruct

# VideoMME, K=16
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model qwen2_5_vl \
  --tasks videomme \
  --model_args max_num_frames=16,use_keyframe=True,pretrained=${QWEN_CKPT},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
  --batch_size 1 \
  --output_path ./results/videomme/ \
  --data_files '{"test": "keyframe_dir/reproduce_videomme_f16.json"}'

# LongVideoBench, K=16
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model qwen2_5_vl \
  --tasks longvideobench_val_v \
  --model_args max_num_frames=16,use_keyframe=True,pretrained=${QWEN_CKPT},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
  --batch_size 1 \
  --output_path ./results/longvideobench_val_v/ \
  --data_files '{"validation": "keyframe_dir/reproduce_lvb_f16.json"}'

# MLVU, K=16
CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model qwen2_5_vl \
  --tasks mlvu_dev \
  --model_args max_num_frames=16,use_keyframe=True,pretrained=${QWEN_CKPT},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
  --batch_size 1 \
  --output_path ./results/mlvu_dev/ \
  --data_files '{"test": "keyframe_dir/reproduce_mlvu_f16.json"}'
```

For additional model examples, refer to the official `lmms-eval` examples: https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/examples

## Full Pipeline Workflow

The full WFS workflow consists of three steps:

1. 🎞️ Extract frame-level features and query-frame similarity scores.
2. 🌊 Run WFS to generate keyframe JSON files with `keyframe_indices`.
3. 🤖 Feed the generated JSONs into `lmms-eval` for LVLM inference.

The unified pipeline currently supports:

- **Benchmarks**: `videomme`, `lvb`, `mlvu`
- **Feature models**: `blip2`, `blip1`, `clip`, `siglip`

### 1. Preprocessing

Extract frame-level features and similarity scores before running WFS.

Example: **VideoMME + BLIP2**

```bash
python -m preprocess.extract \
  --benchmark videomme \
  --feature_model blip2 \
  --dataset_root datasets/videomme \
  --json_file datasets/videomme/videomme_json_file.json \
  --output_dir datasets/videomme/blip2_features_and_scores \
  --device cuda \
  --batch_size 256 \
  --sample_fps 1.0
```

### 2. WFS Frame Selection

Run the WFS pipeline to generate a keyframe JSON file containing `keyframe_indices`.

```bash
python -m wfs.pipeline \
  --benchmark videomme \
  --feature_model blip2 \
  --max_frames 16 \
  --dataset_root datasets/videomme \
  --questions_file datasets/videomme/videomme_json_file.json \
  --features_dir datasets/videomme/blip2_features_and_scores \
  --output_path datasets/videomme/keyframe_dir/WFS_videomme_blip2_16f.json
```

For the other benchmarks, replace the dataset-specific paths accordingly:

- `lvb`: `datasets/longvideobench/lvb_val.json`
- `mlvu`: `datasets/mlvu/mlvu_dev.json`

### 3. LVLM Inference

Run `lmms-eval` with `qwen2_5_vl` and the generated keyframe JSON.

```bash
export QWEN_CKPT=Qwen/Qwen2.5-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m lmms_eval \
  --model qwen2_5_vl \
  --tasks videomme \
  --model_args max_num_frames=16,use_keyframe=True,pretrained=${QWEN_CKPT},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
  --batch_size 1 \
  --output_path ./results/videomme/ \
  --data_files '{"test": "keyframe_dir/WFS_videomme_blip2_16f.json"}'
```

## Project Structure

```text
WFS-OpenSource/
├── configs/
│   ├── dataset_paths.example.yaml
│   └── wfs_defaults.yaml
├── datasets/
│   ├── videomme/
│   ├── longvideobench/
│   └── mlvu/
├── lmms-eval-diff/
│   ├── README.md
│   ├── lmms_eval_wfs.patch
│   └── modified_files/
├── preprocess/
│   └── extract.py
├── run_qwen2_5_vl_lmms_eval_reproduce.sh
├── requirements.txt
└── wfs/
    ├── benchmarks.py
    ├── core.py
    └── pipeline.py
```

## Citation

If you find this project useful, please cite our paper:

```bibtex
@article{chen2026wavelet,
  title={Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding},
  author={Chen, Wang and Zeng, Yuhui and Luo, Yongdong and Xie, Tianyu and Lin, Luojun and Ji, Jiayi and Zhang, Yan and Zheng, Xiawu},
  journal={arXiv preprint arXiv:2603.00512},
  year={2026}
}
```
