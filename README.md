# AB-Rec (TensorFlow 2.9) Reproduction Scaffold

This repository provides a faithful engineering scaffold to reproduce **AB-Rec** from:

> *Aligning and Balancing ID and Multimodal Representations for Recommendation*

It implements the two-stage pipeline:

1. **Multimodal item encoder fine-tuning** with three tasks.
2. **Dual-branch recommender** (ID branch + MM branch), trained with BCE + alignment losses + gradient modulation.

## Features implemented

- Stage-1 multimodal item encoder with:
  - content alignment objective (masked text + image -> full text)
  - metadata-to-description objective
  - multimodal robustness objective (partial text/visual masking)
- `[Item_cls]` token + hidden state extraction as item multimodal embedding
- Stage-2 AB-Rec model:
  - ID branch user/item embeddings
  - MM branch user embedding from mean of recent `k` item embeddings
  - same backbone architecture for both branches (MLP default)
  - prediction head: `concat([h_id, h_mm]) -> linear -> sigmoid`
- Losses:
  - BCE
  - Wasserstein-style alignment loss
  - cosine regularization
- Gradient modulation with contribution ratio `gamma` and coefficient `omega`
- Hyperparameter search space config for:
  - `alpha, beta, eta` in `{0.1, ..., 1.0}`
  - batch size in `{256, 512, 1024, 2048}`
  - learning rate in `{1e-3, 1e-4, 1e-5}`

## Paper-missing details explicitly exposed in config

- `data.recent_k`
- `model.id_embedding_dim`
- `model.mm_embedding_dim`
- `model.backbone_hidden_sizes`
- `mm_encoder.finetune.optimizer`
- `mm_encoder.finetune.learning_rate`
- `mm_encoder.finetune.mode` (`full` or `peft`)
- `mm_encoder.robustness.masking_ratio`
- `mm_encoder.preprocessing.image_size`
- `mm_encoder.preprocessing.video_num_frames`
- `losses.total_loss_weight`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train stage 1: multimodal item encoder

```bash
python scripts/train_mm_encoder.py --config configs/abrec.yaml
```

## Train stage 2: AB-Rec recommender

```bash
python scripts/train_abrec.py --config configs/abrec.yaml
```

## Run tests

```bash
pytest -q
```

## Notes

- The code is structured to stay fully in TensorFlow/Keras for AB-Rec training.
- Qwen2-VL-2B integration is modeled as a configurable encoder interface. In production, swap the internal encoder backend with your preferred Qwen2-VL checkpoint wiring while keeping losses and AB-Rec training identical.
