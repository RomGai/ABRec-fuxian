#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import tensorflow as tf

from abrec.config import load_config
from abrec.models.mm_item_encoder import MultimodalItemEncoder, mm_pretrain_loss


def random_mm_batch(batch_size: int, cfg):
    mm_cfg = cfg.mm_encoder
    text_len = mm_cfg["max_text_len"]
    metadata_len = mm_cfg["metadata_len"]
    vocab = mm_cfg["vocab_size"]
    visual_dim = mm_cfg["visual_token_dim"]

    return {
        "text_tokens": tf.random.uniform((batch_size, text_len), maxval=vocab, dtype=tf.int32),
        "full_text_tokens": tf.random.uniform((batch_size, text_len), maxval=vocab, dtype=tf.int32),
        "metadata_tokens": tf.random.uniform((batch_size, metadata_len), maxval=vocab, dtype=tf.int32),
        "visual_tokens": tf.random.normal((batch_size, 16, visual_dim)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mm_cfg = cfg.mm_encoder

    model = MultimodalItemEncoder(
        vocab_size=mm_cfg["vocab_size"],
        text_len=mm_cfg["max_text_len"],
        metadata_len=mm_cfg["metadata_len"],
        visual_token_dim=mm_cfg["visual_token_dim"],
        item_cls_dim=mm_cfg["item_cls_dim"],
        robustness_masking_ratio=mm_cfg["robustness"]["masking_ratio"],
    )

    lr = float(mm_cfg["finetune"]["learning_rate"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    epochs = cfg.training["epochs"]
    batch_size = cfg.training["batch_size"]

    for epoch in range(epochs):
        batch = random_mm_batch(batch_size, cfg)
        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            loss = mm_pretrain_loss(outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"epoch={epoch+1} mm_pretrain_loss={float(loss):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    model.save_weights("artifacts/mm_item_encoder.weights.h5")
    print("Saved encoder weights to artifacts/mm_item_encoder.weights.h5")


if __name__ == "__main__":
    main()
