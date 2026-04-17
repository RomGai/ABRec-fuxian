#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import tensorflow as tf

from abrec.config import load_config
from abrec.models.abrec import ABRecLossWeights, ABRecModel
from abrec.models.mm_item_encoder import MultimodalItemEncoder


def random_rec_batch(batch_size: int, cfg):
    data_cfg = cfg.data
    mm_cfg = cfg.mm_encoder
    mm_dim = cfg.model["mm_embedding_dim"]

    return {
        "user_id": tf.random.uniform((batch_size,), maxval=data_cfg["num_users"], dtype=tf.int32),
        "item_id": tf.random.uniform((batch_size,), maxval=data_cfg["num_items"], dtype=tf.int32),
        "recent_item_mm_embeddings": tf.random.normal((batch_size, data_cfg["recent_k"], mm_dim)),
        "item_mm_embedding": tf.random.normal((batch_size, mm_dim)),
        "label": tf.random.uniform((batch_size, 1), maxval=2, dtype=tf.int32),
        "encoder_stub": {
            "text_tokens": tf.random.uniform((batch_size, mm_cfg["max_text_len"]), maxval=mm_cfg["vocab_size"], dtype=tf.int32),
            "full_text_tokens": tf.random.uniform((batch_size, mm_cfg["max_text_len"]), maxval=mm_cfg["vocab_size"], dtype=tf.int32),
            "metadata_tokens": tf.random.uniform((batch_size, mm_cfg["metadata_len"]), maxval=mm_cfg["vocab_size"], dtype=tf.int32),
            "visual_tokens": tf.random.normal((batch_size, 16, mm_cfg["visual_token_dim"])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mm_cfg = cfg.mm_encoder

    mm_encoder = MultimodalItemEncoder(
        vocab_size=mm_cfg["vocab_size"],
        text_len=mm_cfg["max_text_len"],
        metadata_len=mm_cfg["metadata_len"],
        visual_token_dim=mm_cfg["visual_token_dim"],
        item_cls_dim=mm_cfg["item_cls_dim"],
        robustness_masking_ratio=mm_cfg["robustness"]["masking_ratio"],
    )

    if os.path.exists("artifacts/mm_item_encoder.weights.h5"):
        dummy = random_rec_batch(2, cfg)["encoder_stub"]
        _ = mm_encoder(dummy, training=False)
        mm_encoder.load_weights("artifacts/mm_item_encoder.weights.h5")

    loss_weights = ABRecLossWeights(
        alpha=float(cfg.losses["alpha"]),
        beta=float(cfg.losses["beta"]),
        eta=float(cfg.losses["eta"]),
        total_loss_weight=float(cfg.losses["total_loss_weight"]),
    )

    model = ABRecModel(
        num_users=cfg.data["num_users"],
        num_items=cfg.data["num_items"],
        id_embedding_dim=cfg.model["id_embedding_dim"],
        mm_embedding_dim=cfg.model["mm_embedding_dim"],
        backbone_hidden_sizes=cfg.model["backbone_hidden_sizes"],
        mm_item_encoder=mm_encoder,
        loss_weights=loss_weights,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg.training["learning_rate"]))

    for epoch in range(cfg.training["epochs"]):
        batch = random_rec_batch(cfg.training["batch_size"], cfg)
        inputs = {
            "user_id": batch["user_id"],
            "item_id": batch["item_id"],
            "recent_item_mm_embeddings": batch["recent_item_mm_embeddings"],
            "item_mm_embedding": batch["item_mm_embedding"],
        }
        y_true = tf.cast(batch["label"], tf.float32)

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            losses = model.compute_losses(y_true, outputs)

        grads = tape.gradient(losses["total"], model.trainable_variables)

        gamma = model.contribution_ratio(outputs["h_id"], outputs["h_mm"])
        modulated_grads = model.gradient_modulation(grads, gamma=gamma, omega=0.6)
        optimizer.apply_gradients(zip(modulated_grads, model.trainable_variables))

        print(
            f"epoch={epoch+1} total={float(losses['total']):.4f} "
            f"bce={float(losses['bce']):.4f} align={float(losses['alignment']):.4f} gamma={float(gamma):.4f}"
        )


if __name__ == "__main__":
    main()
