from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import tensorflow as tf

from abrec.losses import bce_loss, cosine_regularization, sliced_wasserstein_alignment


class SharedBackbone(tf.keras.layers.Layer):
    def __init__(self, hidden_sizes: List[int]):
        super().__init__()
        self.layers_ = [tf.keras.layers.Dense(h, activation="relu") for h in hidden_sizes]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.layers_:
            x = layer(x)
        return x


@dataclass
class ABRecLossWeights:
    alpha: float
    beta: float
    eta: float
    total_loss_weight: float


class ABRecModel(tf.keras.Model):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        id_embedding_dim: int,
        mm_embedding_dim: int,
        backbone_hidden_sizes: List[int],
        mm_item_encoder: tf.keras.Model,
        loss_weights: ABRecLossWeights,
    ) -> None:
        super().__init__()
        self.loss_weights = loss_weights

        self.user_id_embedding = tf.keras.layers.Embedding(num_users, id_embedding_dim)
        self.item_id_embedding = tf.keras.layers.Embedding(num_items, id_embedding_dim)

        self.id_backbone = SharedBackbone(backbone_hidden_sizes)
        self.mm_backbone = SharedBackbone(backbone_hidden_sizes)
        self.prediction_head = tf.keras.layers.Dense(1)

        self.mm_item_encoder = mm_item_encoder
        self.mm_item_encoder.trainable = False

        self.mm_user_proj = tf.keras.layers.Dense(mm_embedding_dim)

    def _mm_user_embedding(self, recent_item_mm_embeddings: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(recent_item_mm_embeddings, axis=1)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        user_ids = inputs["user_id"]
        item_ids = inputs["item_id"]

        uid = self.user_id_embedding(user_ids)
        iid = self.item_id_embedding(item_ids)
        h_id = self.id_backbone(tf.concat([uid, iid], axis=-1))

        recent_item_mm = inputs["recent_item_mm_embeddings"]
        mm_user = self.mm_user_proj(self._mm_user_embedding(recent_item_mm))
        mm_item = inputs["item_mm_embedding"]
        h_mm = self.mm_backbone(tf.concat([mm_user, mm_item], axis=-1))

        logits = self.prediction_head(tf.concat([h_id, h_mm], axis=-1))
        y_pred = tf.nn.sigmoid(logits)

        return {
            "h_id": h_id,
            "h_mm": h_mm,
            "y_pred": y_pred,
        }

    def compute_losses(self, y_true: tf.Tensor, outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        bce = bce_loss(y_true, outputs["y_pred"])
        wasserstein = sliced_wasserstein_alignment(outputs["h_id"], outputs["h_mm"])
        cosine = cosine_regularization(outputs["h_id"], outputs["h_mm"])

        alignment = self.loss_weights.alpha * wasserstein + self.loss_weights.beta * cosine
        total = self.loss_weights.total_loss_weight * (
            bce + self.loss_weights.eta * alignment
        )

        return {
            "bce": bce,
            "wasserstein": wasserstein,
            "cosine": cosine,
            "alignment": alignment,
            "total": total,
        }

    def contribution_ratio(self, h_id: tf.Tensor, h_mm: tf.Tensor) -> tf.Tensor:
        id_energy = tf.reduce_mean(tf.norm(h_id, axis=-1))
        mm_energy = tf.reduce_mean(tf.norm(h_mm, axis=-1))
        return id_energy / (mm_energy + 1e-8)

    def gradient_modulation(
        self,
        grads: List[tf.Tensor | None],
        gamma: float,
        omega: float,
    ) -> List[tf.Tensor | None]:
        scale = tf.cond(
            gamma > 1.0,
            lambda: tf.cast(1.0 / (1.0 + omega * (gamma - 1.0)), tf.float32),
            lambda: tf.cast(1.0 / (1.0 + omega * (1.0 / (gamma + 1e-8) - 1.0)), tf.float32),
        )
        return [None if g is None else g * scale for g in grads]
