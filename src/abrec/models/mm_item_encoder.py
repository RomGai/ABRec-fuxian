from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf


class MultimodalItemEncoder(tf.keras.Model):
    """Qwen2-VL-2B compatible interface (TF scaffold backend).

    Replace internal layers with an actual Qwen2-VL backend while preserving
    task heads and `encode_item()` API.
    """

    def __init__(
        self,
        vocab_size: int,
        text_len: int,
        metadata_len: int,
        visual_token_dim: int,
        item_cls_dim: int,
        robustness_masking_ratio: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.metadata_len = metadata_len
        self.item_cls_dim = item_cls_dim
        self.robustness_masking_ratio = robustness_masking_ratio

        self.token_embedding = tf.keras.layers.Embedding(vocab_size, item_cls_dim)
        self.text_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(item_cls_dim // 2, return_sequences=True)
        )
        self.meta_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(item_cls_dim // 2, return_sequences=False)
        )
        self.visual_proj = tf.keras.layers.Dense(item_cls_dim)

        self.item_cls = self.add_weight(
            "item_cls",
            shape=(1, 1, item_cls_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.fusion = tf.keras.layers.Dense(item_cls_dim, activation="gelu")

        self.text_decoder = tf.keras.layers.Dense(vocab_size)
        self.meta_to_text_decoder = tf.keras.layers.Dense(vocab_size)
        self.robustness_head = tf.keras.layers.Dense(item_cls_dim)

    def _apply_mask(self, x: tf.Tensor, ratio: float) -> tf.Tensor:
        mask = tf.cast(tf.random.uniform(tf.shape(x)) > ratio, x.dtype)
        return x * mask

    def encode_item(
        self, text_tokens: tf.Tensor, visual_tokens: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        text_embed = self.token_embedding(text_tokens)
        text_hidden = self.text_encoder(text_embed)
        visual_hidden = self.visual_proj(visual_tokens)

        pooled_text = tf.reduce_mean(text_hidden, axis=1)
        pooled_visual = tf.reduce_mean(visual_hidden, axis=1)
        fused = self.fusion(tf.concat([pooled_text, pooled_visual], axis=-1))

        batch = tf.shape(text_tokens)[0]
        cls = tf.tile(self.item_cls, [batch, 1, 1])[:, 0, :]
        item_cls_hidden = cls + fused
        return item_cls_hidden, text_hidden

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
        text_tokens = inputs["text_tokens"]
        full_text_tokens = inputs["full_text_tokens"]
        metadata_tokens = inputs["metadata_tokens"]
        visual_tokens = inputs["visual_tokens"]

        masked_text_tokens = self._apply_mask(tf.cast(text_tokens, tf.float32), 0.2)
        masked_text_tokens = tf.cast(masked_text_tokens, tf.int32)

        item_emb, text_hidden = self.encode_item(masked_text_tokens, visual_tokens)
        content_logits = self.text_decoder(text_hidden)

        meta_embed = self.token_embedding(metadata_tokens)
        meta_hidden = self.meta_encoder(meta_embed)
        meta_logits = self.meta_to_text_decoder(meta_hidden)

        robust_text_tokens = tf.cast(
            self._apply_mask(tf.cast(text_tokens, tf.float32), self.robustness_masking_ratio),
            tf.int32,
        )
        robust_visual_tokens = self._apply_mask(visual_tokens, self.robustness_masking_ratio)
        robust_item_emb, _ = self.encode_item(robust_text_tokens, robust_visual_tokens)

        return {
            "item_embedding": item_emb,
            "content_logits": content_logits,
            "meta_logits": meta_logits,
            "robust_embedding": self.robustness_head(robust_item_emb),
            "target_text": full_text_tokens,
        }


def mm_pretrain_loss(outputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    target_text = outputs["target_text"]
    content_logits = outputs["content_logits"]
    meta_logits = outputs["meta_logits"]

    content_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(target_text, content_logits, from_logits=True)
    )

    target_bow = tf.reduce_mean(tf.one_hot(target_text, depth=tf.shape(content_logits)[-1]), axis=1)
    meta_loss = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(target_bow, tf.nn.softmax(meta_logits), from_logits=False)
    )

    robust_target = tf.stop_gradient(outputs["item_embedding"])
    robust_loss = tf.reduce_mean(tf.square(outputs["robust_embedding"] - robust_target))

    return content_loss + meta_loss + robust_loss
