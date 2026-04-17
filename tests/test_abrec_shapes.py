from __future__ import annotations

import tensorflow as tf

from abrec.models.abrec import ABRecLossWeights, ABRecModel
from abrec.models.mm_item_encoder import MultimodalItemEncoder


def test_abrec_forward_shapes():
    mm_encoder = MultimodalItemEncoder(
        vocab_size=100,
        text_len=16,
        metadata_len=8,
        visual_token_dim=32,
        item_cls_dim=32,
        robustness_masking_ratio=0.3,
    )

    model = ABRecModel(
        num_users=50,
        num_items=100,
        id_embedding_dim=16,
        mm_embedding_dim=32,
        backbone_hidden_sizes=[32, 16],
        mm_item_encoder=mm_encoder,
        loss_weights=ABRecLossWeights(alpha=0.25, beta=0.7, eta=0.6, total_loss_weight=1.0),
    )

    batch = 4
    outputs = model(
        {
            "user_id": tf.constant([1, 2, 3, 4]),
            "item_id": tf.constant([3, 2, 1, 0]),
            "recent_item_mm_embeddings": tf.random.normal((batch, 5, 32)),
            "item_mm_embedding": tf.random.normal((batch, 32)),
        },
        training=False,
    )

    assert outputs["y_pred"].shape == (batch, 1)
