#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from abrec.config import load_config
from abrec.models.abrec import ABRecLossWeights, ABRecModel
from abrec.models.mm_item_encoder import MultimodalItemEncoder


@dataclass
class SplitData:
    user_pos: Dict[int, List[int]]
    user_neg: Dict[int, List[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AB-Rec new_data train + inference pipeline")
    parser.add_argument("--config", default="configs/new_data_baby.yaml")
    parser.add_argument("--dataset-prefix", default="Baby_Products")
    parser.add_argument("--data-dir", default="new_data")
    parser.add_argument("--mode", choices=["all", "train", "infer"], default="all")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-steps-per-epoch", type=int, default=200)
    parser.add_argument("--candidate-negatives", type=int, default=1000)
    parser.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--artifact-dir", default="artifacts/new_data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_items(value: str) -> List[int]:
    value = str(value).strip()
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def read_user_items_file(path: Path) -> SplitData:
    user_pos: Dict[int, List[int]] = {}
    user_neg: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            user = int(row[0])
            user_pos[user] = _parse_items(row[1])
            user_neg[user] = _parse_items(row[2])
    return SplitData(user_pos=user_pos, user_neg=user_neg)


def read_item_texts(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                item_id = int(row["item_id"])
            except Exception:
                continue
            summary = str(row.get("summary", "")).strip()
            out[item_id] = summary if summary else f"item {item_id}"
    return out


def _project_or_pad(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[1] == target_dim:
        return x.astype(np.float32)
    if x.shape[1] > target_dim:
        return x[:, :target_dim].astype(np.float32)
    pad = np.zeros((x.shape[0], target_dim - x.shape[1]), dtype=np.float32)
    return np.concatenate([x.astype(np.float32), pad], axis=1)


def encode_items_from_hf(
    item_ids: Sequence[int],
    item_texts: Dict[int, str],
    hf_model_name: str,
    target_dim: int,
    batch_size: int = 64,
) -> np.ndarray:
    print(f"[Stage/MM] Loading HuggingFace backbone: {hf_model_name}")
    try:
        from transformers import AutoTokenizer, TFAutoModel
    except Exception as exc:
        raise RuntimeError(
            "transformers is required. Install dependencies first (pip install -r requirements.txt)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = TFAutoModel.from_pretrained(hf_model_name, from_pt=True)
    model.trainable = False

    all_vecs: List[np.ndarray] = []
    total = len(item_ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        ids = item_ids[start:end]
        texts = [item_texts.get(i, f"item {i}") for i in ids]
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="tf")
        outputs = model(**tokens, training=False)
        # [batch, hidden]
        emb = outputs.last_hidden_state[:, 0, :].numpy()
        all_vecs.append(emb)
        print(f"[Stage/MM] Encoded items {end}/{total}")

    mm = np.concatenate(all_vecs, axis=0)
    mm = _project_or_pad(mm, target_dim)
    print(f"[Stage/MM] Item embedding matrix shape={mm.shape}")
    return mm


def build_recent_embeddings(
    user_ids: Sequence[int],
    train_pos: Dict[int, List[int]],
    item_mm: np.ndarray,
    recent_k: int,
    mm_dim: int,
) -> np.ndarray:
    batch = len(user_ids)
    out = np.zeros((batch, recent_k, mm_dim), dtype=np.float32)
    for i, u in enumerate(user_ids):
        hist = train_pos.get(u, [])[-recent_k:]
        if not hist:
            continue
        h = np.array(hist, dtype=np.int32)
        vectors = item_mm[h]
        out[i, -len(hist) :, :] = vectors
    return out


def sample_train_batch(
    users: Sequence[int],
    train_pos: Dict[int, List[int]],
    train_neg: Dict[int, List[int]],
    item_mm: np.ndarray,
    recent_k: int,
    mm_dim: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    uid_list: List[int] = []
    iid_list: List[int] = []
    labels: List[int] = []

    for _ in range(batch_size):
        user = random.choice(users)
        pos_items = train_pos[user]
        neg_items = train_neg.get(user, [])
        if not pos_items:
            continue

        if random.random() < 0.5:
            item = random.choice(pos_items)
            label = 1
        else:
            if neg_items:
                item = random.choice(neg_items)
            else:
                item = random.choice(pos_items)
            label = 0

        uid_list.append(user)
        iid_list.append(item)
        labels.append(label)

    recent = build_recent_embeddings(uid_list, train_pos, item_mm, recent_k, mm_dim)
    item_vectors = item_mm[np.array(iid_list, dtype=np.int32)]

    return {
        "user_id": np.array(uid_list, dtype=np.int32),
        "item_id": np.array(iid_list, dtype=np.int32),
        "recent_item_mm_embeddings": recent,
        "item_mm_embedding": item_vectors,
        "label": np.array(labels, dtype=np.float32).reshape(-1, 1),
    }


def build_model(cfg, num_users: int, num_items: int) -> ABRecModel:
    mm_cfg = cfg.mm_encoder
    mm_encoder = MultimodalItemEncoder(
        vocab_size=mm_cfg["vocab_size"],
        text_len=mm_cfg["max_text_len"],
        metadata_len=mm_cfg["metadata_len"],
        visual_token_dim=mm_cfg["visual_token_dim"],
        item_cls_dim=mm_cfg["item_cls_dim"],
        robustness_masking_ratio=mm_cfg["robustness"]["masking_ratio"],
    )

    weights = ABRecLossWeights(
        alpha=float(cfg.losses["alpha"]),
        beta=float(cfg.losses["beta"]),
        eta=float(cfg.losses["eta"]),
        total_loss_weight=float(cfg.losses["total_loss_weight"]),
    )

    return ABRecModel(
        num_users=num_users,
        num_items=num_items,
        id_embedding_dim=cfg.model["id_embedding_dim"],
        mm_embedding_dim=cfg.model["mm_embedding_dim"],
        backbone_hidden_sizes=cfg.model["backbone_hidden_sizes"],
        mm_item_encoder=mm_encoder,
        loss_weights=weights,
    )


def _ndcg(rank: int, k: int) -> float:
    return (1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def infer_with_ranking(
    model: ABRecModel,
    test: SplitData,
    train: SplitData,
    item_mm: np.ndarray,
    num_items: int,
    recent_k: int,
    candidate_negatives: int,
) -> None:
    print("[Stage/Infer] Start ranking eval: 1 target + 1000 random negatives")
    rng = random.Random(42)

    total_instances = 0
    hit10 = hit20 = hit40 = 0.0
    ndcg10 = ndcg20 = ndcg40 = 0.0

    all_items = set(range(num_items))

    test_users = sorted(test.user_pos.keys())
    for u_idx, user in enumerate(test_users, start=1):
        user_targets = test.user_pos.get(user, [])
        user_test_negs = set(test.user_neg.get(user, []))
        train_seen = set(train.user_pos.get(user, []))
        user_seen_for_sampling = train_seen | set(user_targets)

        for target in user_targets:
            candidate_pool = list((all_items - user_seen_for_sampling - {target}) | user_test_negs)
            if not candidate_pool:
                continue
            if len(candidate_pool) >= candidate_negatives:
                negs = rng.sample(candidate_pool, k=candidate_negatives)
            else:
                negs = [rng.choice(candidate_pool) for _ in range(candidate_negatives)]

            candidates = [target] + negs
            user_ids = np.full((len(candidates),), user, dtype=np.int32)
            item_ids = np.array(candidates, dtype=np.int32)
            recent = build_recent_embeddings([user] * len(candidates), train.user_pos, item_mm, recent_k, item_mm.shape[1])
            item_vec = item_mm[item_ids]

            outputs = model(
                {
                    "user_id": tf.constant(user_ids),
                    "item_id": tf.constant(item_ids),
                    "recent_item_mm_embeddings": tf.constant(recent),
                    "item_mm_embedding": tf.constant(item_vec),
                },
                training=False,
            )
            scores = outputs["y_pred"].numpy().reshape(-1)
            order = np.argsort(-scores)
            rank = int(np.where(order == 0)[0][0]) + 1

            total_instances += 1
            hit10 += 1.0 if rank <= 10 else 0.0
            hit20 += 1.0 if rank <= 20 else 0.0
            hit40 += 1.0 if rank <= 40 else 0.0
            ndcg10 += _ndcg(rank, 10)
            ndcg20 += _ndcg(rank, 20)
            ndcg40 += _ndcg(rank, 40)

        if total_instances > 0:
            print(
                f"[Stage/Infer][User {u_idx}/{len(test_users)} id={user}] "
                f"avg_HR@10={hit10/total_instances:.4f} avg_HR@20={hit20/total_instances:.4f} avg_HR@40={hit40/total_instances:.4f} "
                f"avg_NDCG@10={ndcg10/total_instances:.4f} avg_NDCG@20={ndcg20/total_instances:.4f} avg_NDCG@40={ndcg40/total_instances:.4f} "
                f"processed_targets={total_instances}"
            )

    if total_instances == 0:
        print("[Stage/Infer] No test targets to evaluate.")
        return

    print("[Stage/Infer] Final metrics")
    print(f"  HR@10={hit10/total_instances:.6f}")
    print(f"  HR@20={hit20/total_instances:.6f}")
    print(f"  HR@40={hit40/total_instances:.6f}")
    print(f"  NDCG@10={ndcg10/total_instances:.6f}")
    print(f"  NDCG@20={ndcg20/total_instances:.6f}")
    print(f"  NDCG@40={ndcg40/total_instances:.6f}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    cfg = load_config(args.config)

    data_dir = Path(args.data_dir)
    prefix = args.dataset_prefix

    train_path = data_dir / f"{prefix}_user_items_negs_train.csv"
    test_path = data_dir / f"{prefix}_user_items_negs_test.csv"
    u_map = data_dir / f"{prefix}_u_map.tsv"
    i_map = data_dir / f"{prefix}_i_map.tsv"
    item_desc = data_dir / f"{prefix}_item_desc.tsv"

    print("[Stage/Data] Reading splits and mapping files")
    train = read_user_items_file(train_path)
    test = read_user_items_file(test_path)

    num_users = sum(1 for _ in u_map.open("r", encoding="utf-8")) - 1
    num_items = sum(1 for _ in i_map.open("r", encoding="utf-8")) - 1
    print(f"[Stage/Data] num_users={num_users} num_items={num_items}")
    print(f"[Stage/Data] train_users={len(train.user_pos)} test_users={len(test.user_pos)}")
    overlap = set(train.user_pos.keys()) & set(test.user_pos.keys())
    print(f"[Stage/DataLeakCheck] overlap(train_users, test_users)={len(overlap)} (expected 0)")

    recent_k = int(cfg.data["recent_k"])
    mm_dim = int(cfg.model["mm_embedding_dim"])

    os.makedirs(args.artifact_dir, exist_ok=True)
    item_mm_path = Path(args.artifact_dir) / f"{prefix}_item_mm.npy"

    if item_mm_path.exists():
        print(f"[Stage/MM] Loading cached item embeddings from {item_mm_path}")
        item_mm = np.load(item_mm_path)
    else:
        item_texts = read_item_texts(item_desc)
        item_ids = list(range(num_items))
        item_mm = encode_items_from_hf(
            item_ids=item_ids,
            item_texts=item_texts,
            hf_model_name=args.hf_model,
            target_dim=mm_dim,
        )
        np.save(item_mm_path, item_mm)
        print(f"[Stage/MM] Saved item embeddings -> {item_mm_path}")

    model = build_model(cfg, num_users=num_users, num_items=num_items)

    ckpt = Path(args.artifact_dir) / f"{prefix}_abrec.weights.h5"

    if args.mode in {"all", "train"}:
        users = list(train.user_pos.keys())
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg.training["learning_rate"]))
        print("[Stage/Train] Start training")

        for epoch in range(1, args.epochs + 1):
            print(f"[Stage/Train] Epoch {epoch}/{args.epochs} started")
            for step in range(1, args.train_steps_per_epoch + 1):
                batch = sample_train_batch(
                    users=users,
                    train_pos=train.user_pos,
                    train_neg=train.user_neg,
                    item_mm=item_mm,
                    recent_k=recent_k,
                    mm_dim=mm_dim,
                    batch_size=args.batch_size,
                )

                with tf.GradientTape() as tape:
                    outputs = model(
                        {
                            "user_id": tf.constant(batch["user_id"]),
                            "item_id": tf.constant(batch["item_id"]),
                            "recent_item_mm_embeddings": tf.constant(batch["recent_item_mm_embeddings"]),
                            "item_mm_embedding": tf.constant(batch["item_mm_embedding"]),
                        },
                        training=True,
                    )
                    losses = model.compute_losses(tf.constant(batch["label"]), outputs)

                grads = tape.gradient(losses["total"], model.trainable_variables)
                gamma = model.contribution_ratio(outputs["h_id"], outputs["h_mm"])
                grads = model.gradient_modulation(grads, gamma=gamma, omega=0.6)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                print(
                    f"[Stage/Train][Epoch {epoch}/{args.epochs}][Step {step}/{args.train_steps_per_epoch}] "
                    f"total={float(losses['total']):.4f} bce={float(losses['bce']):.4f} "
                    f"align={float(losses['alignment']):.4f} gamma={float(gamma):.4f}"
                )

        model.save_weights(str(ckpt))
        print(f"[Stage/Train] Saved checkpoint -> {ckpt}")

    if args.mode in {"all", "infer"}:
        if ckpt.exists():
            _ = model(
                {
                    "user_id": tf.constant([0], dtype=tf.int32),
                    "item_id": tf.constant([0], dtype=tf.int32),
                    "recent_item_mm_embeddings": tf.zeros((1, recent_k, mm_dim), dtype=tf.float32),
                    "item_mm_embedding": tf.zeros((1, mm_dim), dtype=tf.float32),
                },
                training=False,
            )
            model.load_weights(str(ckpt))
            print(f"[Stage/Infer] Loaded checkpoint <- {ckpt}")
        else:
            print("[Stage/Infer] WARNING: checkpoint not found, using current in-memory model weights")

        infer_with_ranking(
            model=model,
            test=test,
            train=train,
            item_mm=item_mm,
            num_items=num_items,
            recent_k=recent_k,
            candidate_negatives=args.candidate_negatives,
        )


if __name__ == "__main__":
    main()
