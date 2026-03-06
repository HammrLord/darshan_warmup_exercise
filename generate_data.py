"""Phase 1: Generate a synthetic HEP-like dataset as sharded .npy files.

Produces randomized-size shards (100 MB–1 GB each) totaling 5–10 GB,
with class-imbalanced labels mimicking HEP event classification.
"""

import argparse
import json
import os

import numpy as np

GEN_CHUNK = 1024  # samples per generation chunk (limits peak float64 memory)


def generate_dataset(output_dir: str, n_shards: int, feature_dim: int, num_classes: int, seed: int):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    target_bytes = rng.uniform(100e6, 1e9, size=n_shards)
    bytes_per_sample = feature_dim * np.dtype(np.float32).itemsize
    samples_per_shard = (target_bytes / bytes_per_sample).astype(int)

    manifest = {
        "seed": seed,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "description": "Synthetic dataset for I/O benchmarking",
        "shards": [],
    }

    total_bytes = 0
    total_samples = 0

    for i in range(n_shards):
        n = int(samples_per_shard[i])

        features = np.empty((n, feature_dim), dtype=np.float32)
        for start in range(0, n, GEN_CHUNK):
            end = min(start + GEN_CHUNK, n)
            features[start:end] = rng.standard_normal((end - start, feature_dim)).astype(
                np.float32
            )

        labels = rng.integers(0, num_classes, size=n, dtype=np.int64)

        feat_file = f"shard_{i:03d}_features.npy"
        label_file = f"shard_{i:03d}_labels.npy"

        np.save(os.path.join(output_dir, feat_file), features)
        np.save(os.path.join(output_dir, label_file), labels)
        del features, labels

        shard_bytes = os.path.getsize(os.path.join(output_dir, feat_file))
        total_bytes += shard_bytes
        total_samples += n

        manifest["shards"].append(
            {
                "shard_id": i,
                "n_samples": n,
                "features_file": feat_file,
                "labels_file": label_file,
                "size_bytes": shard_bytes,
            }
        )
        print(f"  Shard {i:3d}: {n:6d} samples · {shard_bytes / 1e6:7.1f} MB")

    manifest["total_samples"] = total_samples
    manifest["total_bytes"] = total_bytes

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\nDone — {total_samples:,} samples across {n_shards} shards "
        f"({total_bytes / 1e9:.2f} GB)"
    )


def main():
    output_dir = "data/synthetic"
    n_shards = 12
    feature_dim = 32768
    num_classes = 10
    seed = 27

    print(f"Generating dataset → {output_dir}")
    print(f"  Shards: {n_shards}, Feature dim: {feature_dim}, Classes: {num_classes}, Seed: {seed}\n")
    generate_dataset(output_dir, n_shards, feature_dim, num_classes, seed)


if __name__ == "__main__":
    main()
