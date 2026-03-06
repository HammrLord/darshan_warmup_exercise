# PyTorch DataLoader Microbenchmark Submission

Benchmarks data loading throughput and scaling efficiency for a synthetic dataset across `num_workers = {1, 2, 4, 8}`.

## Reproduce

```bash
# 1. Environment & Data Generation
pip install -r requirements.txt
python generate_data.py --output-dir data/synthetic

# 2. Run benchmark
# By default, this prompts for sudo to drop the OS cache before each run.
# To bypass the sudo prompt (less accurate, but easier to run):
python benchmark.py --data-dir data/synthetic --no-sudo
```

Results are written to `results/benchmark_results.json` and `results/benchmark_results.png`.

## Data Schema

The generated dataset is stored as sharded NumPy `.npy` files described by a `manifest.json`:

```
data/synthetic/
├── manifest.json
├── shard_000_features.npy   # float32 array, shape (N_i, 32768)
├── shard_000_labels.npy     # int64 array,   shape (N_i,)
├── shard_001_features.npy
├── shard_001_labels.npy
└── ...
```

| Field | Description |
|---|---|
| **Features** | Random float32 values. Each sample has 32,768 features. |
| **Labels** | Random int64 values between 0 and 9. |
| **Shard sizes** | Randomized between 100 MB and 1 GB (seed-deterministic) |
| **Total size** | ~5–10 GB across 12 shards |
| **Seed** | 27 (fully reproducible) |
