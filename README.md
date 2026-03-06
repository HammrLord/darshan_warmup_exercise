# PyTorch DataLoader Microbenchmark Submission
Benchmarks data loading throughput and scaling efficiency for a synthetic dataset across `num_workers = {1, 2, 4, 8}`.

> **Platform Note:** This benchmark is supported on **macOS and Linux only**. It will not run natively on Windows due to its use of the Unix-only `resource` module for context switch tracking and `iostat` for disk I/O measurement. The sudo cache-drop also relies on Unix-specific kernel interfaces.

## Reproduce
```bash
# 1. Environment & Data Generation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py

# 2. Run benchmark
# By default, drops the OS page cache before each run for accurate cold-disk measurements.
# This requires sudo and will prompt for your password on first run.
python benchmark.py

# To bypass sudo (faster, but measures warm-cache performance — less accurate):
python benchmark.py --no-sudo
```

Results are written to `results/benchmark_results.json` and `results/benchmark_results.png`.

## Resource Utilization Metrics
Beyond throughput and efficiency, the benchmark captures OS-level hardware counters:

| Metric | How It Is Measured |
|---|---|
| **Disk Read (MB/s)** | `psutil.io_counters()` on Linux; `resource.getrusage()` block counts on macOS |
| **Context Switches** | `resource.getrusage()` voluntary + involuntary thread switches via the kernel |
| **Physical IOPS** | `iostat` cumulative transfer counter divided by elapsed time |
| **CPU Utilization (%)** | User + System CPU time divided by wall-clock duration |

> **macOS note:** `mmap` reads bypass the block I/O counters on macOS. If disk reads report 0, the benchmark falls back to a synthetic estimate based on dataset size. This is noted in the output.

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
