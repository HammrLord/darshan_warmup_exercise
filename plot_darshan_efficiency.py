import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_darshan_efficiency(ranks: list[int], run_times: list[float], io_perfs: list[float], output_dir: str):
    """
    Plots Darshan Scaling Efficiency using the formula: E = T1 / (n * Tn) * 100
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(ranks) != len(run_times):
        raise ValueError("Length of ranks must match length of run_times")
        
    t1 = run_times[0]
    efficiencies = []
    
    for n, t_n in zip(ranks, run_times):
        # E = T1 / (n * Tn) * 100
        eff = (t1 / (n * t_n)) * 100
        efficiencies.append(eff)

    x = range(len(ranks))
    str_ranks = [str(r) for r in ranks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # — Raw I/O Throughput (MiB/s) from Darshan —
    ax1.plot(x, io_perfs, "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(str_ranks)
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("POSIX I/O Performance (MiB/s)")
    ax1.set_title("Darshan I/O Throughput")
    for i, v in enumerate(io_perfs):
        ax1.text(i, v + max(io_perfs) * 0.02, f"{v:.0f}", ha="center", fontsize=9)

    # — Scaling efficiency line —
    ax2.plot(x, efficiencies, "o-", color="#C44E52", linewidth=2, markersize=8)
    ax2.axhline(100, ls="--", color="gray", alpha=0.5, label="Ideal (100%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(str_ranks)
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Scaling Efficiency (%)")
    ax2.set_title("E = T₁ / (n × Tₙ) × 100")
    ax2.legend()
    ax2.set_ylim(0, max(120, max(efficiencies) + 10))

    fig.suptitle("Darshan Thread Scaling", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(output_dir, "darshan_efficiency_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Efficiency plot saved to → {path}")

def main():
    parser = argparse.ArgumentParser(description="Plot Darshan efficiency from execution times.")
    parser.add_argument("--ranks", type=int, nargs="+", required=True, help="List of threads used (e.g., 1 2 4 8 32)")
    parser.add_argument("--times", type=float, nargs="+", required=True, help="Execution times in seconds (e.g., 244 298 209 215 228)")
    parser.add_argument("--io-perf", type=float, nargs="+", required=True, help="POSIX I/O performance in MiB/s (e.g., 126.61 242.81 507.56 474.52 473.50)")
    parser.add_argument("--output-dir", default="results", help="Directory to save the plot")
    
    args = parser.parse_args()
    plot_darshan_efficiency(args.ranks, args.times, args.io_perf, args.output_dir)

if __name__ == "__main__":
    main()
