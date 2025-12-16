import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.interpolate import interp1d


def main():
    parser = argparse.ArgumentParser(description="Plot FID scores over epochs.")
    parser.add_argument(
        "--output", type=str, default="fid_plot.svg", help="Output file path (SVG)."
    )
    args = parser.parse_args()
    # fashion mnist
    # - 10000 - fid: 79 (1000 samples)
    # - 30000 - fid: 73 (1000 samples)
    # - 40000 - fid: 60 (1000 samples)
    # - 50000 - fid: 81 (1000 samples)
    # - 60000 - fid: 87 (1000 samples)
    # - 70000 - fid: 78 (1000 samples)
    # - 80000 - fid: 82 (1000 samples)
    # - 90000 - fid: 92 (1000 samples)
    # - 100000 - fid: 89 (1000 samples)

    fashion_epochs_known = np.array(
        [10000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    )
    fashion_fid_known = np.array([79, 73, 60, 81, 87, 78, 82, 92, 89])

    #
    # mnist
    # - 10000 - fid: 44 (1000 samples) ????
    # - 20000 - fid: 82 (1000 samples)
    # - 30000 - fid: 79 (1000 samples)
    # - 40000 - fid: 73 (1000 samples)
    # - 50000 - fid: 60 (1000 samples)
    # - 60000 - fid: 50 (1000 samples)
    # - 70000 - fid: 41 (1000 samples)
    # - 80000 - fid: 34 (1000 samples)
    # - 90000 - fid: 29 (1000 samples)
    # - 100000 - fid: 27 (1000 samples)

    mnist_epochs_known = np.array(
        [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    )
    mnist_fid_known = np.array([82, 79, 73, 60, 50, 41, 34, 29, 27])

    # Defines x-axis: 10000 to 100000, step 10000
    epochs_all = np.arange(10000, 100001, 10000)

    # Interpolate missing values
    # For Fashion MNIST
    f_interp = interp1d(
        fashion_epochs_known, fashion_fid_known, kind="linear", fill_value="extrapolate"
    )
    fashion_fid_all = f_interp(epochs_all)

    # For MNIST (extrapolate backwards for < 30000)
    m_interp = interp1d(
        mnist_epochs_known, mnist_fid_known, kind="linear", fill_value="extrapolate"
    )
    mnist_fid_all = m_interp(epochs_all)

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs_all,
        fashion_fid_all,
        marker="o",
        label="Fashion MNIST",
        linestyle="-",
        color="#FF5733",
    )  # Orange-ish
    plt.plot(
        epochs_all,
        mnist_fid_all,
        marker="s",
        label="MNIST",
        linestyle="--",
        color="#3357FF",
    )  # Blue-ish

    # Highlight known points to distinguish them from interpolated ones (optional but nice)
    plt.scatter(fashion_epochs_known, fashion_fid_known, color="#FF5733", zorder=5)
    plt.scatter(mnist_epochs_known, mnist_fid_known, color="#3357FF", zorder=5)

    plt.title("FID Score vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(epochs_all, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output, format="svg")
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
