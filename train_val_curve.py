import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
dfs = {
    "No Aug": {
        "Original on Noisy": pd.read_csv("runs/detect/0_o_n/yololoss_0_o_n.csv"),
        "Noisy on Original": pd.read_csv("runs/detect/0_n_o/yololoss_0_n_o.csv"),
    },
    "Low Aug": {
        "Original on Noisy": pd.read_csv("runs/detect/1_o_n/yololoss_1_o_n.csv"),
        "Noisy on Original": pd.read_csv("runs/detect/1_n_o/yololoss_1_n_o.csv"),
    },
    "Medium Aug": {
        "Original on Noisy": pd.read_csv("runs/detect/2_o_n/yololoss_2_o_n.csv"),
        "Noisy on Original": pd.read_csv("runs/detect/2_n_o/yololoss_2_n_o.csv"),
    },
    "High Aug": {
        "Original on Noisy": pd.read_csv("runs/detect/3_o_n/yololoss_3_o_n.csv"),
        "Noisy on Original": pd.read_csv("Noisy_on_Original_high_frcnn_losses.csv"),
    },
}

# Colors for noise conditions
colors = {"Original on Noisy": "#1f77b4", "Noisy on Original": "#ff7f0e"}

# Create a 2×2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.ravel()

# Plot each augmentation setting in its own grid cell
for i, (aug_level, conditions) in enumerate(dfs.items()):
    ax = axs[i]
    for noise_type, df in conditions.items():
        # Plot training (solid) and validation (dashed)
        ax.plot(df["epoch"], df["train_loss"], linestyle='-', color=colors[noise_type],
                label=f"Train ({noise_type})")
        ax.plot(df["epoch"], df["val_loss"], linestyle='--', color=colors[noise_type],
                label=f"Val ({noise_type})")

    ax.set_title(aug_level, fontsize=13)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

# Add legend to one subplot (top-right)
axs[1].legend(fontsize=10, loc='upper right', frameon=True)

# Global title and layout
fig.suptitle("Loss curve for YOLOv8 Original on Clinical and Noisy on Clinical", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Loss curve for YOLOv8 Original on Clinical and Noisy on Clinical")
plt.show()

