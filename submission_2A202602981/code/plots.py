"""plots.py — Vẽ biểu đồ huấn luyện và so sánh thí nghiệm.

Mỗi thí nghiệm một ảnh: figures/<exp_id>.png
Ảnh so sánh nhóm: figures/compare_<nhóm>.png
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG gồm 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    cfg = result["cfg"]
    hist = result["history"]
    summary = result["summary"]
    epochs = hist["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", marker="o", markersize=3)
    ax.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", marker="s", markersize=3)
    best_ep = summary.get("best_epoch", 1)
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.7, label=f"Best Ep ({best_ep})")
    ax.set_title("Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Ô 2: Accuracy & Macro-F1
    ax = axes[1]
    ax.plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", marker="^", markersize=3)
    if "val_macro_f1" in hist and len(hist["val_macro_f1"]) > 0:
        ax.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#9467bd", marker="d", markersize=3)
    ax.axvline(best_ep, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Val Metric vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Ô 3: Gradient Norm
    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", color="#d62728", marker="x", markersize=3)
    ax.set_title("Gradient Norm vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Tiêu đề tổng quát
    exp_id = cfg.get("exp_id", "exp")
    desc = f"{cfg.get('optimizer')} (lr={cfg.get('lr')}), loss={cfg.get('loss')}, batch={cfg.get('batch')}"
    fig.suptitle(f"[{exp_id}] {desc}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        exp_id = res["cfg"].get("exp_id", "unnamed")
        hist = res["history"]
        if metric in hist:
            ax.plot(hist["epoch"], hist[metric], label=exp_id, marker="o", markersize=3)

    metric_names = {
        "val_loss": "Validation Loss",
        "val_macro_f1": "Validation Macro-F1",
        "val_acc": "Validation Accuracy",
        "train_loss": "Train Loss",
        "grad_norm": "Gradient Norm"
    }

    display_name = metric_names.get(metric, metric)
    ax.set_title(title or f"So sánh {display_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(display_name)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
