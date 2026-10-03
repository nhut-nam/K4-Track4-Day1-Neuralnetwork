"""train.py — Vòng lặp huấn luyện, đánh giá và thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là đổi dict cfg rồi gọi lại run_experiment (xem GUIDE, Part 2).
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng validation
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=-1))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(X)
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            batch_loss = F.cross_entropy(logits, yb, reduction="sum").item()
        elif loss_name == "mse":
            y_one_hot = F.one_hot(yb, num_classes=logits.shape[1]).float()
            batch_loss = F.mse_loss(logits, y_one_hot, reduction="sum").item()
        else:
            raise ValueError(f"Không hỗ trợ loss: {loss_name}")

        total_loss += batch_loss
        pred = logits.argmax(dim=-1)
        total_correct += (pred == yb).sum().item()

        yb_np = yb.cpu().numpy()
        pred_np = pred.cpu().numpy()
        np.add.at(cm, (yb_np, pred_np), 1)

    avg_loss = total_loss / n
    acc = total_correct / n
    f1 = macro_f1_from_confusion(cm)

    return {"loss": avg_loss, "acc": acc, "macro_f1": f1}


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Hàm tính loss cho mini-batch."""
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Không hỗ trợ loss: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    set_seed(cfg["seed"])
    hidden = tuple(cfg.get("hidden", (256, 128)))
    device = data["X_tr"].device

    model = MLP(
        hidden=hidden,
        dropout=cfg.get("dropout", 0.0),
        init=cfg.get("init", "he")
    ).to(device)

    expected = EXPECTED_PARAMS.get(hidden)
    if expected is not None:
        assert count_params(model) == expected, f"Sai số tham số: {count_params(model)} != {expected}"

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9)
    )

    precision = cfg.get("precision", "fp32")
    use_cuda = device.type == "cuda"
    use_amp = precision in ("fp16", "bf16") and use_cuda
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and use_cuda))

    # Step 0 sanity check trên tập validation
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    step0_loss = step0_res["loss"]

    epochs = cfg["epochs"]
    batch_size = cfg["batch"]
    clip_norm = cfg.get("clip_norm")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    # Tập con cố định của train để đánh giá train loss ở eval mode (50 000 mẫu)
    sub_n = min(50000, len(data["X_tr"]))
    X_tr_sub = data["X_tr"][:sub_n]
    y_tr_sub = data["y_tr"][:sub_n]

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.time()

        model.train()
        epoch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                print(f"[{cfg['exp_id']}] Epoch {epoch}: Loss bị NaN/inf -> Diverged!")
                break

            if precision == "fp16" and use_cuda:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            epoch_grad_norms.append(gn)

        if diverged:
            break

        if use_cuda:
            torch.cuda.synchronize(device)
        ep_time = time.time() - t0

        # Đánh giá cuối epoch ở chế độ eval
        tr_eval = evaluate(model, X_tr_sub, y_tr_sub, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(ep_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    peak_mem_MB = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if use_cuda else 0.0

    # Lấy metrics tại best_epoch (1-indexed)
    best_idx = (best_epoch - 1) if (best_epoch - 1 < len(history["val_acc"])) else -1
    best_val_acc = history["val_acc"][best_idx] if history["val_acc"] else 0.0
    best_val_macro_f1 = history["val_macro_f1"][best_idx] if history["val_macro_f1"] else 0.0
    final_tr_loss = history["train_loss"][-1] if history["train_loss"] else step0_loss
    final_v_loss = history["val_loss"][-1] if history["val_loss"] else step0_loss
    avg_ep_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss if not diverged else float("nan")),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_tr_loss),
        "final_val_loss": float(final_v_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(avg_ep_time),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề row_id,pred."""
    df = pd.DataFrame({"row_id": row_id, "pred": preds.astype(int)})
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dự đoán trên tập eval bằng best_state và ghi file predictions_eval.csv."""
    device = data["X_eval"].device
    model = MLP(
        hidden=tuple(cfg.get("hidden", (256, 128))),
        dropout=cfg.get("dropout", 0.0),
        init=cfg.get("init", "he")
    ).to(device)

    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
    print(f"Đã lưu dự đoán eval vào {pred_path} ({len(preds)} dòng).")
