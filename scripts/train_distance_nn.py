#!/usr/bin/env python3
"""Train / bootstrap the residual pose-geometry distance NN.

Examples
--------
# Synthetic bootstrap (identity residual ≈ Z0) — run once so weights exist:
python scripts/train_distance_nn.py --synthetic --epochs 40

# Fine-tune on labeled CSV (columns: features... + distance_m):
python scripts/train_distance_nn.py --csv datasets/vision/distance/labels.csv

# Dump a template CSV header:
python scripts/train_distance_nn.py --print-header
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Django-free import path for the distance module pieces we need.
from monitoring.services import distance_estimation as dist  # noqa: E402


def _ensure_dirs() -> Path:
    out = ROOT / "models" / "vision" / "distance"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_synthetic(n: int = 4000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic people at random distances; labels ≈ noisy Z0 (residual should stay ~0)."""
    rng = np.random.default_rng(seed)
    fy = 720.0
    h_person = 1.70
    img_h, img_w = 720, 1280

    xs, z0s, ys = [], [], []
    for _ in range(n):
        true_z = float(rng.uniform(1.2, 10.0))
        # Project standing height into pixels with mild noise
        h_px = (fy * h_person) / true_z
        h_px *= float(rng.normal(1.0, 0.04))
        h_px = float(np.clip(h_px, 20.0, img_h * 0.95))
        aspect = float(rng.uniform(2.0, 4.0))
        w_px = h_px / aspect
        cx = float(rng.uniform(0.2, 0.8)) * img_w
        cy = float(rng.uniform(0.35, 0.75)) * img_h
        x1 = cx - w_px / 2
        y1 = cy - h_px / 2
        x2 = cx + w_px / 2
        y2 = cy + h_px / 2
        box = [x1, y1, x2, y2]
        # Occasional fake pose height ≈ bbox
        kps = None
        if rng.random() < 0.5:
            kps = [[0.0, 0.0, 0.0] for _ in range(17)]
            kps[0] = [cx, y1 + 0.05 * h_px, 0.9]
            kps[5] = [cx - 0.15 * w_px, y1 + 0.18 * h_px, 0.9]
            kps[6] = [cx + 0.15 * w_px, y1 + 0.18 * h_px, 0.9]
            kps[11] = [cx - 0.12 * w_px, y1 + 0.5 * h_px, 0.85]
            kps[12] = [cx + 0.12 * w_px, y1 + 0.5 * h_px, 0.85]
            kps[15] = [cx - 0.1 * w_px, y2 - 0.02 * h_px, 0.8]
            kps[16] = [cx + 0.1 * w_px, y2 - 0.02 * h_px, 0.8]

        feats, z0, _ = dist.extract_features(
            box, (img_w, img_h), det_conf=float(rng.uniform(0.5, 0.99)), keypoints=kps
        )
        # Label = true distance (+ small tape noise). Residual learns to undo bbox noise.
        label = true_z * float(rng.normal(1.0, 0.02))
        xs.append(feats)
        z0s.append(z0)
        ys.append(label)

    return (
        np.stack(xs).astype(np.float32),
        np.asarray(z0s, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import csv

    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [float(row[n]) for n in dist.FEATURE_NAMES]
            z0 = float(row.get("z0_m") or math.exp(float(row["log_z0"])))
            y = float(row["distance_m"])
            rows.append((feats, z0, y))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    xs = np.asarray([r[0] for r in rows], dtype=np.float32)
    z0s = np.asarray([r[1] for r in rows], dtype=np.float32)
    ys = np.asarray([r[2] for r in rows], dtype=np.float32)
    return xs, z0s, ys


def train(
    xs: np.ndarray,
    z0s: np.ndarray,
    ys: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dist.ResidualDistanceMLP:
    device = torch.device("cpu")
    model = dist.ResidualDistanceMLP(dist.FEATURE_DIM, (64, 32)).to(device)
    # Warm-start from existing weights if present
    wpath = _ensure_dirs() / "distance_nn.pt"
    if wpath.is_file():
        try:
            model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
            print(f"Warm-start from {wpath}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warm-start skipped: {exc}")

    ds = TensorDataset(
        torch.from_numpy(xs),
        torch.from_numpy(z0s),
        torch.from_numpy(ys),
    )
    n = len(ds)
    n_val = max(1, int(0.15 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, z0b, yb in train_loader:
            xb, z0b, yb = xb.to(device), z0b.to(device), yb.to(device)
            delta = model(xb)
            pred = z0b * torch.exp(delta.clamp(-1.5, 1.5))
            # Relative-friendly: regress log distance
            loss = loss_fn(torch.log(pred.clamp_min(1e-3)), torch.log(yb.clamp_min(1e-3)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(xb)
        tr_loss /= max(1, n_train)

        model.eval()
        va_loss = 0.0
        mae = 0.0
        with torch.no_grad():
            for xb, z0b, yb in val_loader:
                xb, z0b, yb = xb.to(device), z0b.to(device), yb.to(device)
                delta = model(xb)
                pred = z0b * torch.exp(delta.clamp(-1.5, 1.5))
                loss = loss_fn(torch.log(pred.clamp_min(1e-3)), torch.log(yb.clamp_min(1e-3)))
                va_loss += float(loss.item()) * len(xb)
                mae += float(torch.abs(pred - yb).sum().item())
        va_loss /= max(1, n_val)
        mae /= max(1, n_val)
        print(f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_MAE={mae:.3f} m")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAMS residual distance NN")
    parser.add_argument("--synthetic", action="store_true", help="Bootstrap on synthetic Z0 targets")
    parser.add_argument("--csv", type=Path, help="Labeled CSV with feature columns + distance_m")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-synthetic", type=int, default=4000)
    parser.add_argument("--print-header", action="store_true")
    args = parser.parse_args()

    if args.print_header:
        cols = dist.FEATURE_NAMES + ["z0_m", "distance_m"]
        print(",".join(cols))
        return

    # Avoid Django settings for path helpers during script use
    out_dir = _ensure_dirs()
    # Point distance module paths at repo models/ without Django by monkeypatching
    dist._vision_dir = lambda: out_dir  # type: ignore[method-assign]
    dist.save_default_calib(out_dir / "calib.json")

    if args.csv:
        xs, z0s, ys = load_csv(args.csv)
        print(f"Loaded {len(ys)} rows from {args.csv}")
    elif args.synthetic:
        xs, z0s, ys = make_synthetic(args.n_synthetic)
        print(f"Synthetic dataset: {len(ys)} samples")
    else:
        parser.error("Pass --synthetic or --csv PATH (or --print-header)")

    model = train(xs, z0s, ys, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    wpath = out_dir / "distance_nn.pt"
    torch.save(model.state_dict(), wpath)
    meta = {
        "weights": str(wpath.relative_to(ROOT)),
        "n_samples": int(len(ys)),
        "source": str(args.csv) if args.csv else "synthetic",
        "feature_names": dist.FEATURE_NAMES,
        "formula": "Z = Z0 * exp(delta)",
    }
    with (out_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved weights → {wpath}")
    print(f"Calib        → {out_dir / 'calib.json'}")


if __name__ == "__main__":
    main()
