#!/usr/bin/env python3
"""Fine-tune the residual distance network on a labeled CSV.

The pinhole calibration (``models/vision/distance/calib.json``) has to already
exist. Notebook 04 writes it from the dataset projection matrices.

Examples
--------
python scripts/train_distance_nn.py --csv datasets/vision/distance/labels.csv
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

from monitoring.services import distance_estimation as dist


def _ensure_dirs() -> Path:
    out = ROOT / "models" / "vision" / "distance"
    out.mkdir(parents=True, exist_ok=True)
    return out


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
    hidden: tuple[int, ...],
) -> dist.ResidualDistanceMLP:
    device = torch.device("cpu")
    model = dist.ResidualDistanceMLP(dist.FEATURE_DIM, hidden).to(device)
    wpath = _ensure_dirs() / "distance_nn.pt"
    if wpath.is_file():
        try:
            model.load_state_dict(torch.load(wpath, map_location=device, weights_only=True))
            print(f"Warm-start from {wpath}")
        except Exception as exc:
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
            pred = z0b * torch.exp(delta)
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
                pred = z0b * torch.exp(delta)
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
    parser.add_argument("--csv", type=Path, help="Labeled CSV with feature columns + distance_m")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--print-header", action="store_true")
    args = parser.parse_args()

    if args.print_header:
        cols = dist.FEATURE_NAMES + ["z0_m", "distance_m"]
        print(",".join(cols))
        return

    out_dir = _ensure_dirs()
    dist._vision_dir = lambda: out_dir 
    if not args.csv:
        parser.error("Pass --csv PATH (or --print-header). Calibration comes from notebook 04.")
    calib = dist.load_calib()
    xs, z0s, ys = load_csv(args.csv)
    print(f"Loaded {len(ys)} rows from {args.csv}")

    model = train(
        xs, z0s, ys,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=tuple(int(h) for h in calib["hidden"]),
    )
    wpath = out_dir / "distance_nn.pt"
    torch.save(model.state_dict(), wpath)
    meta = {
        "weights": str(wpath.relative_to(ROOT)),
        "n_samples": int(len(ys)),
        "source": str(args.csv),
        "feature_names": dist.FEATURE_NAMES,
        "formula": "Z = Z0 * exp(delta)",
    }
    with (out_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved weights → {wpath}")


if __name__ == "__main__":
    main()
