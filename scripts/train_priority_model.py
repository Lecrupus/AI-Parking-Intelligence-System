from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
HOTSPOTS_PATH = ROOT / "data" / "processed" / "hotspots.csv"
MODEL_PATH = ROOT / "models" / "gpu_priority_ranker.pt"
REPORT_PATH = ROOT / "reports" / "model_training_report.md"

FEATURE_COLUMNS = [
    "violation_count",
    "active_days",
    "unique_vehicle_count",
    "vehicle_recurrence_share",
    "peak_share",
    "weekend_share",
    "approval_rate",
    "sent_to_scita_share",
    "named_junction_share",
    "obstruction_signal_share",
    "severity_score",
    "congestion_impact_score",
]
TARGET_COLUMN = "priority_score"


class PriorityRanker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def main() -> None:
    if not HOTSPOTS_PATH.exists():
        raise FileNotFoundError("Build the prototype artifacts before training the ranker.")

    frame = pd.read_csv(HOTSPOTS_PATH)
    train_frame = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_frame) < 100:
        raise RuntimeError("Not enough hotspot rows to train the priority ranker.")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(train_frame))
    split = int(len(indices) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    x = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = train_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std == 0] = 1
    x_scaled = (x - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(x_scaled[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    x_test = torch.tensor(x_scaled[test_idx], dtype=torch.float32, device=device)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32, device=device)

    torch.manual_seed(42)
    model = PriorityRanker(len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(350):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
        test_pred = model(x_test)
        train_mae = torch.mean(torch.abs(train_pred - y_train)).item()
        test_mae = torch.mean(torch.abs(test_pred - y_test)).item()
        test_rmse = torch.sqrt(torch.mean((test_pred - y_test) ** 2)).item()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "metrics": {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
            },
        },
        MODEL_PATH,
    )

    report = [
        "# GPU Priority Ranker Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Device | {device} |",
        f"| CUDA available | {torch.cuda.is_available()} |",
        f"| CUDA device | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'} |",
        f"| Training rows | {len(train_idx):,} |",
        f"| Test rows | {len(test_idx):,} |",
        f"| Train MAE | {train_mae:.3f} priority points |",
        f"| Test MAE | {test_mae:.3f} priority points |",
        f"| Test RMSE | {test_rmse:.3f} priority points |",
        "",
        "This model learns the current proxy priority score from hotspot features. Replace `priority_score` with verified enforcement outcomes when labels become available.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({"device": str(device), "test_mae": test_mae, "model": str(MODEL_PATH)}, indent=2))


if __name__ == "__main__":
    main()
