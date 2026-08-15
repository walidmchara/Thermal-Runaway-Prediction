from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def safe_mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["output"]["directory"])
    y = np.load(out / "test_targets.npy")
    pred = np.load(out / "test_predictions.npy")

    metrics = {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAPE_percent": safe_mape(y, pred),
        "R2": float(r2_score(y, pred)),
    }

    print(json.dumps(metrics, indent=2))
    with open(out / "hybrid_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y, pred, alpha=0.7)
    lo = min(float(y.min()), float(pred.min()))
    hi = max(float(y.max()), float(pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual Severity Index")
    plt.ylabel("Predicted Severity Index")
    plt.title("Thermal Runaway Severity: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out / "actual_vs_predicted.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/thermal.yaml")
    main(p.parse_args().config)
