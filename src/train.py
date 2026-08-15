from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import TabularDataset
from src.models.mlp import MLPRegressor
from src.preprocessing import load_dataset, build_preprocessor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    dc = cfg["data"]
    frame = load_dataset(dc["csv_path"])

    cat_cols = list(cfg["features"]["categorical"])
    num_cols = list(cfg["features"]["numeric"])
    target = dc["target_column"]

    required = set(cat_cols + num_cols + [target])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    frame = frame.dropna(subset=[target]).copy()
    trainval, test = train_test_split(
        frame, test_size=dc["test_fraction"], random_state=cfg["seed"]
    )
    val_rel = dc["validation_fraction"] / (1.0 - dc["test_fraction"])
    train, val = train_test_split(
        trainval, test_size=val_rel, random_state=cfg["seed"]
    )

    pre = build_preprocessor(cat_cols, num_cols)
    x_train = pre.fit_transform(train[cat_cols + num_cols])
    x_val = pre.transform(val[cat_cols + num_cols])
    x_test = pre.transform(test[cat_cols + num_cols])

    y_train = train[target].to_numpy(dtype=np.float32)
    y_val = val[target].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    tr_loader = DataLoader(
        TabularDataset(x_train, y_train),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    va_loader = DataLoader(
        TabularDataset(x_val, y_val),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc = cfg["model"]
    model = MLPRegressor(
        input_size=x_train.shape[1],
        hidden_size=mc["hidden_size"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "mlp_best.pt"

    best, stale, history = float("inf"), 0, []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_losses = []
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(loss_fn(model(x), y).item())

        tr_loss = float(np.mean(train_losses))
        va_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss})
        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} | val={va_loss:.6f}")

        if va_loss < best:
            best, stale = va_loss, 0
            torch.save({"model_state": model.state_dict(), "input_size": x_train.shape[1]}, best_path)
        else:
            stale += 1
            if stale >= int(cfg["training"]["patience"]):
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        train_pred = model(torch.as_tensor(x_train, dtype=torch.float32, device=device)).cpu().numpy()
        val_pred = model(torch.as_tensor(x_val, dtype=torch.float32, device=device)).cpu().numpy()
        test_pred = model(torch.as_tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()

    # Residual learner: CatBoost learns correction y - MLP(x)
    residual_train = y_train - train_pred
    cb_cfg = cfg["catboost"]
    residual_model = CatBoostRegressor(
        iterations=cb_cfg["iterations"],
        depth=cb_cfg["depth"],
        learning_rate=cb_cfg["learning_rate"],
        loss_function=cb_cfg["loss_function"],
        verbose=cb_cfg["verbose"],
        random_seed=cfg["seed"],
    )
    residual_model.fit(x_train, residual_train)

    final_test_pred = test_pred + residual_model.predict(x_test)

    joblib.dump(pre, out / "preprocessor.joblib")
    residual_model.save_model(str(out / "catboost_residual.cbm"))
    np.save(out / "test_indices.npy", test.index.to_numpy())
    np.save(out / "test_targets.npy", y_test)
    np.save(out / "test_predictions.npy", final_test_pred)

    with open(out / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")
    print("MLP test RMSE:", mean_squared_error(y_test, test_pred) ** 0.5)
    print("Hybrid test RMSE:", mean_squared_error(y_test, final_test_pred) ** 0.5)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/thermal.yaml")
    args = p.parse_args()
    main(args.config)
