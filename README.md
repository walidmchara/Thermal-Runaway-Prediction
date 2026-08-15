# V1 Thermal Runaway Prediction

This V1 implements a reproducible hybrid regression baseline for battery thermal-runaway severity prediction.

Pipeline:

```text
Physical + categorical battery descriptors
        ↓
Preprocessing
        ↓
MLP regressor
        ↓
Initial severity estimate
        ↓
CatBoost residual correction
        ↓
Final Severity Index
```

## Run

```bash
pip install -r requirements.txt
python -m src.train --config configs/thermal.yaml
python -m src.evaluate --config configs/thermal.yaml
```

Metrics:
- MAE
- RMSE
- MAPE
- R²
