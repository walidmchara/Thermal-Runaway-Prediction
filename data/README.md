# Thermal Runaway Dataset

Place the project CSV at:

```text
data/raw/thermal_runaway.csv
```

Default expected columns:

```text
Manufacturer
Cell_Description
Geometry
Capacity
Trigger_Mechanism
Bottom_Vent
Gravimetric_Energy_Density
Volumetric_Energy_Density
Severity_Index
```

The default V1 uses:
- one-hot encoding for categorical variables,
- standardized physical/numeric descriptors,
- an MLP regressor,
- CatBoost residual correction,
- MAE / RMSE / MAPE / R² evaluation.

Raw datasets are intentionally excluded from GitHub.
