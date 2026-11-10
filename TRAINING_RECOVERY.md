# Persistent training recovery

This PyTorch training entry point now supports atomic checkpoints after each
completed epoch, written to a location you select with `--checkpoint-dir`.
A snapshot contains model state, optimizer state, random-number generator
states, best validation state, and scheduler/AMP/early-stopping information
where used. It is a *continuation snapshot*, not a scientific result.

First run (supply **real datasets** and mount Google Drive before execution):

```bash
python -m src.train --config configs/thermal.yaml --checkpoint-dir /content/drive/MyDrive/recovery/Thermal-Runaway-Prediction/experiment_01
```

To resume with the **same** configuration after interruption:

```bash
python -m src.train --config configs/thermal.yaml --checkpoint-dir /content/drive/MyDrive/recovery/Thermal-Runaway-Prediction/experiment_01 --resume
```

On RunPod, replace the Drive directory with a mounted persistent volume.
The last incomplete epoch is repeated. The snapshot must survive in
persistent storage; Colab temporary files can disappear at disconnect.

**Project note:** The neural MLP stage is resumable; the CatBoost residual model is fitted after the neural stage completes.

A resume rejects mismatched training configuration. Avoid changing the total
number of epochs, data split, architecture, seed, or hyperparameters between
sessions. Do not use untrusted PyTorch pickle checkpoints. No manuscript
results or architectural equations have been modified. Checkpoints are
not automatically committed to GitHub by this feature.
