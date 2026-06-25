# Cardio Diagnostics — Automation Instructions

## Firmware

- Always read serial monitor output before modifying firmware.
- Use `pio device monitor` from `firmware/` to capture live serial output.

## ML Hyperparameter Tuning

- Check training logs in `ml/logs/` before changing hyperparameters.
- Target accuracy: heart ≥ 85%, lung ≥ 82%.
- If accuracy is below target, adjust model depth or learning rate and retrain automatically.
- Log every experiment attempt to `ml/logs/experiments.md`.

## Experiment Logging Format (`ml/logs/experiments.md`)

Each entry must include:
- Date/time
- Model (heart / lung)
- Architecture changes (layers, units)
- Learning rate
- Epochs run
- Final train/val accuracy
- Whether target was met
- Next action if target not met

## Retraining Rules

1. On accuracy below target: first try adjusting learning rate (±50%).
2. If still below after 1 retry: increase model depth by one hidden layer.
3. If still below after 2 retries: reduce batch size and retrain.
4. Log every attempt regardless of outcome.
