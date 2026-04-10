## Bottom-3 Checkpoint

This branch captures the cleaned checkpoint where:

- the standalone bottom-3 experiment exists under `inverse_bottom3/`
- the main `train.py` keeps the bottom-3 integration path
- the PyTorch side-model has been removed from the main trainer
- the XGBoost blend is forced to single-threaded execution on macOS to avoid
  the OpenMP crash that was causing `Python quit unexpectedly`

### Standalone bottom-3

Command:

```bash
./.venv/bin/python inverse_bottom3/run_experiment.py \
  --groups relative,specialization,race_context \
  --tag standalone \
  --iterations 100
```

Reproduced metrics:

- `val_bottom3_slot_accuracy=0.625491`
- `test_bottom3_slot_accuracy=0.552304`
- `val_runner_auc=0.771257`
- `test_runner_auc=0.696613`

### Integrated winner-model test

Command:

```bash
PYTHONUNBUFFERED=1 ./.venv/bin/python train.py
```

Reproduced metrics:

- `val_top1_accuracy=0.469929`
- `test_top1_accuracy=0.313735`
- `bottom3_alpha=0.00`
- `bottom3_groups=relative,specialization,race_context`

### Current conclusion

The bottom-3 model works as a standalone task, but the current global penalty
integration does not improve the winner stack. The best integrated setting is
still `alpha=0.00`.
