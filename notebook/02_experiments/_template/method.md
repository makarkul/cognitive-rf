# Method — EXX: <title>

## Code

- Commit hash: `<sha>`
- Branch: `experiment/EXX`
- Entry point: `experiments/<folder>/<script>.py`
- Single-command reproducer: `make EXX`  *(or explicit `python ... --args` if simpler)*

## Data

- Source: synthetic (seed=...) / recorded capture (path=...) / hybrid
- Distribution: channel model, SNR range, impairments, modulation, grid shape
- Train/val/test split

## Model

- Architecture: ...
- Parameter count: ...
- Hyperparameters: d_model, n_heads, n_layers, d_ff, dropout, positional embedding type

## Training

- Optimizer + schedule
- Batch size, steps, epochs
- Loss
- Hardware, wallclock

## Evaluation

- Metrics (with definitions if nonstandard)
- Baselines
- Number of independent runs / seeds

## W&B run ID

`...`
