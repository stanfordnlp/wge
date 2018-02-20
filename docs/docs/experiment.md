# Experiment Management

All training runs are managed by the `MiniWoBTrainingRuns` object. For example,
to get training run #141, do this:
```python
runs = MiniWoBTrainingRuns()
run = runs[141]  # a MiniWoBTrainingRun object
```

A `TrainingRun` is responsible for constructing a model, training it, saving it
and reloading it (see superclasses `gtd.ml.TrainingRun` and
`gtd.ml.TorchTrainingRun` for details.)

The most important methods on `MiniWobTrainingRun` are:
- `__init__`: the policy, the environment, demonstrations, etc, are all loaded
here.
- `train`: actual training of the policy happens here
