"""Project-local model definitions.

Add your custom `nn.Module` here when the shared `src/models.py`
registry does not already cover your architecture. Follow the same
pattern as the shared module: a `build_model(config)` factory keyed on
`config["model"]["name"]`, with all hyperparameters drawn from the
config.

If you only ever use a registered model, this file can stay empty.
"""
