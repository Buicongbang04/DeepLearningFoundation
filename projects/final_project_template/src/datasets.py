"""Project-local dataset factories.

Add your custom `Dataset` and a `loaders(cfg)` function here when the
shared `src/datasets.py` registry does not already cover your data.
Re-export the function from this file so the training driver can
import it.

A typical local custom dataset is at least:

    class MyCustomDataset(torch.utils.data.Dataset):
        def __init__(self, root, train=True, transform=None):
            ...
        def __len__(self):
            ...
        def __getitem__(self, i):
            return x, y

    def my_custom_loaders(cfg):
        ...
        return train_loader, val_loader, info  # info: dict with `classes`, `num_classes`

If you only ever use a registered dataset, this file can stay empty.
"""
