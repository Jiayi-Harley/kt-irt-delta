# pykt-toolkit local patches

`pykt-toolkit/` (the cloned official pyKT library) is gitignored, so it is not tracked in this repo
and a fresh clone of pyKT will NOT contain the two local fixes this project relies on. This folder
preserves those fixes as a patch so they can be re-applied after re-cloning pyKT.

## What is patched

`pykt-local-changes.patch` changes two files:

1. **`pykt/models/que_base_model.py`** (`batch_to_device`): the original builds `data_new` from the
   raw dataloader tensors but never moves them to the model's device, so QueBaseModel models (QIKT,
   IEKT, ...) silently run on CPU and error out on GPU with a device mismatch. The patch moves every
   tensor in `data_new` to `self.device`. This is what lets QIKT train and the diagnostic run on the
   GPU.

2. **`pykt/preprocess/assist2009_preprocess.py`**: the raw assist2009 CSV is not valid UTF-8, so
   `pd.read_csv(..., encoding='utf-8')` fails. The patch switches the encoding to `latin-1`.

## How to apply after a fresh pyKT clone

From the `code/` directory, with `pykt-toolkit/` freshly cloned:

```
cd pykt-toolkit
git apply ../pykt-patches/pykt-local-changes.patch
```

(If `git apply` complains about line endings, `git apply --ignore-whitespace` usually resolves it.)