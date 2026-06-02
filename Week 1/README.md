# Week 1 — Data Pipeline & a CNN from Scratch

This week builds the full data pipeline for **FRUIT-16K** and trains a
convolutional neural network **from scratch** (no transfer learning), first as a
plain baseline and then with regularization and hyperparameter tuning.

## Notebook

[`FreshHarvest_Week1.ipynb`](FreshHarvest_Week1.ipynb) — runs top to bottom.

### Sections

1. **Load** the dataset with `torchvision.datasets.ImageFolder`.
2. **Deduplicate** by MD5 content hash (the raw data has exact duplicates — see
   the [dataset README](../dataset/README.md)) so no image leaks across splits.
3. **Augment** the training data (random flip, rotation, colour-jitter) +
   ImageNet normalization; validation/test get a deterministic resize only.
4. **Split** 70 / 15 / 15 train / val / test, stratified over the 16 classes,
   on unique images only.
5. **Visualize** raw samples, fresh-vs-spoiled pairs, and an augmented batch.
6. **Train a CNN from scratch:**
   - a **plain** VGG-style CNN (no dropout / batch-norm / weight-decay) as a
     baseline, with epoch tuning via the validation curve;
   - a **regularized + tuned** version adding **batch normalization, dropout,
     weight decay, and early stopping**, with a small hyperparameter grid
     (learning rate × weight decay × dropout) selected on validation.
7. **Save** the best regularized model as `best_model.pth`.

## Targets

- **16-class** — fruit × state (`F_Banana` … `S_Tomato`), the primary metric.
- **Binary fresh / spoiled** — the business outcome, derived from the 16-class
  predictions.

Goal: **> 90 %** accuracy on validation and test.

## Run

```bash
# from this folder, with the project virtualenv active
jupyter notebook FreshHarvest_Week1.ipynb
# or execute headless:
jupyter nbconvert --to notebook --execute --inplace FreshHarvest_Week1.ipynb
```

Requires the dataset at [`../dataset/FRUIT-16K`](../dataset/README.md) and a
CUDA GPU is recommended (CPU works but is slow).

## Outputs

- `best_model.pth` — the best regularized CNN (state_dict + config + class
  names). Excluded from git; regenerate by running the notebook.

> The committed notebook is saved without cell outputs. Run it to reproduce the
> figures and the validation/test accuracy (which clear the 90 % target after
> deduplication).
