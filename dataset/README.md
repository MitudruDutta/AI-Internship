# Dataset — FRUIT-16K

> **Note:** the image files are **not** committed to this repository (they are
> excluded via `.gitignore`). This document describes the dataset so the
> notebooks can be reproduced once the data is placed here.

## Expected location

```
dataset/
└── FRUIT-16K/
    ├── F_Banana/      F_Lemon/      F_Lulo/      F_Mango/
    ├── F_Orange/      F_Strawberry/ F_Tamarillo/ F_Tomato/
    ├── S_Banana/      S_Lemon/      S_Lulo/      S_Mango/
    └── S_Orange/      S_Strawberry/ S_Tamarillo/ S_Tomato/
```

All notebooks expect the dataset at `../dataset/FRUIT-16K` (relative to a
`Week N/` folder).

## Description

**FRUIT-16K** is an image-classification dataset of fruit photographs for the
FreshHarvest freshness-inspection project.

| property | value |
|---|---|
| Total images | 16,000 |
| Image size | 224 × 224, RGB, JPEG |
| Classes | 16 |
| Structure | 8 fruits × 2 states (Fresh / Spoiled) |
| Balance | 1,000 images per class (raw) |

### Classes

The folder prefix encodes freshness:

- `F_` = **Fresh**
- `S_` = **Spoiled**

Eight fruits/vegetables: **Banana, Lemon, Lulo, Mango, Orange, Strawberry,
Tamarillo, Tomato.**

So the 16 classes are `F_Banana, S_Banana, F_Lemon, S_Lemon, …, F_Tomato,
S_Tomato`. A binary **fresh vs. spoiled** label (the real business target) is
derived directly from the `F_`/`S_` prefix.

## Important — duplicate images

The raw dataset contains **exact-duplicate image files** (identical bytes under
different filenames): **16,000 files but only 14,727 unique images**.

If duplicates are split naively, copies of the same image can land in both the
training and the test set, leaking test data into training and producing
**falsely perfect accuracy**. The notebooks therefore **deduplicate by MD5
content hash before splitting**, and split only over the 14,727 unique images
(with assertions that no image content crosses splits).

After dedup, the per-class unique counts are uneven (e.g. some Lemon/Lulo/
Tamarillo classes have ~550–700 unique images while others retain ~1,000), but
every split remains stratified and leakage-free.

## Splits used in the notebooks

70 / 15 / 15 train / validation / test, **stratified** over the 16 classes,
computed on unique images only, seed = 42.
