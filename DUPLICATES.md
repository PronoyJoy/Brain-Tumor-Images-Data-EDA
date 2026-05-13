# Duplicate-File Investigation

## Trigger
Phase 1 sanity checks flagged 153 fast-hash collision groups in the dataset. Five pairs were verified with full MD5 — all were byte-identical, not hash collisions.

## How to reproduce
```bash
python src/find_duplicates.py
```

## Full MD5 scan results
- **7,200 files scanned**
- **7,013 distinct content hashes**
- **153 duplicate groups** containing **187 extra duplicate files**

### Distribution

| Split | Class | Duplicates |
|---|---|---|
| Training | notumor | **119** |
| Training | pituitary | 38 |
| Training | meningioma | 14 |
| Training | glioma | 0 |
| Testing | glioma | 14 |
| Testing | meningioma | 2 |
| Testing | notumor | 0 |
| Testing | pituitary | 0 |
| **Total** | | **187** |

## What the filenames reveal
Files like `Tr-aug-me_89.jpg` (prefix suggests *augmentation*) are byte-identical to `Tr-me_889.jpg`. So the dataset's "1400 per class" was achieved by **copying originals**, not by genuine augmentation. The `aug` prefix is misleading.

`Training/glioma` is the only Training class with zero duplicates, suggesting its raw count was already at or near 1400 and no padding was needed.

## Implications
1. Effective Training set is **~5,413 unique images**, not 5,600.
2. Effective Testing set is **~1,584 unique images**, not 1,600.
3. Class balance is partly synthetic — `notumor` lost 119 (8.5 %) to duplicates; `glioma` lost 0.
4. Any model trained without dedup may overfit to whichever images were duplicated, and naive accuracy will be inflated.
5. The Testing duplicates within `Training/glioma` — actually, here Testing has 14 dups in glioma — should be checked for train-test leakage in a future phase.

## What was done
- All 7,200 files hashed with full MD5.
- `outputs/processed/duplicates.csv` lists every redundant file with the path it duplicates.
- **No files were moved or deleted.** Source data is unchanged.

## What was NOT done (open follow-ups)
- Train/test leakage check (does any test file MD5-match a train file?).
- Optional `dedup=True` flag in `build_tf_dataset` to filter duplicates from training.

## Files added
- `src/find_duplicates.py`
- `outputs/processed/duplicates.csv`
