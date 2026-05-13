"""Scan all images under data/raw/ for byte-identical duplicates via full MD5.

Writes outputs/processed/duplicates.csv with one row per duplicated file.
Does NOT modify or move any source files.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "processed" / "duplicates.csv"


def md5_of(fp: Path) -> str:
    h = hashlib.md5()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(RAW.rglob("*.jpg"))
    print(f"Hashing {len(files)} files...")
    hashes: dict[str, list[Path]] = defaultdict(list)
    for fp in tqdm(files):
        hashes[md5_of(fp)].append(fp)

    dup_groups = {h: lst for h, lst in hashes.items() if len(lst) > 1}
    rows = []
    for h, lst in dup_groups.items():
        # keep first as canonical, mark rest as duplicates
        canonical = lst[0]
        for fp in lst[1:]:
            rows.append({
                "md5": h,
                "duplicate_of": str(canonical.relative_to(RAW)),
                "duplicate_path": str(fp.relative_to(RAW)),
                "split": fp.parts[-3],
                "class": fp.parts[-2],
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"\nTotal files scanned:      {len(files)}")
    print(f"Distinct content hashes:  {len(hashes)}")
    print(f"Duplicate groups:         {len(dup_groups)}")
    print(f"Extra duplicate files:    {len(rows)}  (would be removed if deduplicated)")
    print(f"Wrote {OUT.relative_to(ROOT)}")

    if not df.empty:
        print("\nDuplicates by split/class:")
        print(df.groupby(["split", "class"]).size().to_string())


if __name__ == "__main__":
    main()
