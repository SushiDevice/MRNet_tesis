#!/usr/bin/env python3.6
"""Create K folds over train cases (by case_id) to be shared across planes.

Usage:
  make_train_folds.py <data_dir> --k=<k> [--seed=<s>] [--output=<csv>]
  make_train_folds.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <data_dir>         Path to a directory where the data lives e.g. 'MRNet-v1.0'

Options:
  --k=<k>            Number of folds (K) [required]
  --seed=<s>         Random seed [default: 42]
  --output=<csv>     Output CSV path [default: folds_train.csv]
"""

import os
import sys
import pandas as pd
import numpy as np
from docopt import docopt
from sklearn.model_selection import KFold


def main(data_dir: str, k: int, seed: int, output_csv: str) -> None:
    labels_csv = os.path.join(data_dir, "train_labels.csv")
    df = pd.read_csv(labels_csv)
    # normalize case id to int
    df["case"] = df["case"].astype(int)
    cases = df["case"].drop_duplicates().sort_values().to_numpy()

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (_, val_idx) in enumerate(kf.split(cases), start=1):
        val_cases = cases[val_idx]
        folds.extend([{"case": int(c), "fold": int(fold_idx)} for c in val_cases])

    folds_df = pd.DataFrame(folds).sort_values(["fold", "case"]).reset_index(drop=True)
    folds_df.to_csv(output_csv, index=False)
    print(f"Folds CSV escrito: {output_csv} (K={k}, seed={seed})")


if __name__ == "__main__":
    args = docopt(__doc__)
    data_dir = args["<data_dir>"]
    k = int(args["--k"])
    seed = int(args["--seed"]) if args["--seed"] is not None else 42
    output_csv = args["--output"] or "folds_train.csv"
    main(data_dir, k, seed, output_csv)




