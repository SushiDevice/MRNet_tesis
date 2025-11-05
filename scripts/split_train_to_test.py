#!/usr/bin/env python
"""Create a new train/test split from the current train using stratified sampling
without modifying the original files.

Usage:
  split_train_to_test.py <data_dir> [--test-size=<n>] [--min-pos=<k>] [--seed=<s>] [--prefix=<name>] [--patient-map=<csv>]
  split_train_to_test.py (-h | --help)

General options:
  -h --help            Show this screen.

Arguments:
  <data_dir>           Path to MRNet data dir e.g. 'MRNet-v1.0'

Options:
  --test-size=<n>      Target number of exams in test [default: 120]
  --min-pos=<k>        Minimum positives per label in test [default: 50]
  --seed=<s>           Random seed [default: 42]
  --prefix=<name>      Prefix for new split dirs/files (creates train_<p>, test_<p>) [default: split1]
  --patient-map=<csv>  Optional CSV with columns: case,patient_id to keep all
                       exams from the same patient in the same split.
"""

import os
import sys
import shutil
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from docopt import docopt


DIAGNOSES = ["abnormal", "acl", "meniscus"]
PLANES = ["axial", "coronal", "sagittal"]


def _ensure_dirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _read_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # ensure correct dtypes
    df["case"] = df["case"].astype(int)
    for d in DIAGNOSES:
        df[d] = df[d].astype(int)
    return df


def _aggregate_by_patient(train_df: pd.DataFrame, mapping_csv: str) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """Returns a patient-level dataframe (one row per patient) and a map patient->cases.

    The patient-level labels are the max over their exams per diagnosis.
    """
    mapping = pd.read_csv(mapping_csv)
    # expected columns: case, patient_id
    mapping = mapping[["case", "patient_id"]].copy()
    mapping["case"] = mapping["case"].astype(int)

    merged = train_df.merge(mapping, on="case", how="left")
    if merged["patient_id"].isna().any():
        missing = merged[merged["patient_id"].isna()]["case"].tolist()
        raise ValueError(f"patient_id faltante para cases: {missing[:5]}... total {len(missing)}")

    agg = merged.groupby("patient_id")[DIAGNOSES].max().reset_index()
    # build map patient->cases
    patient_to_cases: Dict[int, List[int]] = (
        merged.groupby("patient_id")["case"].apply(lambda s: sorted(s.astype(int).unique().tolist())).to_dict()
    )
    return agg, patient_to_cases


def _labels_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[DIAGNOSES].values.astype(int)


def _greedy_cover_minimum(df: pd.DataFrame, min_pos: int, rng: np.random.RandomState) -> List[int]:
    """Greedy selection to ensure at least min_pos positives per label.

    Works for both case-level (index is case id) and patient-level (index is patient id).
    Returns list of selected indices (values of df.index).
    """
    labels = _labels_matrix(df)
    remaining_idx = df.index.to_numpy()
    rng.shuffle(remaining_idx)

    selected: List[int] = []
    counts = labels[selected].sum(axis=0) if selected else np.zeros(labels.shape[1], dtype=int)

    def deficit() -> np.ndarray:
        return np.maximum(0, min_pos - counts)

    remaining_set = set(remaining_idx.tolist())
    while np.any(deficit() > 0):
        # score each remaining by how much it helps reduce deficits
        best_idx = None
        best_score = -1
        for idx in list(remaining_set):
            row = labels[df.index.get_loc(idx)]
            score = int((row > 0) @ deficit())  # how many deficits it covers (weighted)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score == 0:
            # not enough positives to satisfy constraint
            break
        selected.append(best_idx)
        counts += labels[df.index.get_loc(best_idx)]
        remaining_set.remove(best_idx)

    return selected


def _fill_to_size(pool_indices: List[int], target_count: int, rng: np.random.RandomState) -> List[int]:
    if len(pool_indices) <= target_count:
        return list(pool_indices)
    return rng.choice(pool_indices, size=target_count, replace=False).tolist()


def _copy_cases_to_subdir(cases: List[int], data_dir: str, subdir: str) -> None:
    """Create <subdir>/{planes} folders and copy files for each plane from train.

    Files are taken from train/<plane>/<case>.npy and placed into <subdir>/<plane>/.
    """
    split_dir = os.path.join(data_dir, subdir)
    for plane in PLANES:
        src_plane = os.path.join(data_dir, "train", plane)
        dst_plane = os.path.join(split_dir, plane)
        _ensure_dirs(dst_plane)
        for case in cases:
            fname = f"{case:04d}.npy" if os.path.exists(os.path.join(src_plane, f"{case:04d}.npy")) else f"{case}.npy"
            src = os.path.join(src_plane, fname)
            dst = os.path.join(dst_plane, fname)
            if not os.path.exists(src):
                # try raw case id if zero-padded not found
                src = os.path.join(src_plane, f"{case}.npy")
            if not os.path.exists(src):
                raise FileNotFoundError(f"No se encuentra el archivo del caso {case} en {src_plane}")
            shutil.copy2(src, dst)


def _write_labels_csv(df: pd.DataFrame, cases: List[int], out_csv: str) -> None:
    out = df[df["case"].isin(cases)].copy()
    out = out[["case"] + DIAGNOSES]
    out.to_csv(out_csv, index=False)


def _write_remaining_labels(train_df: pd.DataFrame, remaining_cases: List[int], out_csv: str) -> None:
    keep = train_df[train_df["case"].isin(remaining_cases)].copy()
    keep.to_csv(out_csv, index=False)


def main(data_dir: str,
         test_size: int,
         min_pos: int,
         seed: int,
         prefix: str,
         patient_map_csv: Optional[str]) -> None:
    rng = np.random.RandomState(seed)

    train_labels_csv = os.path.join(data_dir, "train_labels.csv")
    train_df = _read_labels(train_labels_csv)

    # Build selection space (case-level or patient-level)
    if patient_map_csv is not None:
        entity_df, patient_to_cases = _aggregate_by_patient(train_df, patient_map_csv)
        entity_df = entity_df.set_index("patient_id")
        # greedy satisfy minimum
        selected_patients = _greedy_cover_minimum(entity_df, min_pos, rng)
        # expand to cases
        sel_cases = sorted({c for p in selected_patients for c in patient_to_cases[int(p)]})
        # fill to target exam count (approximate; may overshoot/undershoot due to grouping)
        remaining_patients = [p for p in entity_df.index.tolist() if p not in selected_patients]
        # build pool of remaining cases grouped by patient
        remaining_cases = [c for p in remaining_patients for c in patient_to_cases[int(p)]]
        if len(sel_cases) < test_size:
            add_needed = test_size - len(sel_cases)
            add_cases = _fill_to_size(remaining_cases, add_needed, rng)
            sel_cases = sorted(set(sel_cases + add_cases))
    else:
        # case-level
        case_df = train_df.set_index("case")
        selected_cases = _greedy_cover_minimum(case_df, min_pos, rng)
        sel_cases = sorted(selected_cases)
        # fill randomly to desired size
        remaining_cases = [c for c in case_df.index.tolist() if c not in sel_cases]
        if len(sel_cases) < test_size:
            add_needed = test_size - len(sel_cases)
            sel_cases += _fill_to_size(remaining_cases, add_needed, rng)
            sel_cases = sorted(set(sel_cases))

    # Safety checks
    test_df = train_df[train_df["case"].isin(sel_cases)].copy()
    pos_counts = test_df[DIAGNOSES].sum(axis=0).to_dict()
    if any(pos_counts[d] < min_pos for d in DIAGNOSES):
        print("[ADVERTENCIA] No se alcanzó el mínimo de positivos para alguna etiqueta.")
        print({d: int(pos_counts[d]) for d in DIAGNOSES})

    # Compute remaining cases for the new train split
    all_cases = train_df["case"].tolist()
    remaining_cases = sorted([c for c in all_cases if c not in sel_cases])

    # Create new split dirs and copy files (originals remain intact)
    test_subdir = f"test_{prefix}"
    train_subdir = f"train_{prefix}"
    _copy_cases_to_subdir(sel_cases, data_dir, test_subdir)
    _copy_cases_to_subdir(remaining_cases, data_dir, train_subdir)

    # Write labels for the new split
    _write_labels_csv(train_df, sel_cases, os.path.join(data_dir, f"test_{prefix}_labels.csv"))
    _write_remaining_labels(train_df, remaining_cases, os.path.join(data_dir, f"train_{prefix}_labels.csv"))

    # Report
    total_train_before = len(train_df)
    total_test = len(sel_cases)
    print("=== Split resumen ===")
    print(f"Casos train (origen): {total_train_before}")
    print(f"Casos test_{prefix} (creados): {total_test}")
    print(f"Dir nuevo train: {train_subdir}")
    print(f"Dir nuevo test:  {test_subdir}")
    print("Positivos en test:", {d: int(pos_counts[d]) for d in DIAGNOSES})


if __name__ == "__main__":
    args = docopt(__doc__)

    data_dir = args["<data_dir>"]
    test_size = int(args["--test-size"]) if args["--test-size"] is not None else 120
    min_pos = int(args["--min-pos"]) if args["--min-pos"] is not None else 50
    seed = int(args["--seed"]) if args["--seed"] is not None else 42
    prefix = args["--prefix"] or "split1"
    patient_map_csv = args["--patient-map"]

    main(data_dir, test_size, min_pos, seed, prefix, patient_map_csv)


