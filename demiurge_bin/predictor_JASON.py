#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch predictor: reads .mol files, pulls predicted 1H/13C shifts via BeautifulJASON,
and exports per-molecule CSVs containing only the "shift" column.

Usage (optional CLI):
    python bj_predict.py --mol-dir "path/to/folder" --predictor 1H
    python bj_predict.py --mol-dir "path/to/folder" --predictor 13C
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from typing import List, Tuple

import pandas as pd
import beautifuljason as bjason

# ANSI colors
COLORS = [
    "\033[38;5;46m",   # Green
    "\033[38;5;196m",  # Red
    "\033[38;5;214m",  # Orange
]
RESET = "\033[0m"

# Short green ASCII progress bar (like your example)
PROGRESS_BAR_LEN = 25
ANSI_GREEN = COLORS[0]
ANSI_RESET = RESET


def print_progress(current: int, total: int) -> None:
    """Draw a coloured, in-place ASCII progress bar."""
    if total <= 0:
        total = 1
    filled = int(PROGRESS_BAR_LEN * current / total)
    bar = ANSI_GREEN + "█" * filled + "-" * (PROGRESS_BAR_LEN - filled) + ANSI_RESET
    percent = int(100 * current / total)
    sys.stdout.write(f"\rProgress: |{bar}| {current}/{total} ({percent}%)")
    sys.stdout.flush()
    if current >= total:
        print()  # newline at the end


def _numeric_key(series: pd.Series) -> List[Tuple[int, str]]:
    """
    Sort helper for labels like H1, H2(abc), C13, etc.
    Returns list of (number, mark) tuples for stable natural ordering.
    """
    match = series.str.extract(r"^\D*(\d+)(?:\(([^)]*)\))?$")
    idx = pd.to_numeric(match[0], errors="coerce").fillna(10**9).astype(int)
    mark = match[1].fillna("~").str.casefold()
    return list(zip(idx, mark))


def _normalize_predictor(predictor: str) -> str:
    """
    Normalize user input to '1H' or '13C'.
    Accepts variants like 'h1', '1h', 'c13', '13c'.
    """
    p = predictor.strip().upper().replace(" ", "")
    if p in {"1H", "H1"}:
        return "1H"
    if p in {"13C", "C13"}:
        return "13C"
    raise ValueError("Predictor must be '1H' or '13C'.")


def _extract_shifts_for_predictor(mol: bjason.Molecule, predictor: str) -> pd.DataFrame:
    """
    Extract predicted shifts for the requested nucleus into a single-column DataFrame 'shift'.
    """
    predictor = _normalize_predictor(predictor)

    target = (
        bjason.Molecule.Atom.NuclType.H1 if predictor == "1H"
        else bjason.Molecule.Atom.NuclType.C13
    )

    rows = []
    element = "H" if predictor == "1H" else "C"

    for spec in mol.spectra:
        if spec.nucleus != target:
            continue
        for shift in spec.shifts:
            try:
                value = shift.value[-1]  # heuristic: last value
            except Exception:
                continue
            for num in shift.nums:
                atom_id = element + str(num) + (f"({shift.mark})" if shift.mark else "")
                rows.append((atom_id, value))

    if not rows:
        return pd.DataFrame(columns=["shift"])

    df = (
        pd.DataFrame(rows, columns=["atom_id", "shift"])
        .sort_values("atom_id", key=_numeric_key)
        .reset_index(drop=True)
    )
    return df[["shift"]]


def JASON_predictor(mol_directory: str, predictor: str) -> str:
    """
    Process all .mol files in 'mol_directory' and write CSVs with predicted shifts
    for the requested 'predictor' ('1H' or '13C').

    The CSV files contain only a single column 'shift' (no header, no index),
    saved under a new folder in the current working directory:
        ./predicted_spectra_{predictor}/<mol_basename>.csv

    Returns
    -------
    str
        Absolute path to the created CSV output folder.
    """
    predictor_norm = _normalize_predictor(predictor)

    csv_output_folder = os.path.join(
        os.getcwd(), f"predicted_spectra_{predictor_norm}"
    )
    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder, exist_ok=True)
        print(f"\nCreated directory: {COLORS[2]}{csv_output_folder}{RESET}")

    if not os.path.isdir(mol_directory):
        raise NotADirectoryError(f"Directory not found: {mol_directory}")

    mol_files = [
        os.path.join(mol_directory, f)
        for f in os.listdir(mol_directory)
        if f.lower().endswith(".mol")
    ]
    total = len(mol_files)
    if total == 0:
        print(f"No .mol files found in: {mol_directory}")
        return os.path.abspath(csv_output_folder)

    errors: List[Tuple[str, str]] = []
    success = 0

    jason = bjason.JASON(plugins=None)  # load all plugins

    print(f"\nPrediction of {predictor} spectra in progres ...")
    print_progress(0, total)

    for idx, mol_path in enumerate(mol_files, start=1):
        try:
            abs_path = os.path.abspath(mol_path)
            with jason.create_document(abs_path, rules="off") as doc:
                mols = list(doc.mol_data)
                if not mols:
                    raise RuntimeError("No molecules present in the document.")
                mol = mols[0]

                df = _extract_shifts_for_predictor(mol, predictor_norm)
                if df.empty:
                    raise RuntimeError(
                        f"No predicted {predictor_norm} shifts found in the molecule."
                    )

                # Sort by value ascending before saving
                df["shift"] = pd.to_numeric(df["shift"], errors="coerce")
                df = df.sort_values(
                    "shift", ascending=True, kind="mergesort", na_position="last"
                )

                out_name = os.path.splitext(os.path.basename(abs_path))[0] + ".csv"
                out_path = os.path.join(csv_output_folder, out_name)

                # Save only the 'shift' column, no header and no index
                df.to_csv(out_path, index=False, header=False)
                success += 1

        except Exception as exc:
            errors.append((mol_path, f"{type(exc).__name__}: {exc}"))
        finally:
            print_progress(idx, total)

    print(
        f"\n{COLORS[0]}Done.{RESET} Successfully predicted "
        f"{success}/{total} {predictor} spectra and saved to CSV files in: {csv_output_folder}"
    )

    if errors:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(csv_output_folder, f"errors_{ts}.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            for path, msg in errors:
                logf.write(f"{path} :: {msg}\n")
        print(
            f"{COLORS[1]}There were {len(errors)} failures. "
            f"See log: {log_path}{RESET}"
        )

    return os.path.abspath(csv_output_folder)


# Optional CLI wrapper — all operational logic lives in JASON_predictor()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict NMR shifts from .mol files and export CSVs with 'shift' only."
    )
    parser.add_argument(
        "--mol-dir",
        type=str,
        required=True,
        help="Path to the input directory containing .mol files.",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        choices=["1H", "13C", "h1", "1h", "c13", "13c"],
        required=True,
        help="Which nucleus to export: 1H or 13C.",
    )
    args = parser.parse_args()
    out_dir = JASON_predictor(args.mol_dir, args.predictor)
    print(f"\nCSV output folder: {out_dir}")
