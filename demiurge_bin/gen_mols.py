#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 2 D MOL files from SMILES strings. (parallel version)

Pipeline (jak w oryginale)
--------------------------
1.  Canonicalise the SMILES.
2.  Add explicit hydrogens.
3.  Try to embed a 3 D conformer with ETKDG (3 retries).
    • If ETKDG fails → fall back to RDKit CoordGen (2 D).
4.  Force a switch to OpenBabel for molecules that
    contain hyper-valent sulphur (valence > 4) or after an
    ETKDG failure.
    • obabel -d --gen2D strips wedge bonds and flattens the structure.
5.  Write a V3000 MOL file (2 D coordinates, no stereo wedges).
6.  Log errors to *mol_creation_error.log* and all fall-backs/
    warnings to *mol_creation_warning.log*.

Notes
-----
*   OpenBabel (`obabel`) must be on your system `PATH`
    (e.g. `conda install -c conda-forge openbabel`).
*   RDKit warnings are silenced via `RDLogger.DisableLog('rdApp.*')`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdCoordGen

# Silence all RDKit log output (also in workers via _worker_init)
RDLogger.DisableLog("rdApp.*")

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
ANSI_GREEN = "\033[38;5;46m"
ANSI_RED = "\033[38;5;196m"
ANSI_ORANGE = "\033[38;5;214m"
ANSI_RESET = "\033[0m"

PROGRESS_BAR_LEN = 25
MAX_ETKDG_RETRIES = 3
EMBED_RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────
# Helper functions (pure / picklable)
# ──────────────────────────────────────────────────────────────
def print_progress(current: int, total: int) -> None:
    """Draw a coloured, in-place ASCII progress bar."""
    filled = int(PROGRESS_BAR_LEN * current / max(1, total))
    bar = ANSI_GREEN + "█" * filled + "-" * (PROGRESS_BAR_LEN - filled) + ANSI_RESET
    percent = int(100 * current / max(1, total))
    sys.stdout.write(f"\rProgress: |{bar}| {current}/{total} ({percent}%)")
    sys.stdout.flush()
    if current >= total:
        print()  # newline


def canonical_smiles(smiles: str) -> str:
    """Return RDKit-canonical SMILES or raise *ValueError*."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def safe_embed_molecule(
    mol: Chem.Mol,
    max_retries: int = MAX_ETKDG_RETRIES,
    seed: int = EMBED_RANDOM_SEED,
) -> Tuple[Chem.Mol | None, str | None]:
    """
    Try ETKDG embedding up to *max_retries* times, fall back to CoordGen.

    Returns
    -------
    mol
        RDKit molecule with at least one conformer, or None on hard fail.
    warning
        None on ETKDG success, otherwise a human-readable note.
    """
    # Pick the newest ETKDG params available
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        try:
            params = AllChem.ETKDGv2()
        except AttributeError:
            params = AllChem.ETKDG()

    # Set only attributes guaranteed to exist across versions
    params.randomSeed = seed
    # DO NOT set params.maxAttempts – not present in some RDKit builds

    for _ in range(max_retries):
        mol.RemoveAllConformers()
        if AllChem.EmbedMolecule(mol, params) == 0:
            return mol, None  # ETKDG success

    # Fallback to 2D coords if 3D embedding failed
    try:
        rdCoordGen.AddCoords(mol)  # 2D fallback
        return mol, f"ETKDG failed ({max_retries}x) → used CoordGen"
    except Exception as exc:
        return None, f"ETKDG + CoordGen failed: {exc}"


EXOTIC_VALENCE_LIMITS = {"S": 4, "P": 4, "As": 4, "Se": 4}
TRANSITION_METALS = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                     39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                     57, 72, 73, 74, 75, 76, 77, 78, 79}

def needs_openbabel(mol: Chem.Mol, smiles: str, warn_msg: Optional[str]) -> Tuple[bool, List[str]]:
    """Heurystyka kiedy wymusić OpenBabel."""
    reasons: List[str] = []

    # hyper-valent chalcogens / pnictogens
    if any(
        a.GetSymbol() in EXOTIC_VALENCE_LIMITS
        and a.GetTotalValence() > EXOTIC_VALENCE_LIMITS[a.GetSymbol()]
        for a in mol.GetAtoms()
    ):
        reasons.append("exotic valence (S/P/As/Se)")

    # transition metals
    if any(a.GetAtomicNum() in TRANSITION_METALS for a in mol.GetAtoms()):
        reasons.append("transition metal")

    # radicals
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        reasons.append("radical")

    # very large molecules
    if mol.GetNumHeavyAtoms() > 150:
        reasons.append("very large molecule")

    # dot-SMILES
    if "." in smiles:
        reasons.append("dot-SMILES")

    # ETKDG fail earlier
    if warn_msg:
        reasons.append("ETKDG failure")

    return (len(reasons) > 0), reasons


def openbabel_fallback(rdkit_mol: Chem.Mol, out_path: str) -> Tuple[bool, str | None]:
    """
    Run *obabel* -d --gen2D on *rdkit_mol*; write to *out_path*.

    Returns *(success, error_message)*.
    """
    with tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as tmp:
        tmp.write(Chem.MolToMolBlock(rdkit_mol, forceV3000=True).encode())
        tmp_path = tmp.name

    cmd = ["obabel", tmp_path, "-O", out_path, "-d", "--gen2D"]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.remove(tmp_path)
        return True, None
    except subprocess.CalledProcessError as exc:
        os.remove(tmp_path)
        return False, exc.stderr.decode().strip()


# ──────────────────────────────────────────────────────────────
# Worker side
# ──────────────────────────────────────────────────────────────
def _worker_init() -> None:
    """Initialize per-process state (silence RDKit)."""
    RDLogger.DisableLog("rdApp.*")


def _process_one(
    name: str,
    raw_smiles: str,
    strict_mode: bool,
    output_dir: str,
) -> Tuple[bool, str, Optional[str]]:
    """
    Przerób jedną cząsteczkę → zapisz .mol.
    Returns: (success, warning_text_or_None, error_text_or_None)
    """
    try:
        smiles = canonical_smiles(raw_smiles)
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        # RDKit embedding
        mol, warn_msg = safe_embed_molecule(mol)
        if mol is None:
            raise ValueError(warn_msg)

        # 3D sanity (strict mode)
        if strict_mode:
            conf = mol.GetConformer()
            if all(conf.GetAtomPosition(i).Length() < 0.1 for i in range(mol.GetNumAtoms())):
                raise ValueError("All atoms at origin (invalid 3D)")

        # Decide if we force OpenBabel
        force_babel, reason_list = needs_openbabel(mol, smiles, warn_msg)

        # Flatten copy to 2D; (jak w oryginale – linie 2D zostawione jako komentarz)
        mol2d = Chem.Mol(mol)
        # AllChem.Compute2DCoords(mol2d)
        # Chem.RemoveStereochemistry(mol2d)

        out_path = os.path.join(output_dir, f"{name}.mol")

        if not force_babel:
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(Chem.MolToMolBlock(mol2d, forceV3000=True))
        else:
            ok, ob_err = openbabel_fallback(mol2d, out_path)
            if not ok:
                raise ValueError(f"OpenBabel fallback failed: {ob_err}")

        warn_text = None
        if warn_msg or force_babel:
            parts = []
            if warn_msg:
                parts.append(warn_msg)
            if force_babel:
                parts.append("OpenBabel fallback → " + ", ".join(reason_list))
            warn_text = f"{name}: " + " | ".join(parts)

        return True, warn_text, None

    except Exception as exc:  # pylint: disable=broad-except
        err = f"Molecule: {name}\nSMILES: {raw_smiles}\nError: {exc}\n"
        return False, None, err


# ──────────────────────────────────────────────────────────────
# Public API (parallel)
# ──────────────────────────────────────────────────────────────
def generate_mol_files(csv_path: str, strict_mode: bool = True, workers: Optional[int] = None) -> str:
    """
    Convert SMILES in *csv_path* to flat MOL files (parallel).

    Parameters
    ----------
    csv_path : str
        CSV with columns ``MOLECULE_NAME`` and ``SMILES``.
    strict_mode : bool
        If *True*, reject molecules whose 3D coords all sit at (0, 0, 0).
    workers : Optional[int]
        Number of parallel workers (default: all available CPUs).

    Returns
    -------
    str
        Output directory path.
    """
    output_dir = os.path.join(os.getcwd(), "mols")
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(csv_path)
    data = data.drop_duplicates(subset="MOLECULE_NAME", keep="first")

    jobs: List[Tuple[str, str]] = [
        (row.MOLECULE_NAME, row.SMILES) for row in data.itertuples(index=False)
    ]

    total = len(jobs)
    if total == 0:
        print(f"{ANSI_ORANGE}No rows in CSV after de-duplication by MOLECULE_NAME.{ANSI_RESET}")
        return output_dir

    # Auto-detect workers; clamp to [1, total]
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, total))

    print("\nGenerating *.mol files …")
    print(f"Using {ANSI_ORANGE}{workers}{ANSI_RESET} parallel workers")
    print_progress(0, total)

    errors: List[str] = []
    warnings: List[str] = []
    saved_files = 0
    done = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futures = [
            ex.submit(_process_one, name, smiles, strict_mode, output_dir)
            for (name, smiles) in jobs
        ]
        for fut in as_completed(futures):
            ok, warn_txt, err_txt = fut.result()
            if ok:
                saved_files += 1
            if warn_txt:
                warnings.append(warn_txt)
            if err_txt:
                errors.append(err_txt)
            done += 1
            print_progress(done, total)

    # ── Write logs ─────────────────────────────────────────────────────
    if errors:
        with open("mol_creation_error.log", "w", encoding="utf-8") as fh_err:
            fh_err.write("==== MOL CREATION ERRORS ====\n\n" + "\n".join(errors))
    if warnings:
        with open("mol_creation_warning.log", "w", encoding="utf-8") as fh_warn:
            fh_warn.write("==== MOL CREATION WARNINGS ====\n\n" + "\n".join(warnings))

    # ── Summary to console ─────────────────────────────────────────────
    print(f"\n{ANSI_GREEN}Generated {saved_files} MOL files in '{output_dir}'.{ANSI_RESET}")
    print(f"{ANSI_GREEN}Failed to generate {len(errors)} MOL files.{ANSI_RESET}")
    if errors:
        print(f"{ANSI_RED}See 'mol_creation_error.log' for details.{ANSI_RESET}")
    if warnings:
        print(f"{ANSI_ORANGE}See 'mol_creation_warning.log' for fallbacks.{ANSI_RESET}")

    return output_dir


# ──────────────────────────────────────────────────────────────
# Optional CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate V3000 .mol files from SMILES (parallel).")
    parser.add_argument("--csv", required=True, help="Path to CSV with MOLECULE_NAME and SMILES columns.")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict 3D sanity check.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all CPUs).")
    args = parser.parse_args()

    out_dir = generate_mol_files(
        csv_path=args.csv,
        strict_mode=not args.no_strict,
        workers=args.workers,
    )
    print(f"\nOutput folder: {out_dir}")
