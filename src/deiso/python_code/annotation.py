"""
FFC (Fragment-Fragment Correlation) Annotation Pipeline
========================================================
Annotates mass spectrometry FFC data for peptide fragmentation analysis.

Pipeline steps:
    1. Partition m/z pairs by charge state combinations
    2. Select partitions matching parental isotopic envelope
    3. Annotate fragment ions (b/y) with optional neutral loss & isotope shifts
    4. Build a coverage table summarizing bond-by-bond sequence coverage
    5. Post-process: deduplicate shared FFCs, compute isocolumns, overlap stats

External dependencies:
    - peptide  (custom module providing Pep class with ion_mass())
    - interpreter_modify  (custom module, unused in core logic but imported at caller level)
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Physical Constants ──────────────────────────────────────────────────────
ISOTOPE_OFFSET = 1.00335  # Da, average spacing between isotopic peaks
MASS_H = 1.00784          # Da, hydrogen mass for charge-state adjustment


# =============================================================================
# 1. CHARGE-STATE PARTITIONING
# =============================================================================

def partition_dataframe_by_charge(
    df: pd.DataFrame,
    charge_list: Iterable[int],
    mz_col_a: str = "m/z A",
    mz_col_b: str = "m/z B",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    For each total charge z, create columns for every (z_A, z_B) split where
    z_A + z_B = z.  Each column holds z_A * mz_A + z_B * mz_B, i.e. the
    reconstructed *neutral* parental mass candidate.

    Parameters
    ----------
    df : DataFrame
        Must contain ``mz_col_a`` and ``mz_col_b``.
    charge_list : iterable of int
        Charge states to consider (duplicates and None are ignored; z < 2 skipped).
    mz_col_a, mz_col_b : str
        Column names for the two m/z values.

    Returns
    -------
    result_df : DataFrame
        Copy of *df* with added component and partition-sum columns.
    partition_names : list[str]
        Names of the partition-sum columns (used downstream for matching).
    """
    result_df = df.copy()
    partition_names: List[str] = []

    # Deduplicate charges, preserving order
    unique_charges = list(dict.fromkeys(
        int(c) for c in charge_list if c is not None and int(c) >= 2
    ))

    for charge in unique_charges:
        for z_a in range(1, charge):
            z_b = charge - z_a

            sum_col = f"{z_a}*{mz_col_a} + {z_b}*{mz_col_b}"
            partition_names.append(sum_col)

            comp_a = f"comp_{z_a}_{mz_col_a}"
            comp_b = f"comp_{z_b}_{mz_col_b}"

            if comp_a not in result_df.columns:
                result_df[comp_a] = z_a * result_df[mz_col_a]
            if comp_b not in result_df.columns:
                result_df[comp_b] = z_b * result_df[mz_col_b]

            result_df[sum_col] = result_df[comp_a] + result_df[comp_b]

    return result_df, partition_names


# =============================================================================
# 2. PARENTAL-MASS MATCHING (with isotopic envelope)
# =============================================================================

def select_best_partition(
    df: pd.DataFrame,
    keep_cols: List[str],
    target_mass: float,
    threshold: float,
    partition_names: List[str],
    iso_range: int = 0,
) -> pd.DataFrame:
    """
    For every FFC row, find the charge-split whose reconstructed mass best
    matches the target parental mass (within *threshold*), optionally
    considering isotopic satellites up to *iso_range*.

    Returns a DataFrame with the original ``keep_cols`` plus:
        selected_total, component_x1, component_x2,
        charge_A, charge_B, adj_mass_A, adj_mass_B,
        source_column, deviation, parent_isotope_idx
    """
    target_masses = [target_mass + i * ISOTOPE_OFFSET for i in range(iso_range + 1)]
    all_results: List[pd.DataFrame] = []

    for t_mass in target_masses:
        deviations = df[partition_names].sub(t_mass).abs()
        min_dev = deviations.min(axis=1)
        best_col = deviations.idxmin(axis=1)
        mask = min_dev <= threshold

        if not mask.any():
            continue

        subset = df.loc[mask, keep_cols].copy()
        row_indices = np.where(mask)[0]

        records = []
        for idx, col_name in zip(row_indices, best_col[mask]):
            w1, w2 = (int(w) for w in re.findall(r"(\d+)\*", col_name))
            val_a = df.iloc[idx][f"comp_{w1}_m/z A"]
            val_b = df.iloc[idx][f"comp_{w2}_m/z B"]

            records.append({
                "selected_total": df.iloc[idx][col_name],
                "component_x1": val_a,
                "component_x2": val_b,
                "charge_A": w1,
                "charge_B": w2,
                # Convert to singly-charged mass for b/y matching
                "adj_mass_A": val_a - (w1 - 1) * MASS_H,
                "adj_mass_B": val_b - (w2 - 1) * MASS_H,
            })

        extra = pd.DataFrame(records, index=subset.index)
        subset = pd.concat([subset, extra], axis=1)
        subset["source_column"] = best_col[mask].values
        subset["deviation"] = min_dev[mask].values
        subset["parent_isotope_idx"] = (t_mass - target_mass) / ISOTOPE_OFFSET

        all_results.append(subset)

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results)
    # Keep the best (lowest deviation) match per original row
    return (
        combined
        .sort_values("deviation")
        .drop_duplicates(subset=keep_cols)
        .reset_index(drop=True)
    )


# =============================================================================
# 3. FRAGMENT-ION ANNOTATION
# =============================================================================

def _build_theoretical_ions(pep, iso_range: int = 0) -> List[dict]:
    """
    Build a list of theoretical b/y ions with isotopic variants.

    Each entry: {'base_name': 'b3', 'iso': 0, 'name': '0', 'mass': 123.456}
    The 'name' field encodes the isotope shift label (e.g. '0', '+1', '+2').
    """
    ions = []
    for i in range(1, pep.pep_len):
        for ion_type in ("b", "y"):
            base_name = f"{ion_type}{i}"
            base_mass = pep.ion_mass(base_name)
            for iso in range(iso_range + 1):
                label = "0" if iso == 0 else f"+{iso}"
                ions.append({
                    "base_name": base_name,
                    "iso": iso,
                    "name": label,
                    "mass": base_mass + iso * ISOTOPE_OFFSET,
                })
    return ions


def _find_all_matches(observed_mass: float, theoretical_ions: List[dict],
                      threshold: float, mono_only: bool = False) -> List[dict]:
    """
    Return all theoretical ions within *threshold* of *observed_mass*.
    If *mono_only*, only consider monoisotopic (iso == 0) ions.
    Falls back to a single None-placeholder when nothing matches.
    """
    matches = []
    for ion in theoretical_ions:
        if mono_only and ion["iso"] != 0:
            continue
        diff = abs(observed_mass - ion["mass"])
        if diff <= threshold:
            matches.append({
                "name": ion["name"],
                "dev": diff,
                "theo": ion["mass"],
                "iso": ion["iso"],
                "base_name": ion["base_name"],
            })
    if not matches:
        return [{"name": None, "dev": None, "theo": None,
                 "iso": None, "base_name": None}]
    return matches


def annotate_dataframe(df: pd.DataFrame, pep, threshold: float) -> pd.DataFrame:
    """
    Simple monoisotopic b/y annotation (no neutral loss, no isotope expansion).

    Adds columns: explanation_A/B, deviation_A/B, theoretical_mass_A/B
    """
    theoretical_ions = {
        f"{t}{i}": pep.ion_mass(f"{t}{i}")
        for t in ("b", "y")
        for i in range(1, pep.pep_len)
    }

    def _best_match(observed_mass: float):
        best_name, best_dev, best_theo = None, None, None
        min_diff = float("inf")
        for name, theo_mass in theoretical_ions.items():
            diff = abs(observed_mass - theo_mass)
            if diff <= threshold and diff < min_diff:
                min_diff = diff
                best_name, best_dev, best_theo = name, diff, theo_mass
        return best_name, best_dev, best_theo

    for side in ("A", "B"):
        results = df[f"adj_mass_{side}"].apply(_best_match)
        df[f"explanation_{side}"] = [r[0] for r in results]
        df[f"deviation_{side}"] = [r[1] for r in results]
        df[f"theoretical_mass_{side}"] = [r[2] for r in results]

    return df


def annotate_dataframe_iso(
    df: pd.DataFrame, pep, threshold: float, iso_range: int = 0,
) -> pd.DataFrame:
    """
    Annotate with isotopic variants.  Produces a cross-product expansion:
    each row may yield multiple rows (one per (match_A, match_B) combination).
    """
    ions = _build_theoretical_ions(pep, iso_range)

    new_rows = []
    for _, row in df.iterrows():
        matches_a = _find_all_matches(row["adj_mass_A"], ions, threshold)
        matches_b = _find_all_matches(row["adj_mass_B"], ions, threshold)
        for ma in matches_a:
            for mb in matches_b:
                new_row = row.copy()
                for side, m in [("A", ma), ("B", mb)]:
                    new_row[f"explanation_{side}"] = m["name"]
                    new_row[f"deviation_{side}"] = m["dev"]
                    new_row[f"theoretical_mass_{side}"] = m["theo"]
                    new_row[f"iso_{side}"] = m["iso"]
                    new_row[f"base_name_{side}"] = m["base_name"]
                new_rows.append(new_row)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def annotate_dataframe_loss(
    df: pd.DataFrame,
    pep,
    threshold: float,
    diffs: List[float],
    charge: int,
    iso_range: int = 0,
) -> pd.DataFrame:
    """
    Annotate allowing for neutral losses.

    For each theoretical ion, additional shifted masses are generated:
        ion_mass - (loss + n * ISOTOPE_OFFSET)  for n in 0..iso_range

    The `diffs` list holds the neutral-loss values to consider.  When
    diff == 0 (parent line), only monoisotopic matches are kept.  For
    small losses (0 <= diff < 71.037 Da), rows are filtered to those
    whose charges sum to the precursor charge.
    """
    base_ions = _build_theoretical_ions(pep, iso_range)

    # Extend with neutral-loss shifted masses
    extended_ions = list(base_ions)  # start with the originals
    for ion in base_ions:
        for loss in diffs:
            if loss > 0:
                loss_variants = [
                    loss + n * ISOTOPE_OFFSET for n in range(iso_range + 1)
                ] if iso_range > 0 else [loss]
            else:
                loss_variants = [0]

            for shifted_loss in loss_variants:
                if shifted_loss == 0:
                    continue
                extended_ions.append({
                    "base_name": ion["base_name"],
                    "iso": ion["iso"],
                    "name": f'({ion["name"]})-{round(shifted_loss, 3)}',
                    "mass": ion["mass"] - shifted_loss,
                })

    mono_only = (diffs[0] == 0)
    enforce_charge_sum = (0 <= diffs[0] < 71.037)

    new_rows = []
    for _, row in df.iterrows():
        matches_a = _find_all_matches(row["adj_mass_A"], extended_ions, threshold, mono_only)
        matches_b = _find_all_matches(row["adj_mass_B"], extended_ions, threshold, mono_only)

        for ma in matches_a:
            for mb in matches_b:
                if enforce_charge_sum and (row["charge_A"] + row["charge_B"] != charge):
                    continue

                new_row = row.copy()
                for side, m in [("A", ma), ("B", mb)]:
                    new_row[f"explanation_{side}"] = m["name"]
                    new_row[f"deviation_{side}"] = m["dev"]
                    new_row[f"theoretical_mass_{side}"] = m["theo"]
                    new_row[f"iso_{side}"] = m["iso"]
                    new_row[f"base_name_{side}"] = m["base_name"]
                new_rows.append(new_row)

    return pd.DataFrame(new_rows).reset_index(drop=True)


# =============================================================================
# 4. CLASSIFICATION & COMPLEMENTARY ION HELPERS
# =============================================================================

def classify_isotopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'isotopic_classification' column describing the isotope state
    of each annotated FFC (e.g. 'Parent', 'b+1, y+2', etc.).
    """
    _suffix_re = re.compile(r"([by])\d*\+(\d+)")

    def _get_suffix(name):
        if not isinstance(name, str):
            return None
        m = _suffix_re.search(name)
        return f"{m.group(1)}+{m.group(2)}" if m else None

    def _classify(row):
        suffixes = sorted(filter(None, [
            _get_suffix(row["explanation_A"]),
            _get_suffix(row["explanation_B"]),
        ]))
        if suffixes:
            return ", ".join(suffixes)
        if float(row["parent_isotope_idx"]) == 0:
            return "Parent"
        return f"Parent+{int(row['parent_isotope_idx'])} (Unannotated)"

    df["isotopic_classification"] = df.apply(_classify, axis=1)
    return df


def get_complementary_ion(pep, base_name: str) -> str:
    """
    Given 'b5' return 'y{L-5}' and vice-versa, where L = peptide length.
    """
    ion_type = base_name[0]
    index = int(base_name[1:])
    length = len(pep.AA_array)
    return f"y{length - index}" if ion_type == "b" else f"b{length - index}"


# =============================================================================
# 5. COVERAGE TABLE
# =============================================================================

def _build_row_labels(pep) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Produce row labels ('b1, y{L-1}', ...) and theoretical mass pairs.
    """
    length = len(pep.AA_array)
    labels = [f"b{i}, y{length - i}" for i in range(1, length)]
    masses = [
        (round(pep.ion_mass(f"b{i}"), 3), round(pep.ion_mass(f"y{length - i}"), 3))
        for i in range(1, length)
    ]
    return labels, masses


def _line_key(loss_value: float):
    """Convert a loss-list value to the column key used in the coverage table."""
    return "parent" if loss_value == 0 else -loss_value


def coverage_table(
    df: pd.DataFrame,
    loss_list: List[float],
    pep,
    iso: int,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Build a bond-by-bond coverage table.

    Rows   = backbone cleavage sites (b_i, y_{L-i})
    Columns = neutral-loss lines from *loss_list*

    Cells contain annotation strings describing which isotopic variant
    was matched, the charge split, mass deviations, and FFC ranking.
    """
    length = len(pep.AA_array)
    row_labels, theo_masses = _build_row_labels(pep)
    columns = [_line_key(v) for v in loss_list]

    # ── Initialise per-column dictionaries ────────────────────────────────
    annot_table = {col: {r: None for r in row_labels} for col in columns}
    iso_tracker = {col: {r: float("inf") for r in row_labels} for col in columns}

    # False-positive counters
    fpc_both_unknown = {col: 0 for col in columns}
    fpc_one_unknown = {col: 0 for col in columns}
    ffc_counts = {col: 0 for col in columns}

    # Sample lists for diagnostics (keep up to 3 examples)
    MAX_EXAMPLES = 3
    examples_both = {col: [] for col in columns}     # (mz_A, mz_B, charges, rank)
    examples_both_t = {col: [] for col in columns}   # total-mass form
    examples_one = {col: [] for col in columns}
    examples_one_t = {col: [] for col in columns}

    # ── Populate table ────────────────────────────────────────────────────
    for loss_val in loss_list:
        col = _line_key(loss_val)
        line_df = df[df["line"] == col]

        for _, row in line_df.iterrows():
            base_a, base_b = row["base_name_A"], row["base_name_B"]
            both_known = base_a is not None and base_b is not None
            both_unknown = base_a is None and base_b is None

            if both_known:
                _record_full_match(row, col, annot_table, iso_tracker)

            elif both_unknown:
                fpc_both_unknown[col] += 1
                if len(examples_both[col]) < MAX_EXAMPLES:
                    _record_example(row, examples_both[col], examples_both_t[col])

            else:  # one side known
                fpc_one_unknown[col] += 1
                _record_partial_match(row, col, pep, annot_table,
                                      examples_one[col], examples_one_t[col],
                                      MAX_EXAMPLES)

            ffc_counts[col] = row["num_ffc"]

    # ── Assemble DataFrame ────────────────────────────────────────────────
    table = pd.DataFrame(annot_table)

    # Sort columns by completeness (fewest missing first)
    table = table[table.isna().sum().sort_values().index]

    # Add per-row summary columns (35 rows = cleavage sites)
    table["Row Count"] = table.notna().sum(axis=1).astype("Int64")
    table["Offset Cov"] = _compute_offset_coverage(columns, pep, length, iso, threshold, row_labels)
    table["theoretical mass"] = theo_masses

    # Add "Col Count" row via concat (avoids FutureWarning from .loc assignment)
    col_count = table.notna().sum(axis=0).astype("Int64")
    col_count_df = pd.DataFrame([col_count.values], columns=table.columns, index=["Col Count"])
    table = pd.concat([table, col_count_df])
    table["AA"] = pep.AA_array

    # ── Append summary/diagnostic rows (FFC, FPR, offset, examples) ──────
    summary_df = _append_summary_rows(
        table.columns, ffc_counts, fpc_both_unknown, fpc_one_unknown,
        examples_both, examples_both_t,
        examples_one, examples_one_t,
        columns, pep, length, iso, threshold, MAX_EXAMPLES,
    )
    table = pd.concat([table, summary_df])

    # ── Coverage columns (computed after all rows exist) ──────────────────
    table["Covered"] = (table["Row Count"] != 0).map({True: "+", False: "0"})
    cov_pct = (table["Covered"] == "+").sum() / length * 100
    cov_col_name = f"Coverage={cov_pct:.1f}%"
    table.rename(columns={"Covered": cov_col_name}, inplace=True)

    # Combined coverage (direct + offset)
    table["Covered"] = _combine_coverage(table[cov_col_name], table["Offset Cov"])
    valid = table["Covered"].isin(["0", "+", "+*"])
    covered = table["Covered"].isin(["+", "+*"])
    all_cov_pct = covered[valid].mean() * 100
    all_cov_col = f"All Coverage={all_cov_pct:.1f}%"
    table.rename(columns={"Covered": all_cov_col}, inplace=True)

    # Drop intermediate coverage columns, keep combined
    table.drop(columns=["Offset Cov", cov_col_name], inplace=True)

    # ── Final column ordering: AA first, then all_cov_col ─────────────────
    ordered = ["AA", all_cov_col] + [
        c for c in table.columns if c not in ("AA", all_cov_col)
    ]
    return table[ordered]


# ── coverage_table helpers ────────────────────────────────────────────────────

def _format_annotation(bases: list) -> str:
    """
    Format a sorted [(base, explanation, charge, deviation), ...] pair
    into the annotation string stored in coverage-table cells.
    """
    b0, b1 = bases[0], bases[1]
    return (
        f"{b0[1]}, {b1[1]}, [{b0[2]}, {b1[2]}], "
        f"{b0[3], b1[3]}, [{int(bases[0][4]) if len(bases[0]) > 4 else ''}]"
    )


def _record_full_match(row, col, annot_table, iso_tracker):
    """Handle an FFC where both fragments are identified."""
    bases = [
        (row["base_name_A"], row["explanation_A"], row["charge_A"],
         round(row["deviation_A"], 3)),
        (row["base_name_B"], row["explanation_B"], row["charge_B"],
         round(row["deviation_B"], 3)),
    ]
    bases.sort()
    pair_key = f"{bases[0][0]}, {bases[1][0]}"

    if pair_key not in annot_table[col]:
        return

    iso_sum = row["iso_A"] + row["iso_B"]
    if iso_sum < iso_tracker[col][pair_key]:
        annot_table[col][pair_key] = (
            f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], "
            f"{bases[0][3], bases[1][3]}, [{int(row['Ranking'])}]"
        )
        iso_tracker[col][pair_key] = iso_sum


def _record_partial_match(row, col, pep, annot_table,
                          example_list, example_t_list, max_ex):
    """Handle an FFC where only one fragment is identified."""
    if row["base_name_A"] is not None:
        known_base = row["base_name_A"]
        known_exp = row["explanation_A"]
        known_charge = row["charge_A"]
        known_dev = round(row["deviation_A"], 3)
        unknown_charge = row["charge_B"]
    else:
        known_base = row["base_name_B"]
        known_exp = row["explanation_B"]
        known_charge = row["charge_B"]
        known_dev = round(row["deviation_B"], 3)
        unknown_charge = row["charge_A"]

    complement = get_complementary_ion(pep, known_base)
    bases = [
        (known_base, known_exp, known_charge, known_dev),
        (complement, "???", unknown_charge, "n/a"),
    ]
    bases.sort()
    pair_key = f"{bases[0][0]}, {bases[1][0]}"

    annotation = (
        f"{bases[0][1]}, {bases[1][1]}, [{bases[0][2]}, {bases[1][2]}], "
        f"{bases[0][3], bases[1][3]}, [{int(row['Ranking'])}]"
    )

    if pair_key in annot_table[col] and annot_table[col][pair_key] is not None:
        annot_table[col][pair_key] = annotation

    if len(example_list) < max_ex:
        example_list.append(annotation + "|")
        example_t_list.append((
            round(row["m/z A"] * row["charge_A"], 4),
            round(row["m/z B"] * row["charge_B"], 4),
            round(row["m/z A"] * row["charge_A"] + row["m/z B"] * row["charge_B"], 4),
        ))


def _record_example(row, example_list, example_t_list):
    """Record diagnostic examples for fully-unmatched FFCs."""
    example_list.append((
        row["m/z A"], row["m/z B"],
        [row["charge_A"], row["charge_B"]],
        row["Ranking"],
    ))
    example_t_list.append((
        round(row["m/z A"] * row["charge_A"], 4),
        round(row["m/z B"] * row["charge_B"], 4),
        round(row["m/z A"] * row["charge_A"] + row["m/z B"] * row["charge_B"], 4),
    ))


def _compute_offset_coverage(
    columns, pep, length: int, iso: int, threshold: float,
    row_labels: List[str],
) -> dict:
    """
    Check whether any negative-loss column value matches a b/y ion mass
    (with isotopic shifts), marking those rows as '+*' (offset-covered).
    """
    # Build ion mass lookup with isotope shifts
    ion_masses = {}
    for i in range(0, length):
        ion_masses[f"b{i}"] = pep.ion_mass(f"b{i}")
    for i in range(1, length):
        ion_masses[f"y{i}"] = pep.ion_mass(f"y{i}")

    ion_masses_extended = dict(ion_masses)
    for n in range(1, iso + 1):
        for name, mass in ion_masses.items():
            ion_masses_extended[f"{name}-{n}"] = mass - n * ISOTOPE_OFFSET
            ion_masses_extended[f"{name}+{n}"] = mass + n * ISOTOPE_OFFSET

    # Which loss columns correspond to known ion masses?
    offset_matches = {}
    for col in columns:
        if col == "parent" or not isinstance(col, (int, float)) or col >= 0:
            continue
        for ion_name, ion_mass in ion_masses_extended.items():
            if abs(float(-col) - float(ion_mass)) < threshold:
                offset_matches[col] = ion_name
                break

    offset_cov = {r: "0" for r in row_labels}
    for col, matched_ion in offset_matches.items():
        for label in offset_cov:
            # label is e.g. "b3, y12" — check if matched_ion equals b3
            if matched_ion == label.split(",")[0].strip():
                offset_cov[label] = "+*"

    return offset_cov


def _combine_coverage(direct_col: pd.Series, offset_col: pd.Series) -> pd.Series:
    """Merge direct coverage ('+') with offset coverage ('+*')."""
    return pd.Series(
        np.select(
            [
                (direct_col == "+") | (offset_col == "+"),
                ((direct_col == "+*") | (offset_col == "+*"))
                & ~((direct_col == "+") | (offset_col == "+")),
                (direct_col == "0") & (offset_col == "0"),
            ],
            ["+", "+*", "0"],
            default=None,
        ),
        index=direct_col.index,
    )


def _append_summary_rows(
    table_columns, ffc_counts, fpc_both, fpc_one,
    ex_both, ex_both_t, ex_one, ex_one_t,
    columns, pep, length, iso, threshold, max_k,
) -> pd.DataFrame:
    """
    Build a DataFrame of summary/diagnostic rows (FFC, FPR, offset, examples)
    aligned to *table_columns*.  Caller should concat this onto the main table.
    """
    # Offset row
    ion_masses = {}
    for i in range(0, length):
        ion_masses[f"b{i}"] = pep.ion_mass(f"b{i}")
    for i in range(1, length):
        ion_masses[f"y{i}"] = pep.ion_mass(f"y{i}")
    ion_masses_ext = dict(ion_masses)
    for n in range(1, iso + 1):
        for name, mass in ion_masses.items():
            ion_masses_ext[f"{name}-{n}"] = mass - n * ISOTOPE_OFFSET
            ion_masses_ext[f"{name}+{n}"] = mass + n * ISOTOPE_OFFSET

    offset_matches = {}
    for col in columns:
        if col == "parent" or not isinstance(col, (int, float)) or col >= 0:
            continue
        for ion_name, ion_mass in ion_masses_ext.items():
            if abs(float(-col) - float(ion_mass)) < threshold:
                offset_matches[col] = ion_name
                break
        else:
            offset_matches[col] = None

    def _expand_examples(example_dict, prefix):
        rows = []
        for k in range(max_k):
            row_data = {
                col: (vals[k] if k < len(vals) else None)
                for col, vals in example_dict.items()
            }
            rows.append(row_data)
        return pd.DataFrame(rows, index=[f"{prefix}_{k+1}" for k in range(max_k)])

    parts = [
        pd.DataFrame([ffc_counts], index=["FFC"]),
        pd.DataFrame([fpc_both], index=["FPR0"]),
        pd.DataFrame([fpc_one], index=["FPR1"]),
        pd.DataFrame([offset_matches], index=["offset"]),
        _expand_examples(ex_both, "FPR0_LIST"),
        _expand_examples(ex_both_t, "TFPR0_LIST"),
        _expand_examples(ex_one, "FPR1_LIST"),
        _expand_examples(ex_one_t, "TFPR1_LIST"),
    ]

    # Concat all summary parts, then reindex to match the main table's columns
    summary = pd.concat(parts)
    return summary.reindex(columns=table_columns)


# =============================================================================
# 6. POST-PROCESSING: DUPLICATE FFC HANDLING
# =============================================================================

def process_ffc_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate FFCs that appear in multiple loss-line columns.

    Rules:
        - If an FFC appears in both a parental column and a loss column,
          remove it from the loss column.
        - If an FFC appears in multiple loss columns (but not parent),
          mark all occurrences as '(used)'.
    """
    df_clean = df.copy()

    # Identify parental columns (named 'parent' or positive integers)
    parental_cols = set()
    for col in df_clean.columns:
        s = str(col)
        if s == "parent" or (s.isdigit() and int(s) > 0):
            parental_cols.add(col)

    # Scan: map FFC-ID → list of (row, col) locations
    ffc_locations: dict[str, list] = {}
    ffc_id_re = re.compile(r"\[(\d+)\]")
    for row_idx in df_clean.index:
        for col in df_clean.columns:
            val = df_clean.at[row_idx, col]
            if pd.isna(val) or not isinstance(val, str):
                continue
            parts = val.split(",")
            match = ffc_id_re.search(parts[-1].strip()) if parts else None
            if match:
                ffc_locations.setdefault(match.group(1), []).append((row_idx, col))

    # Decide what to remove vs. mark
    cells_to_remove = set()
    cells_to_mark = set()

    for ffc_id, locs in ffc_locations.items():
        if len(locs) <= 1:
            continue
        in_parental = any(col in parental_cols for _, col in locs)
        if in_parental:
            for r, col in locs:
                if col not in parental_cols:
                    cells_to_remove.add((r, col))
        else:
            for r, col in locs:
                cells_to_mark.add((r, col))

    # Apply
    mod_counts = {col: 0 for col in df_clean.columns}
    for r, col in cells_to_remove:
        df_clean.at[r, col] = None
        mod_counts[col] += 1
    for r, col in cells_to_mark:
        val = df_clean.at[r, col]
        if isinstance(val, str) and "('used')" not in val:
            df_clean.at[r, col] = f"{val} ('used')"
        mod_counts[col] += 1

    summary = pd.DataFrame([mod_counts], index=["Modifications_Count"])
    return pd.concat([df_clean, summary])


# =============================================================================
# 7. ISOCOLUMN ANALYSIS
# =============================================================================

def isocolumns(
    df: pd.DataFrame,
    mass_list: List[float],
    base_is_parent: Optional[bool] = None,
    keep_multiple: bool = False,
) -> pd.DataFrame:
    """
    Collapse multiple isotopic-offset columns into a single 'isocolumn'
    by finding the minimum isotope-shift pair (db, dy) for each row.

    The (db, dy) pair represents how many isotope steps fragment-A and
    fragment-B are shifted from monoisotopic.  (0, 0) means a perfect
    monoisotopic match; (0, 1) means fragment-B is one isotope heavier, etc.
    """
    df = df.copy()
    if "parent" in df.columns:
        df.rename(columns={"parent": 0}, inplace=True)

    mass_list = sorted(mass_list)
    used_cols = [c for c in mass_list if c in df.columns]
    if not used_cols:
        raise ValueError("None of the requested mass_list columns are in df.")

    # Auto-detect whether first column is the parent line
    if base_is_parent is None:
        base_is_parent = (
            abs(float(used_cols[0])) < 1e-9
            and all(abs(float(c) - round(float(c))) < 1e-9 for c in used_cols)
        )

    all_columns = df[used_cols]

    # ── Regex helpers ─────────────────────────────────────────────────────
    _row_label_re = re.compile(r"b\s*(\d+)\s*,\s*y\s*(\d+)", re.IGNORECASE)

    _iso_token_pattern = (
        r"(?:\(\s*[+-]?\d+\s*\)(?:\s*-\s*\d+(?:\.\d+)?)?|[+-]?\d+)"
    )
    _annotation_re = re.compile(
        rf"({_iso_token_pattern})\s*,\s*({_iso_token_pattern})\s*,\s*\["
    )

    def _parse_iso_token(tok: str) -> Optional[int]:
        tok = tok.strip()
        for pattern in [r"^\(\s*([+-]?\d+)\s*\)", r"^([+-]?\d+)\b"]:
            m = re.match(pattern, tok)
            if m:
                return int(m.group(1))
        m = re.search(r"\(\s*([+-]?\d+)\s*\)", tok)
        return int(m.group(1)) if m else None

    def _extract_offset_pairs(cell) -> List[Tuple[int, int]]:
        """Extract all (db, dy) pairs from a cell annotation string."""
        if cell is None:
            return []
        s = str(cell).strip()
        if not s or s.lower() in ("none", "nan", ""):
            return []

        pairs = []
        seen = set()
        for m in _annotation_re.finditer(s):
            a = _parse_iso_token(m.group(1))
            b = _parse_iso_token(m.group(2))
            if a is not None and b is not None and (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b))
        return pairs

    def _mixed_minima(pairs: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Take the min of each component independently across all pairs."""
        if not pairs:
            return None
        return (min(p[0] for p in pairs), min(p[1] for p in pairs))

    def _prune(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Keep only the non-dominated pair(s) by sum, optionally just the best."""
        if not pairs:
            return []
        pairs = list(dict.fromkeys(pairs))
        key = lambda t: (t[0] + t[1], t[0], t[1])
        kept = []
        for p in sorted(pairs, key=key):
            if not any((q[0] + q[1]) <= (p[0] + p[1]) for q in kept):
                kept.append(p)
        return kept if keep_multiple else [min(kept, key=key)]

    # ── Build isocolumn ───────────────────────────────────────────────────
    iso_values = []
    for idx, row in all_columns.iterrows():
        # Skip summary rows
        if not _row_label_re.search(str(idx)):
            iso_values.append(None)
            continue

        isoline: List[Tuple[int, int]] = []
        for j, col in enumerate(used_cols):
            cell = row[col]
            if pd.isna(cell) or str(cell).strip().lower() in ("", "none", "nan"):
                continue

            if j == 0 and base_is_parent:
                isoline = [(0, 0)]
                break

            offset_pairs = _extract_offset_pairs(cell)
            if not offset_pairs:
                continue

            pair = _mixed_minima(offset_pairs)
            if pair == (0, 0):
                isoline = [(0, 0)]
                break

            # Add with dominance check
            s_new = pair[0] + pair[1]
            if not any((a + b) <= s_new for a, b in isoline):
                isoline.append(pair)

        pruned = _prune(isoline)
        iso_values.append(str(pruned[0]) if pruned else None)

    return pd.DataFrame(
        {f"isocolumn:{min(mass_list)}": iso_values},
        index=df.index,
    )


# =============================================================================
# 8. UTILITY FUNCTIONS
# =============================================================================

def group_consecutive_floats(
    nums: List[float], tolerance: float = 0.1,
) -> list:
    """
    Group a sorted list of floats into runs of consecutive integers
    (where 'consecutive' means difference ≈ 1.0 within *tolerance*).

    Returns a mixed list: sub-lists for groups of 2+, bare values otherwise.
    Example: [0, 1, 2, 5, 8, 9] → [[0, 1, 2], 5, [8, 9]]
    """
    if not nums:
        return []

    groups: list = []
    current = [nums[0]]

    for i in range(1, len(nums)):
        if abs((nums[i] - current[-1]) - 1.0) <= tolerance:
            current.append(nums[i])
        else:
            groups.append(current if len(current) > 1 else current[0])
            current = [nums[i]]

    groups.append(current if len(current) > 1 else current[0])
    return groups


def prioritize_zero(mixed_list: list) -> list:
    """
    Reorder so that items containing 0 come first.
    Works for both bare values and sub-lists.
    """
    with_zero, without = [], []
    for item in mixed_list:
        if (isinstance(item, list) and 0 in item) or (not isinstance(item, list) and item == 0):
            with_zero.append(item)
        else:
            without.append(item)
    return with_zero + without


def cov_notation(df: pd.DataFrame) -> str:
    """
    Produce a compact binary string from the second column of the coverage
    table: '+' → '1', '0' → '0', other → skip.  Appends total cut count.
    """
    col = df.iloc[:, 1]
    result = ""
    cuts = 0
    for val in col:
        if val == "+":
            result += "1"
            cuts += 1
        elif val == "0" or val == 0:
            result += "0"
        # Skip non-coverage rows (summary, etc.)
    # Trim trailing summary characters and append count
    return result[:-2] + f"({cuts})" if len(result) >= 2 else result + f"({cuts})"


def rename_loss_columns(
    df: pd.DataFrame,
    annotations: dict,
) -> pd.DataFrame:
    """
    Rename loss-line columns by appending parenthesised annotation info.

    Parameters
    ----------
    df : DataFrame
        The final output table whose columns include loss-line keys
        (e.g. ``'parent'``, ``1``, ``2``, ``-347.163``, …).
    annotations : dict
        Mapping from *existing* column name → annotation string.
        Example: ``{'parent': '0.02', 1: '0.05', 2: '0.03'}``

    Returns
    -------
    DataFrame with matching columns renamed, e.g.
        ``'parent'`` → ``'parent(0.02)'``,  ``1`` → ``'1(0.05)'``

    Columns not present in *annotations* (or not in *df*) are left unchanged.
    """
    rename_map = {}
    for col, note in annotations.items():
        if col in df.columns:
            rename_map[col] = f"{col}({note})"
    return df.rename(columns=rename_map)


def add_overlap_stats(
    df: pd.DataFrame, parent_col: str = "parent",
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute overlap and difference counts between each column and the
    *parent_col* column.

    Returns (sim_row, diff_row) — how many rows each column shares with
    vs. differs from the parent column.
    """
    has_content = df.notna()
    parent_mask = has_content[parent_col]

    overlap = (has_content & parent_mask.values[:, None]).sum()
    non_overlap = (has_content & ~parent_mask.values[:, None]).sum()

    stats = pd.DataFrame(
        [overlap, non_overlap],
        index=["|SIM(A*,B)|", "|DIFF(A*,B)|"],
    )
    return stats.loc["|SIM(A*,B)|"], stats.loc["|DIFF(A*,B)|"]


# =============================================================================
# 9. MAIN — EXAMPLE PIPELINE
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    # ── Project imports ───────────────────────────────────────────────────
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir.parent))
    import peptide
    import interpreter_modify  # noqa: F401

    # ── Configuration ─────────────────────────────────────────────────────
    #PEP_SEQ = "YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY"
    #PEP_SEQ = "VEADIAGHGQEVLIR"
    #PEP_SEQ = "YLEFISDAIIHVLHSK"
    #PEP_SEQ = "HGTVVLTALGGILK"
    #PEP_SEQ = "HADGSFSDEMNTILDNLAARDFINWLIQTKITD"
    #PEP_SEQ = "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK" # 667.90419 * 6 NH2
    PEP_SEQ = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES" # 749.43703 * 6
    #PEP_SEQ = "YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY" # 712.52139 * 6 NH2
    
    
    
    CHARGE = 6
    ISO_RANGE = 6
    TOP_N = 2000
    MASS_THRESHOLD = 0.5
    #LOSS_LIST = [-1, -2, -3, 0, 229.112, 228.109, 100.069, 99.069, 1]
    #LOSS_LIST = [-1, -2, -3, -4, 0, 346.151, 347.163, 345.133]
    #LOSS_LIST = [-1, -2, -3, 0, 229.112, 228.109, 99.065, 100.069]
    #LOSS_LIST = [-1, -2, 0, 276.1440, 277.1460, 113.0785, 26.9880]
    #LOSS_LIST = [-1, -2, 0, 113.083, 112.080, 18.006, 114.086]
    #LOSS_LIST = [-1, -2, 0, -3, -4, 345.1410, 346.1430, 344.1370, 14.9770, 13.9760, 347.1470]
    #LOSS_LIST = [-1, -2, 0, -3, 223.1455, 224.1505, 222.1420, 110.0600, 241.1700, 242.1580]
    #LOSS_LIST = [-1, -2, 0, -3, -4,15.0080,16.0005, 914.5350, 913.5320]
    LOSS_LIST = [-1, -2, 0, -3, -4, 161.0390, 14.9970, 16.0020, 162.0400, 259.0970, 260.1000]
    LOSS_LIST = [-1, -2, 0, -3]
    
    COLUMN_ANNOTATIONS = {
    "parent": "0.002388",
    1: "0.001692",
    2: "0.001986",
    3: "0.001452",
    4: "0.001570",
    5: "0.00",
    6: "0.00",
    -229.112: "0.0017",
    -228.109:"0.0018",
    -99.065: "0.000173",
    -100.069:"0.000199", 
    -113.0785:"0.000998",
    -276.1440:"0.002023",
    -277.1460:"0.004119",
    -26.9880:"0.007049",
    -113.083:"0.001148",
    -112.080:"0.001862",
    -18.006:"0.001105",
    -114.086:"0.001035",
    -345.1410:"0.001458",
    -346.1430:"0.001967",
    -344.1370:"0.004634",
    -347.1470:"0.003197",
    -14.9770:"0.002888",
    -13.9760:"0.006366",
    -223.1455:"0.007473",
    -224.1505:"0.007756",
    -222.1420:"0.002582",
    -110.0600:"0.002019",
    -241.1700:"0.004770",
    -242.1580:"0.001349",
    -15.0080:"0.001282",
    -16.0005: "0.002647",
    -914.5350:"0.000961",
    -913.5320:"0.001052",
    -161.0390:"0.001340",
    -14.9970:"0.006422",
    -16.0020:"0.000900",
    -162.0400:"0.000454",
    -259.0970:"0.002471",
    -260.1000:"0.001784"
    }
    

    OUTPUT_CSV_DETAIL = "protein_result.csv"
    OUTPUT_CSV_COV = "protein.csv"
    #OUTPUT_EXCEL = "result/all.xlsx"
    OUTPUT_EXCEL = "result/jamesv2.xlsx"
    #OUTPUT_EXCEL = "result/James_V1.xlsx"
    #OUTPUT_SHEET = f"VEA3+_{TOP_N}"
    OUTPUT_SHEET = "LLG6+_test4"
    #OUTPUT_SHEET = "Neuropeptide_Sum_Top10000"

    # ── Build peptide ─────────────────────────────────────────────────────
    #pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20="NH3")
    pep = peptide.Pep(f"[{PEP_SEQ}+{CHARGE}H]{CHARGE}+", end_h20=True)
    print(f"Precursor mass: {pep.pep_mass}")

    # ── Load FFC data ─────────────────────────────────────────────────────
    
    '''
    data_path = (
        os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "data/long_peptide/"
                     "Covariances_Neuropeptide_ChargeStates/"
                     "Covariance_Data.Neuropeptide_z7_611_dm2_NCE25")
    )
    '''
    
    data_path = os.path.join(os.environ.get("DENOVOFFC_DATA_ROOT", "data"), "deiso/python_code/result/LLG_merged.tsv")
    
    df = pd.read_csv(data_path, sep=r"\s+", skiprows=1, header=None, engine="python")
    
    
    #df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking"]
    #df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", 'iso']
    #df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", 'intensity A', 'intensity B']
    #df.columns = ["m/z A", "m/z B","Score", "Ranking"]
    #df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", 'intensity A', 'intensity B', 'line ID']
    #df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", "intensity A", "intensity B", "charge_A", "charge_B", "deconv_method"]
    df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", "intensity A", "intensity B", "mz_A_original", "mz_B_original", "charge_A", "charge_B", "deconv_method"]
    
    print(df.head())
    num_ffcs_total = len(df)

    # ── Filter & sort ─────────────────────────────────────────────────────
    df = df[df["Score"] > 0].copy()
    df["Ranking"] = df["Ranking"].fillna(-1).astype(int)
    df = df.sort_values("Ranking").query("Ranking != -1")
    df = df[["m/z A", "m/z B", "Ranking"]].head(TOP_N)
    df = df[df['Ranking'] <= TOP_N]

    print("Top rows:")
    print(df.shape)
    print(df.head())

    # ── Partition by charge ───────────────────────────────────────────────
    partitioned, partition_names = partition_dataframe_by_charge(df, [CHARGE, CHARGE - 1])
    partitioned.to_csv('partitioned.csv')
    print(partitioned.shape)

    # ── Annotate each loss line ───────────────────────────────────────────
    annotated_frames = []
    for loss in LOSS_LIST:
        matched = select_best_partition(
            partitioned, ["m/z A", "m/z B", "Ranking"],
            pep.pep_mass - loss, MASS_THRESHOLD, partition_names, iso_range=0,
        )
        num_ffc = len(matched)
        annotated = annotate_dataframe_loss(
            matched, pep, MASS_THRESHOLD, diffs=[loss],
            charge=CHARGE, iso_range=ISO_RANGE,
        )
        annotated["line"] = "parent" if loss == 0 else -loss
        annotated["num_ffc"] = num_ffc
        annotated_frames.append(annotated)

    df_all = (
        pd.concat(annotated_frames, ignore_index=True)
        .sort_values("Ranking")
    )
    print(df_all)
    df_all.to_csv(OUTPUT_CSV_DETAIL)

    # ── Build coverage table ──────────────────────────────────────────────
    cov_table = coverage_table(df_all, LOSS_LIST, pep, ISO_RANGE)
    cov_table = process_ffc_annotations(cov_table)
    print(cov_table)
    #cov_table.to_csv(OUTPUT_CSV_COV)

    # ── Assemble final output with isocolumns ─────────────────────────────
    numeric_cols = [c for c in cov_table.columns if not isinstance(c, str)]
    numeric_cols.append(0)
    numeric_cols.sort()
    grouped = prioritize_zero(group_consecutive_floats(numeric_cols))

    # Start with the first two columns (AA + coverage)
    df_list = [cov_table[cov_table.columns[0]], cov_table[cov_table.columns[1]]]

    for i, group in enumerate(grouped):
        if isinstance(group, list):
            df_list.append(isocolumns(cov_table, group))
            display_names = ["parent" if x == 0 else x for x in group]
            df_list.append(cov_table[display_names])
        else:
            # Remaining ungrouped columns
            df_list.append(cov_table[grouped[i:]])
            break

    final_df = pd.concat(df_list, axis=1)
    # Append last two columns from cov_table (typically Row Count & theoretical mass)
    final_df = final_df.join(cov_table[cov_table.columns[-2]])
    final_df = final_df.join(cov_table[cov_table.columns[-1]])

    # ── Overlap statistics ────────────────────────────────────────────────
    loss_col_names = [-v if v != 0 else "parent" for v in LOSS_LIST]
    sim, diff = add_overlap_stats(
        final_df.iloc[: pep.pep_len - 1][loss_col_names]
    )
    final_df = pd.concat([final_df, sim.to_frame().T, diff.to_frame().T])

    # ── Rename loss-line columns with annotations ──────────────────────────
    # Provide a dict mapping original column names → parenthesised info.
    # Example: threshold, p-value, or any parameter you want displayed.
    


    final_df = rename_loss_columns(final_df, COLUMN_ANNOTATIONS)

    print(final_df)
    print(final_df['parent(0.002388)'])

    # ── Write to Excel ────────────────────────────────────────────────────
    with pd.ExcelWriter(
        OUTPUT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace",
    ) as writer:
        final_df.to_excel(writer, sheet_name=OUTPUT_SHEET, index_label=f"N={TOP_N}")

    print(cov_notation(final_df))
    print(f"Total FFCs in raw data: {num_ffcs_total}")
    print(pep.ion_mass('y9'))