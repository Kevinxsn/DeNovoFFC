"""
Infer Isotope Number from Original m/z and Deconvoluted m/z
============================================================

The deconvoluted m/z ("m/z A", "m/z B") and original m/z
("mz_A_original", "mz_B_original") are both in m/z space.

The isotope offset in m/z space is:  delta = k * 1.00235 / z

So:  k = (mz_original - mz_deconvoluted) * z / 1.00235

When charge is NaN, we scan z = 1..max_z for the best integer k.
"""

import pandas as pd
import numpy as np

ISOTOPE_SPACING = 1.00235


def infer_k_known_charge(mz_original, mz_deconvoluted, charge):
    """Infer isotope number k when charge state z is known."""
    if pd.isna(mz_original) or pd.isna(mz_deconvoluted) or pd.isna(charge):
        return np.nan
    z = int(charge)
    delta = mz_original - mz_deconvoluted
    k_exact = delta * z / ISOTOPE_SPACING
    k_rounded = round(k_exact)
    return k_rounded if k_rounded >= 0 else np.nan


def infer_k_unknown_charge(mz_original, mz_deconvoluted, max_charge=60, tol=0.05):
    """
    Infer isotope number k by scanning charge states when z is unknown.
    Returns (best_k, best_z).
    """
    if pd.isna(mz_original) or pd.isna(mz_deconvoluted):
        return np.nan, np.nan

    delta = mz_original - mz_deconvoluted
    best_k, best_z, best_dev = np.nan, np.nan, 1e9

    for z in range(1, max_charge + 1):
        k_exact = delta * z / ISOTOPE_SPACING
        k_rounded = round(k_exact)
        k_dev = abs(k_exact - k_rounded)

        if k_rounded >= 0 and k_dev < best_dev:
            best_k = k_rounded
            best_z = z
            best_dev = k_dev

    return best_k, best_z


def add_isotope_columns(df, max_charge=60):
    """
    Add 'isotope_k_A' and 'isotope_k_B' columns to the dataframe.

    Expected columns:
        - mz_A_original, mz_B_original  (observed m/z from spectrum)
        - m/z A, m/z B                  (deconvoluted m/z, monoisotopic)
        - charge_A, charge_B            (charge states, may contain NaN)
    """
    df = df.copy()

    iso_a, iso_b = [], []
    charge_a_used, charge_b_used = [], []

    for _, row in df.iterrows():
        # --- Side A ---
        if pd.notna(row.get("charge_A")):
            k_a = infer_k_known_charge(row["mz_A_original"], row["m/z A"], row["charge_A"])
            iso_a.append(k_a)
            charge_a_used.append(int(row["charge_A"]))
        else:
            k_a, z_a = infer_k_unknown_charge(row["mz_A_original"], row["m/z A"], max_charge)
            iso_a.append(k_a)
            charge_a_used.append(z_a)

        # --- Side B ---
        if pd.notna(row.get("charge_B")):
            k_b = infer_k_known_charge(row["mz_B_original"], row["m/z B"], row["charge_B"])
            iso_b.append(k_b)
            charge_b_used.append(int(row["charge_B"]))
        else:
            k_b, z_b = infer_k_unknown_charge(row["mz_B_original"], row["m/z B"], max_charge)
            iso_b.append(k_b)
            charge_b_used.append(z_b)

    df["isotope_k_A"] = iso_a
    df["isotope_k_B"] = iso_b
    df["charge_A_used"] = charge_a_used
    df["charge_B_used"] = charge_b_used

    return df


# ─────────────────────────────────────────────
#  Demo with data from your screenshot
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data = {
        "m/z A":          [186.088018, 186.088018, 280.182530, 280.182530, 379.730680],
        "m/z B":          [630.395586, 798.239231, 861.767405, 861.767405, 811.993130],
        "mz_A_original":  [186.088018, 186.088018, 280.182538, 280.182538, 379.730691],
        "mz_B_original":  [630.395586, 798.239231, 862.268876, 862.018239, 812.243980],
        "charge_A":       [float("nan"), float("nan"), 2.0, 2.0, 2.0],
        "charge_B":       [1.0, float("nan"), 4.0, 4.0, 4.0],
    }

    df = pd.DataFrame(data)

    print("── Input DataFrame ──")
    print(df.to_string(index=True))
    print()

    result = add_isotope_columns(df)

    print("── Result with inferred isotope numbers ──")
    display_cols = [
        "mz_A_original", "m/z A", "charge_A_used", "isotope_k_A",
        "mz_B_original", "m/z B", "charge_B_used", "isotope_k_B",
    ]
    print(result[display_cols].to_string(index=True))
    print()

    # Show detail for each row
    print("── Row-by-row detail ──")
    for i, row in result.iterrows():
        delta_a = row["mz_A_original"] - row["m/z A"]
        delta_b = row["mz_B_original"] - row["m/z B"]
        print(f"Row {i}:")
        print(f"  A: mz_orig={row['mz_A_original']:.6f}, mz_deconv={row['m/z A']:.6f}, "
              f"delta={delta_a:.6f}, z={row['charge_A_used']}, k={row['isotope_k_A']}")
        print(f"  B: mz_orig={row['mz_B_original']:.6f}, mz_deconv={row['m/z B']:.6f}, "
              f"delta={delta_b:.6f}, z={row['charge_B_used']}, k={row['isotope_k_B']}")
        print()