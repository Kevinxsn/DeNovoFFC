"""
Mass spectrum deconvolution script for paired m/z columns.

Method (per Prof. Xiaowen Liu):
1. Group peaks into isotopic envelopes by searching for consistent
   spacing of 1/z (z = 1..4).
2. Determine charge state z from the spacing.
3. Deconvolute: M_neutral = z * (m/z - 1.00728).
4. Determine the monoisotopic m/z using a candidate-based approach:
   for each candidate monoisotopic m/z (each observed peak in the
   envelope, plus a few "virtual" peaks extrapolated below the lowest
   observed peak to handle missing-monoisotopic cases), look up the
   closest theoretical envelope from TopPIC's theo_patt.txt, align
   the theoretical intensity distribution to the observed peaks, and
   compute cosine similarity. Select the candidate with the highest
   similarity.
5. Report deconvoluted m/z = (M_neutral + z*proton) / z.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Parse the TopPIC theoretical-pattern file
# ---------------------------------------------------------------------------

def parse_theo_patt(filepath: str) -> list[dict]:
    envelopes = []
    current_peaks = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("formula:") or line.startswith("#"):
                if current_peaks:
                    intensities = [p[1] for p in current_peaks]
                    envelopes.append({
                        "mono_mass": current_peaks[0][0],
                        "peaks": current_peaks,
                        "max_peak_index": int(np.argmax(intensities)),
                    })
                current_peaks = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    current_peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

    if current_peaks:
        intensities = [p[1] for p in current_peaks]
        envelopes.append({
            "mono_mass": current_peaks[0][0],
            "peaks": current_peaks,
            "max_peak_index": int(np.argmax(intensities)),
        })
    return envelopes


def build_lookup_table(envelopes):
    envelopes_sorted = sorted(envelopes, key=lambda e: e["mono_mass"])
    mono_masses = np.array([e["mono_mass"] for e in envelopes_sorted])
    return mono_masses, envelopes_sorted


def find_closest_envelope(mono_masses, envelopes_sorted, target_mass):
    idx = np.searchsorted(mono_masses, target_mass)
    candidates = [i for i in [idx - 1, idx] if 0 <= i < len(mono_masses)]
    best = min(candidates, key=lambda i: abs(mono_masses[i] - target_mass))
    return envelopes_sorted[best]


# ---------------------------------------------------------------------------
# 2. Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm else 0.0


# ---------------------------------------------------------------------------
# 3. Core deconvolution logic
# ---------------------------------------------------------------------------

PROTON_MASS = 1.00728
ISOTOPE_SPACING = 1.003355  # neutron mass defect (C12 -> C13)


def _find_isotopic_envelopes(mz_vals, intensities, mz_tol=0.02):
    """
    Group peaks into isotopic envelopes by spacing.

    For each unassigned peak, try z = 4, 3, 2, 1.
    Accept charge that explains the most peaks (ties -> higher z).
    """
    n = len(mz_vals)
    order = np.argsort(mz_vals)
    mz_s = mz_vals[order]
    int_s = intensities[order]
    assigned = [False] * n
    envelopes = []

    for i in range(n):
        if assigned[i]:
            continue

        best_env = None
        best_z = 0
        best_count = 0

        for z in [4, 3, 2, 1]:
            spacing = 1.0 / z
            members = [i]
            last_mz = mz_s[i]

            j = i + 1
            while j < n:
                if assigned[j]:
                    j += 1
                    continue
                diff = mz_s[j] - (last_mz + spacing)
                if diff < -mz_tol:
                    j += 1
                    continue
                if diff > mz_tol:
                    break
                members.append(j)
                last_mz = mz_s[j]
                j += 1

            if len(members) >= 2:
                if (len(members) > best_count or
                        (len(members) == best_count and z > best_z)):
                    best_env = list(members)
                    best_z = z
                    best_count = len(members)

        if best_env is not None:
            for idx in best_env:
                assigned[idx] = True
            envelopes.append({
                "sorted_indices": best_env,
                "original_indices": order[best_env].tolist(),
                "mz_values": mz_s[best_env].tolist(),
                "intensities": int_s[best_env].tolist(),
                "charge": best_z,
            })
        else:
            assigned[i] = True
            envelopes.append({
                "sorted_indices": [i],
                "original_indices": [order[i]],
                "mz_values": [mz_s[i]],
                "intensities": [int_s[i]],
                "charge": 1,
            })

    return envelopes


def determine_monoisotopic_mz(envelope,
                               mono_masses_lookup=None,
                               envelopes_lookup=None):
    """
    Determine the monoisotopic m/z of an envelope using Prof. Liu's
    most-intense-anchor + lookup-table approach.

    Logic (mirrors Prof. Liu's 5000 Da example word-for-word):
      1. Find the most intense observed peak -> compute its neutral mass
         assuming it IS the monoisotopic peak.
      2. Look up the theoretical envelope in theo_patt.txt whose mono
         mass is closest to that.
      3. Read `max_peak_index` from the matched theoretical envelope:
         this tells us how many isotope steps the theoretical max peak
         sits above the theoretical monoisotopic peak.
            - max_peak_index = 0  =>  for molecules in this mass range,
              the most intense peak IS the monoisotopic peak, so no
              shift needed. The observed max peak is the monoisotopic.
            - max_peak_index = k  =>  the observed max peak is the k-th
              isotope; the monoisotopic sits k isotope steps below it,
              possibly at an m/z not present in the observed data.
      4. Cosine similarity is used as a confidence score against the
         matched theoretical envelope but does NOT override the shift
         determined by `max_peak_index`. The shape of the theoretical
         envelope has already told us where the monoisotopic is; the
         cosine score just tells us how well the observed envelope
         resembles the theoretical one.

    Examples from Prof. Liu's emails:
      - 300.156/301.160, z=1: mass_of_max=299.15 -> matches
        C13H21N4O4 envelope (mono=297.156, max_peak_index=0) ->
        no shift -> monoisotopic m/z = 300.156. Correct.
      - 5000 Da example: mass_of_max~5008 -> matches
        C223H353N61O66S2 envelope (mono=5005.56, max_peak_index=3) ->
        shift 3 steps down -> monoisotopic mass ~4997. Correct.

    Why cosine similarity alone cannot drive shift selection:
      Cosine similarity is scale-invariant, so with very few observed
      peaks (especially 2-3) the ratio of theoretical tail peaks can
      coincidentally match the observed ratio better than the true
      monoisotopic region. We therefore trust the lookup table's
      max_peak_index directly, which is self-consistent across the
      full mass range of the table.

    Returns
    -------
    mono_mz   : float  monoisotopic m/z
    mono_mass : float  neutral monoisotopic mass
    charge    : int
    sim       : float  cosine similarity between observed envelope
                       and matched theoretical envelope (confidence)
    """
    z = envelope["charge"]
    mz_arr = np.array(envelope["mz_values"], dtype=float)
    int_arr = np.array(envelope["intensities"], dtype=float)
    spacing_mz = ISOTOPE_SPACING / z

    # Find the most intense observed peak -- this is our anchor.
    max_idx = int(np.argmax(int_arr))
    max_mz = float(mz_arr[max_idx])
    mass_of_max = z * (max_mz - PROTON_MASS)

    # Fallback when we have no lookup table or a single-peak envelope:
    # assume the lowest observed m/z is the monoisotopic peak.
    if (mono_masses_lookup is None or envelopes_lookup is None
            or len(mz_arr) < 2):
        mono_mz = float(mz_arr[0])
        mono_mass = z * (mono_mz - PROTON_MASS)
        return mono_mz, mono_mass, z, float("nan")

    # Look up the theoretical envelope closest to the most-intense-peak
    # mass. This gives us the max_peak_index we need.
    ref_env = find_closest_envelope(
        mono_masses_lookup, envelopes_lookup, mass_of_max
    )
    ref_int_full = np.array([p[1] for p in ref_env["peaks"]], dtype=float)
    if ref_int_full.size == 0 or ref_int_full.max() <= 0:
        return max_mz, mass_of_max, z, float("nan")
    ref_int_full = ref_int_full / ref_int_full.max() * 100.0
    ref_max_idx = ref_env["max_peak_index"]

    # The observed max peak sits at theoretical position ref_max_idx.
    # The monoisotopic m/z is ref_max_idx isotope steps below it.
    mono_mz = max_mz - ref_max_idx * spacing_mz
    mono_mass = z * (mono_mz - PROTON_MASS)

    # Compute cosine similarity as a confidence score: align each
    # observed peak to its theoretical index under the chosen monoisotopic,
    # then compare the intensity shape.
    obs_aligned = []
    theo_aligned = []
    for obs_mz, obs_int in zip(mz_arr, int_arr):
        delta_steps = (obs_mz - mono_mz) / spacing_mz
        theo_idx = int(round(delta_steps))
        if abs(delta_steps - theo_idx) > 0.3:
            continue
        if 0 <= theo_idx < len(ref_int_full):
            obs_aligned.append(obs_int)
            theo_aligned.append(ref_int_full[theo_idx])

    if len(obs_aligned) >= 1:
        obs_vec = np.array(obs_aligned, dtype=float)
        if obs_vec.max() > 0:
            obs_vec = obs_vec / obs_vec.max() * 100.0
        sim = cosine_similarity(obs_vec, np.array(theo_aligned))
    else:
        sim = float("nan")

    return float(mono_mz), float(mono_mass), z, float(sim)


# ---------------------------------------------------------------------------
# 4. Internal helpers
# ---------------------------------------------------------------------------

def _load_lookup(theo_patt_path):
    if theo_patt_path and Path(theo_patt_path).exists():
        raw_envs = parse_theo_patt(theo_patt_path)
        mono_lookup, env_lookup = build_lookup_table(raw_envs)
        print(f"Loaded {len(env_lookup)} theoretical envelopes "
              f"from {theo_patt_path}")
        return mono_lookup, env_lookup
    elif theo_patt_path:
        print(f"Warning: {theo_patt_path} not found. "
              "Lowest observed peak assumed to be monoisotopic.")
    return None, None


def _deconvolute_single_column(mz_vals, intensities,
                                mono_lookup, env_lookup, mz_tol=0.02):
    n = len(mz_vals)
    charges = np.zeros(n, dtype=int)
    envelope_ids = np.full(n, -1, dtype=int)
    mono_masses = np.full(n, np.nan)
    deconv_mz = np.full(n, np.nan)
    similarities = np.full(n, np.nan)

    envelopes = _find_isotopic_envelopes(mz_vals, intensities, mz_tol)

    for env_id, env in enumerate(envelopes):
        mono_mz, mono_mass, z, sim = determine_monoisotopic_mz(
            env, mono_lookup, env_lookup
        )
        for orig_idx in env["original_indices"]:
            charges[orig_idx] = z
            envelope_ids[orig_idx] = env_id
            mono_masses[orig_idx] = round(mono_mass, 4)
            deconv_mz[orig_idx] = round(mono_mz, 4)
            similarities[orig_idx] = round(sim, 4) if not np.isnan(sim) else np.nan

    return charges, envelope_ids, mono_masses, deconv_mz, similarities


# ---------------------------------------------------------------------------
# 5. Main public function
# ---------------------------------------------------------------------------

def deconvolute_pairs(df: pd.DataFrame,
                      mz_col_a: str = "m/z A",
                      mz_col_b: str = "m/z B",
                      intensity_col_a: str | None = None,
                      intensity_col_b: str | None = None,
                      theo_patt_path: str | None = None,
                      mz_tol: float = 0.01):
    """
    Deconvolute two m/z columns (A and B) in a DataFrame.

    Returns
    -------
    df_annotated : original columns + for each of A and B:
        charge_{A/B}, envelope_id_{A/B},
        monoisotopic_mass_{A/B}, deconvoluted_mz_{A/B},
        isotope_similarity_{A/B}
    df_replaced : same shape & columns as `df`, but mz_col_a and mz_col_b
        values are replaced by the deconvoluted monoisotopic m/z.
    """
    df_ann = df.copy()
    mono_lookup, env_lookup = _load_lookup(theo_patt_path)

    # ---- Column A ----
    mz_a = df[mz_col_a].values.astype(float)
    int_a = (df[intensity_col_a].values.astype(float)
             if intensity_col_a and intensity_col_a in df.columns
             else np.ones(len(mz_a)))
    if intensity_col_a and intensity_col_a not in df.columns:
        print(f"Warning: intensity column '{intensity_col_a}' not found; "
              "using uniform intensities for column A.")

    ch_a, env_a, mass_a, dmz_a, sim_a = _deconvolute_single_column(
        mz_a, int_a, mono_lookup, env_lookup, mz_tol
    )
    df_ann["charge_A"] = ch_a
    df_ann["envelope_id_A"] = env_a
    df_ann["monoisotopic_mass_A"] = mass_a
    df_ann["deconvoluted_mz_A"] = dmz_a
    df_ann["isotope_similarity_A"] = sim_a

    # ---- Column B ----
    mz_b = df[mz_col_b].values.astype(float)
    int_b = (df[intensity_col_b].values.astype(float)
             if intensity_col_b and intensity_col_b in df.columns
             else np.ones(len(mz_b)))
    if intensity_col_b and intensity_col_b not in df.columns:
        print(f"Warning: intensity column '{intensity_col_b}' not found; "
              "using uniform intensities for column B.")

    ch_b, env_b, mass_b, dmz_b, sim_b = _deconvolute_single_column(
        mz_b, int_b, mono_lookup, env_lookup, mz_tol
    )
    df_ann["charge_B"] = ch_b
    df_ann["envelope_id_B"] = env_b
    df_ann["monoisotopic_mass_B"] = mass_b
    df_ann["deconvoluted_mz_B"] = dmz_b
    df_ann["isotope_similarity_B"] = sim_b

    # ---- Replaced DataFrame ----
    df_rep = df.copy()
    df_rep[mz_col_a] = dmz_a
    df_rep[mz_col_b] = dmz_b

    return df_ann, df_rep




# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/HAD4+_with_intensity"
    save_path = "/Users/kevinmbp/Desktop/2D_spec_dict/data/long_peptide/deconv/"

    #df = pd.read_csv(data_path, sep=r"\s+", skiprows=1, header=None, engine="python")
    df = pd.read_csv(data_path, sep="\t", skiprows=1, header=None, engine="python")
    df.columns = ["m/z A", "m/z B", "Covariance", "Partial Cov.", "Score", "Ranking", "intensity A", "intensity B"]
    df = df[df["Ranking"] != -1]
    df = df.sort_values("Ranking")
    df = df.head(2000)

    # NOTE: your VEA3+.txt file does not contain per-peak intensity columns.
    # Covariance / Partial Cov. are per-PAIR statistics, not per-peak
    # intensities, so they cannot be used directly as intensities here.
    # If you have access to the underlying 1D spectrum intensities, load
    # them and map each m/z to its intensity before calling this function.
    # Otherwise, pass intensity_col_a=None / intensity_col_b=None.
    df_annotated, df_replaced = deconvolute_pairs(
        df,
        mz_col_a="m/z A",
        mz_col_b="m/z B",
        intensity_col_a="intensity A",
        intensity_col_b="intensity B",
        theo_patt_path="theo_patt.txt",
    )

    df_annotated.to_csv(save_path + "HAD4+intensity_annotated.txt", sep="\t", index=False)
    df_replaced.to_csv(save_path + "HAD4+intensity_replaced.txt", sep="\t", index=False)