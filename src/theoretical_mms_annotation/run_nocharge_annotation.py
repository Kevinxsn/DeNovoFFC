"""Charge-free FFC annotation, one definitive FFC per rank.

Companion to run_annotation.py, for raw MMS tables that carry no
deconvolution output at all -- i.e. no 'charge_A'/'charge_B' columns
(e.g. $DENOVOFFC_DATA_ROOT/mms_with_intensity/VEA_15_Z3_revised). Charge is
never read from the input and never constrains matching; every charge
reported comes from the theoretical ion that was matched.

Difference from ffc_annotator.annotate_ffcs(): when no compatible FFC pair
exists jointly, annotate_ffcs() falls back to listing *every* candidate for
that side (charge is the only way to disambiguate and there is none here),
so one rank can carry many candidate interpretations. Here the fallback
keeps only the single best candidate per side -- the same
(priority, |mass error|) choice ffc_annotator.match_mz() makes -- so every
rank gets exactly one FFC. Because both sides are then always a definitive
ion, the full extra-info column set (annotation / breaking point / TFFC /
offset / deviation) is available, which annotate_ffcs() cannot produce.

Fallback labels are prefixed '*' (the module's "charge unconfirmed" marker):
in this charge-free mode a fallback side was never verified as part of a
compatible pair, so its charge rests entirely on the single-m/z match.

Usage:
    PY=python
    $PY -c "from run_nocharge_annotation import run_nocharge_annotation; \
      run_nocharge_annotation( \
        '$DENOVOFFC_DATA_ROOT/mms_with_intensity/VEA_15_Z3_revised', \
        'VEADIAGHGQEVLIR', 3, top_n=50)"
"""

from pathlib import Path

import pandas as pd

from ffc_annotator import (
    DEFAULT_ISOMAX_CUTOFF_PCT,
    DEFAULT_TOL_PPM,
    _dedup_breakpoint,
    _enforce_by_placement,
    _write_xlsx,
    build_mz_index,
    candidates_for,
    classify_ffc_relationship,
    compute_peptide_mass,
    generate_theoretical_ions,
    match_ffc_pair_with_charge,
    raw_breakpoints,
    tffc_mass,
)


def _best_single_ion(mz_query, sorted_ions, mzs, tol_ppm):
    """Single most plausible ion for one m/z, ignoring FFC compatibility and
    charge -- the same (priority, |error|) choice ffc_annotator.match_mz()
    makes, but returning the Ion itself so the caller can classify with it.

    Returns (label, error_da, ion) or ("No Match", -1000, None).
    """
    cands = candidates_for(mz_query, sorted_ions, mzs, tol_ppm)
    if not cands:
        return "No Match", -1000, None
    best = min(cands, key=lambda ion: (ion.priority, abs(mz_query - ion.mz)))
    return best.label, mz_query - best.mz, best


def _load_top_ffcs(path, top_n, sep='\t'):
    """Load an MMS table and take the top `top_n` FFCs, one row per rank.

    Drops the unranked rows (Ranking <= 0, e.g. the -1 diagonal), collapses
    a pair seen in both orientations onto its best (lowest) rank, and keeps
    a single row per rank.
    """
    df = pd.read_csv(path, sep=sep)
    print(f"Loaded      : {len(df)} rows from {Path(path).name}")

    df = df[df['Ranking'] > 0].sort_values('Ranking')
    pair_key = df.apply(
        lambda r: tuple(sorted((round(r['m/z A'], 3), round(r['m/z B'], 3)))), axis=1)
    df = df[~pair_key.duplicated()]
    df = df[~df['Ranking'].duplicated()]
    print(f"Ranked      : {len(df)} unique pairs, one per rank")

    return df.head(top_n)


def run_nocharge_annotation(path, sequence, charge, output_path=None, top_n=50,
                            residual='H2O', tol_ppm=DEFAULT_TOL_PPM,
                            isomax_cutoff=DEFAULT_ISOMAX_CUTOFF_PCT,
                            enforce_charge_sum=False, sep='\t'):
    """Annotate the top `top_n` ranked FFCs of `path` without using charge.

    Parameters
    ----------
    path : str
        MMS table with 'm/z A', 'm/z B', 'Ranking' columns. No charge
        columns are needed or read.
    sequence, charge : str, int
        Peptide sequence and precursor charge (fragment ions are generated
        for charges 1..charge-1).
    output_path : str, optional
        Output xlsx. Defaults to result/nocharge/<stem>_top<top_n>.xlsx.
    top_n : int
        Number of ranked FFCs to annotate (default 50).
    residual : str or float
        Terminal residual group: 'H2O' (default) or 'NH3', or a numeric mass.
    tol_ppm : float
        Mass tolerance in ppm (default 20, matching the production pipeline).
    isomax_cutoff : float
        Averagine-table isomax cutoff in percent (default 5).
    enforce_charge_sum : bool
        Pass through to the joint pair match (Note 11 criterion ii).
        Default False, as elsewhere in this module.

    Returns
    -------
    pd.DataFrame with columns rank, X, Y, annotation, breaking point,
    m/z X, m/z Y, error X, error Y, TFFC X, TFFC Y, offset, deviation.
    """
    if output_path is None:
        out_dir = Path(__file__).parent / 'result' / 'nocharge'
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f'{Path(path).stem}_top{top_n}.xlsx')

    df_top = _load_top_ffcs(path, top_n, sep=sep)
    print(f"Annotating  : top {len(df_top)} FFCs")

    res_label = residual if isinstance(residual, str) else f'{residual:.5f}'
    print(f"Generating theoretical ions for {sequence} (charge {charge}, "
          f"residual={res_label}, isomax_cutoff={isomax_cutoff}%)...")
    ions, n = generate_theoretical_ions(sequence, charge, residual=residual,
                                        isomax_cutoff=isomax_cutoff)
    print(f"  -> {len(ions)} theoretical m/z values generated.")
    sorted_ions, mzs = build_mz_index(ions)
    parent_mass = compute_peptide_mass(sequence, residual=residual)

    rows = []
    for _, row in df_top.iterrows():
        # iterrows() upcasts through the row's mixed dtypes, so restore the
        # integer rank.
        mz_a, mz_b, ranking = row['m/z A'], row['m/z B'], int(row['Ranking'])

        # Joint match with both charges left unknown: filter_by_charge_soft()
        # is a no-op on a None charge, so this is plain charge-free pair
        # matching -- but it hands back the Ion objects, which
        # match_ffc_pair() does not.
        joint = match_ffc_pair_with_charge(mz_a, mz_b, sorted_ions, mzs, n, charge,
                                           charge_a=None, charge_b=None,
                                           tol_ppm=tol_ppm,
                                           enforce_charge_sum=enforce_charge_sum)
        fallback_used = joint is None

        if joint is not None:
            ion_a, ion_b = joint
            label_a, err_a = ion_a.label, mz_a - ion_a.mz
            label_b, err_b = ion_b.label, mz_b - ion_b.mz
        else:
            # One best candidate per side (not the full candidate list
            # annotate_ffcs() would emit), so the rank keeps a single FFC.
            label_a, err_a, ion_a = _best_single_ion(mz_a, sorted_ions, mzs, tol_ppm)
            label_b, err_b, ion_b = _best_single_ion(mz_b, sorted_ions, mzs, tol_ppm)

        (mz_a, label_a, err_a, mz_b, label_b, err_b, ion_a, ion_b,
         _ca, _cb, _cak, _cbk) = _enforce_by_placement(
            mz_a, label_a, err_a, mz_b, label_b, err_b, ion_a, ion_b)

        display_a = f'*{label_a}' if fallback_used else label_a
        display_b = f'*{label_b}' if fallback_used else label_b

        tffc_a = tffc_mass(mz_a, ion_a.charge) if ion_a is not None else None
        tffc_b = tffc_mass(mz_b, ion_b.charge) if ion_b is not None else None

        if ion_a is not None and ion_b is not None:
            annotation = classify_ffc_relationship(ion_a, ion_b, n)
            row_raw_bp = raw_breakpoints(ion_a, ion_b, n, annotation)
        else:
            annotation = 'unmatched'
            row_raw_bp = None

        row_dict = {
            'rank': ranking,
            'X': display_a,
            'Y': display_b,
            'annotation': annotation,
            'breaking point': None,  # filled in the dedup pass below
            'm/z X': round(mz_a, 3),
            'm/z Y': round(mz_b, 3),
            'error X': round(err_a, 3) if err_a != -1000 else '-',
            'error Y': round(err_b, 3) if err_b != -1000 else '-',
            'TFFC X': round(tffc_a, 3) if tffc_a is not None else -1000,
            'TFFC Y': round(tffc_b, 3) if tffc_b is not None else -1000,
            'offset': (round(parent_mass - (tffc_a + tffc_b), 3)
                       if tffc_a is not None and tffc_b is not None else -1000),
        }
        if (ion_a is not None and ion_b is not None
                and annotation.startswith(('parental', 'secondary', 'internal'))):
            theo_offset = parent_mass - (tffc_mass(ion_a.mz, ion_a.charge)
                                         + tffc_mass(ion_b.mz, ion_b.charge))
            row_dict['deviation'] = round((parent_mass - (tffc_a + tffc_b)) - theo_offset, 4)
        else:
            row_dict['deviation'] = ''

        rows.append((row_dict, row_raw_bp))

    seen = set()
    for row_dict, row_raw_bp in rows:
        row_dict['breaking point'] = _dedup_breakpoint(row_raw_bp, seen)

    df_result = pd.DataFrame([row_dict for row_dict, _ in rows])
    _write_xlsx(df_result, output_path)
    print(f"Saved to    : {output_path}")
    return df_result
