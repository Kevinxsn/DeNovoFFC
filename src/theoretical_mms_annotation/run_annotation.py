import sys
from pathlib import Path

# merge_duplicate_ffcs / merge_duplicate_ffcs_charge are generic dedup
# utilities with no dependency on the old simple_annot.py annotator, so
# they're reused directly from their original location rather than
# duplicated here.
sys.path.insert(0, str(Path(__file__).parent.parent / 'deiso' / 'python_code'))
from merge_ffcs import merge_duplicate_ffcs, merge_duplicate_ffcs_charge

from ffc_annotator import annotate_ffcs, annotate_ffcs_with_charge
import pandas as pd


def run_deconv_annotation(
    tsv_path,
    sequence,
    charge,
    output_path=None,
    top_n=2000,
    residual='H2O',
    tol_ppm=50,
    isomax_cutoff=5.0,
    use_charge=True,
    extra_info=True,
):
    """
    Full deconvolution annotation pipeline for one peptide dataset.

    Steps
    -----
    1. Load the merged TSV.
    2. Merge duplicate FFCs (symmetric, reset ranking).
    3. Take the top `top_n` rows by ranking.
    4. Annotate each m/z pair against the theoretical FFC map (compatible
       b/y/internal ion pairs only; single neutral loss from H2O/NH3/CO on
       any ion type; isomax from the Averagine table). If use_charge=True,
       uses the charge-aware annotator (ffc_annotator.annotate_ffcs_with_charge):
       each side is constrained to its charge_A/charge_B assignment from the
       TSV (from line-finding/deconvolution) whenever that value isn't NaN,
       and if no compatible pair is found jointly, each side is still
       annotated independently -- the single best match if its charge is
       known, or every matching candidate (joined by " / ") if unknown. If
       use_charge=False, uses the plain annotator (ffc_annotator.annotate_ffcs),
       which ignores charge entirely and works on input with no charge_A /
       charge_B columns at all -- extra_info is not available in this case
       and is silently ignored.
    5. Reorder columns and save to xlsx.

    Parameters
    ----------
    tsv_path : str
        Path to the *_merged.tsv deconvolution output. Must contain
        'charge_A' and 'charge_B' columns (may hold NaN per row) if
        use_charge=True; not required if use_charge=False.
    sequence : str
        Peptide sequence (single-letter codes).
    charge : int
        Precursor charge state.
    output_path : str, optional
        Output xlsx path. If None, auto-derived from the tsv filename
        (strips a trailing _merged suffix) and saved under result/deconv_annot/.
    top_n : int
        Max rows to annotate after merging duplicates (default 2000).
    residual : str or float
        Terminal residual group: 'H2O' (default) or 'NH3', or a numeric mass.
    tol_ppm : float
        Mass tolerance in ppm (default 50).
    isomax_cutoff : float
        Relative-intensity cutoff (percent) for the Averagine-table isomax
        lookup (default 5).
    use_charge : bool
        If True (default), annotate with per-side charge constraints and the
        single-m/z fallback (annotate_ffcs_with_charge). If False, ignore
        charge entirely and use the plain FFC-pair annotator (annotate_ffcs).
    extra_info : bool
        If True (default) and use_charge=True, add the 'annotation',
        'breaking point', 'TFFC X', 'TFFC Y', and 'offset' columns from
        ffc_annotator.annotate_ffcs_with_charge(). Has no effect when
        use_charge=False.

    Returns
    -------
    If use_charge=True: pd.DataFrame with columns X | Y | annotation |
        breaking point | m/z X | m/z Y | error X | error Y | rank |
        TFFC X | TFFC Y | offset (the last 5 only if extra_info=True) --
        see ffc_annotator.annotate_ffcs_with_charge().
    If use_charge=False: pd.DataFrame with columns Interpretation A |
        Interpretation B | m/z A | m/z B | error A | error B | Ranking.
    """
    # Auto-derive output path from input filename if not given
    if output_path is None:
        stem = Path(tsv_path).stem
        if stem.endswith('_merged'):
            stem = stem[:-len('_merged')]
        out_dir = Path('result/deconv_annot')
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f'{stem}_v2.xlsx')

    # 1. Load
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded      : {len(df)} rows from {Path(tsv_path).name}")

    # 2. Merge duplicates
    if use_charge:
        df_merged = merge_duplicate_ffcs_charge(df, symmetric=True, reset_ranking=True)
    else:
        df_merged = merge_duplicate_ffcs(df, tolerance=0.001, symmetric=True, reset_ranking=True)
    print(f"After merge : {len(df_merged)} unique pairs")

    # 3. Top N
    df_top = df_merged.head(top_n)

    # 4. Annotate against the theoretical FFC map (writes a first-pass xlsx internally)
    if use_charge:
        df_annot = annotate_ffcs_with_charge(df_top, sequence, charge,
                                              output_path=output_path,
                                              tol_ppm=tol_ppm,
                                              residual=residual,
                                              isomax_cutoff=isomax_cutoff,
                                              extra_info=extra_info)
    else:
        df_annot = annotate_ffcs(df_top, sequence, charge,
                                  output_path=output_path,
                                  tol_ppm=tol_ppm,
                                  residual=residual,
                                  isomax_cutoff=isomax_cutoff)

    # 5. Reorder columns. annotate_ffcs_with_charge() already returns its
    # columns in the desired final order/names (X, Y, annotation,
    # breaking point, m/z X, m/z Y, error X, error Y, rank, TFFC X, TFFC Y,
    # offset), so only the plain (use_charge=False) path -- whose column
    # names/order are unchanged -- needs reordering here.
    if not use_charge:
        col_order = ['Interpretation A', 'Interpretation B',
                     'm/z A', 'm/z B', 'error A', 'error B', 'Ranking']
        df_annot = df_annot[col_order]
        df_annot.to_excel(output_path, index=False)

    print(f"Saved to    : {output_path}")

    return df_annot

    