"""
Benjamini-Hochberg FDR correction, applied separately within each of the
three full-scale recursion tests (center-embedding, tail-end embedding,
boundary embedding), as reported in the Results and Methods sections of
the paper.

Usage:
    python fdr_correction.py ../data/derived/all_recursion_FINAL.csv
"""
import sys
import pandas as pd
from statsmodels.stats.multitest import multipletests


def correct_all_tests(df, alpha=0.05):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns aba_p, tail_p, bound_p (one row per file).
    alpha : float
        FDR threshold.

    Returns
    -------
    dict of {test_name: corrected DataFrame with an added 'fdr_sig' column}
    """
    out = {}
    for test_name, pcol, zcol in [
        ("center_embedding", "aba_p", "aba_z"),
        ("tail_end_embedding", "tail_p", "tail_z"),
        ("boundary_embedding", "bound_p", "bound_z"),
    ]:
        sub = df[df[pcol].notna()].copy()
        reject, p_adj, _, _ = multipletests(sub[pcol].values, alpha=alpha, method="fdr_bh")
        sub["p_adjusted"] = p_adj
        sub["fdr_significant"] = reject
        out[test_name] = sub

        n_raw = (sub[pcol] < alpha).sum()
        n_fdr = reject.sum()
        print(f"{test_name}: n={len(sub)}, raw p<{alpha}: {n_raw}, FDR-significant: {n_fdr}")

        # species with FDR-surviving POSITIVE results specifically
        surviving_positive = sub[sub.fdr_significant & (sub[zcol] > 0)]
        if len(surviving_positive) > 0:
            species = sorted(surviving_positive["species"].unique())
            print(f"   FDR-surviving positive-direction species: {species}")
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/derived/all_recursion_FINAL.csv"
    df = pd.read_csv(path)
    correct_all_tests(df)
