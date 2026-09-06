"""
Naive OLS vs. phylogenetic generalized least squares (PGLS) regression of
tail-end embedding z-score on ordinal vocal-learning category, using the
composite phylogeny built for this study.

Requires: ../phylogeny/phylo_tree_v5.nwk (or the covariance matrix +
labels, already precomputed as phylo_cov_matrix_v5.npy / phylo_labels_v5.txt
to save re-deriving it from the Newick tree every run).

Usage:
    python pgls_analysis.py
"""
import numpy as np
import pandas as pd
from scipy import stats


def build_covariance_from_newick(newick_path):
    """
    Rebuild the phylogenetic covariance matrix from scratch from the Newick
    tree, if you don't want to rely on the precomputed .npy file. Requires
    dendropy (pip install dendropy).
    """
    import dendropy
    tree = dendropy.Tree.get(path=newick_path, schema="newick")
    tree.is_rooted = True
    leaves = list(tree.leaf_node_iter())
    labels = [l.taxon.label.replace(" ", "_") for l in leaves]
    n = len(leaves)

    def dist_to_root(node_):
        d = 0
        n_ = node_
        while n_.parent_node is not None:
            d += n_.edge.length
            n_ = n_.parent_node
        return d

    root_dists = [dist_to_root(l) for l in leaves]
    C = np.zeros((n, n))
    for i, li in enumerate(leaves):
        for j, lj in enumerate(leaves):
            if i == j:
                C[i, j] = root_dists[i]
            else:
                mrca_node = tree.mrca(taxa=[li.taxon, lj.taxon])
                C[i, j] = dist_to_root(mrca_node)
    return C, labels


def run_ols_pgls(y, X, C):
    """Shared regression math for both the naive and phylogenetic fits."""
    n, p = X.shape

    beta_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_ols
    sigma2 = (resid @ resid) / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta_ols / se
    p_ols = 2 * (1 - stats.t.cdf(np.abs(t), n - p))

    C_inv = np.linalg.inv(C)
    XtCinv = X.T @ C_inv
    beta_pgls = np.linalg.inv(XtCinv @ X) @ (XtCinv @ y)
    resid_p = y - X @ beta_pgls
    sigma2_p = (resid_p @ C_inv @ resid_p) / (n - p)
    se_p = np.sqrt(np.diag(sigma2_p * np.linalg.inv(XtCinv @ X)))
    t_p = beta_pgls / se_p
    p_pgls = 2 * (1 - stats.t.cdf(np.abs(t_p), n - p))

    return dict(
        ols_slope=beta_ols[1], ols_p=p_ols[1],
        pgls_slope=beta_pgls[1], pgls_p=p_pgls[1],
    )


def main(results_csv="../data/derived/all_recursion_FINAL.csv",
         cov_matrix_path="../phylogeny/phylo_cov_matrix_v5.npy",
         labels_path="../phylogeny/phylo_labels_v5.txt",
         value_col="tail_z"):
    df = pd.read_csv(results_csv)
    sp_means = df.groupby(["species", "learning_category"])[value_col].mean().reset_index()

    with open(labels_path) as f:
        labels = [l.replace(" ", "_") for l in f.read().strip().split("\n")]
    C_full = np.load(cov_matrix_path)

    ordinal_map = {"nonlearner": 0, "usage_learner": 1, "production_learner": 2}
    sub = sp_means[sp_means.learning_category.isin(ordinal_map.keys())].copy()
    sub["category_code"] = sub.learning_category.map(ordinal_map)
    matched = [l for l in labels if l in sub["species"].values]
    sub = sub.set_index("species").loc[matched].reset_index()

    idx = [labels.index(s) for s in sub["species"].tolist()]
    C = C_full[np.ix_(idx, idx)]

    y = sub[value_col].values
    X = np.column_stack([np.ones(len(sub)), sub["category_code"].values])

    res = run_ols_pgls(y, X, C)
    print(f"n = {len(sub)} species")
    print(f"Naive OLS:  slope = {res['ols_slope']:.3f}, p = {res['ols_p']:.4f}")
    print(f"PGLS:       slope = {res['pgls_slope']:.3f}, p = {res['pgls_p']:.4f}")
    return res


if __name__ == "__main__":
    main()
