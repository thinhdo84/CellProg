import numpy as np
import scanpy as sc
import pandas as pd
from scipy.optimize import nnls
import argparse
import anndata as ad
import matplotlib.pyplot as plt
from src.utils import *
from casadi import *
import time



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--initial', type=str, default='fibroblast')
    argparser.add_argument('--target', type=str, default='myotube')
    argparser.add_argument('--recipe_len', type=int, default=1)
    argparser.add_argument('--target_expression', type=float, default=10.)
    argparser.add_argument('--fold_change', type=float, default=4.65)
    # args = argparser.parse_args()
    args = argparser.parse_args([])

    if args.initial != 'fibroblast':
        raise ValueError(f'In the current version, initial cell must be fibroblast but got {args.initial}')

    cell_dict = {
        'esc' : 'ESC (ENCODE GSE23316)',
        'myotube' : 'Myotube (ENCODE GSE52529)'
    }

    adata = sc.read_h5ad('data/raw/A_matrix_2015.h5ad')
    cell_profiles = sc.read_h5ad('data/raw/cell_targets.h5ad')

    target = cell_profiles[cell_profiles.obs.index == cell_dict[args.target]].copy()

    adata, target = preprocess_data(adata, target)
    dgc = DGC(gene_list=adata.var_names.to_list())

    all_tfs = np.array(dgc.B.var_names.to_list())
    candidate_tfs = filter_tfs(
        initial = adata[0],
        target = target,
        target_expression = args.target_expression,
        fold_change = args.fold_change,
        tf_list = all_tfs
    )
    recipe_list = generate_recipes(candidate_tfs, args.recipe_len)

    adata = map_genes_to_TADs(adata)
    target = map_genes_to_TADs(target)
    x_Initital = adata.X[0]

    d0 = np.linalg.norm(adata.X[-1] - target.X[0], ord=2)

    dgc.build_A_matrices(adata)

    sol = dgc.estimate_tfs_constant(
        initial = adata.X[0],
        target = target.X[0],
        recipe_list = recipe_list
    )

    sorted_scores = {', '.join(k): d0 - v['d'] for k, v in sorted(sol.items(), key=lambda item: item[1]['d'])}

    print_scores(sorted_scores)

    adata = dgc.B

    X = adata.layers["sum"] if "sum" in adata.layers else adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    plt.figure(figsize=(10, 8))
    plt.imshow(X, aspect="auto", interpolation="nearest")
    plt.xlabel("TFs (vars)")
    plt.ylabel("Cells / obs")
    plt.title("Heatmap of adata matrix")
    plt.colorbar(label="value")
    plt.show()