import numpy as np
import scanpy as sc
import itertools
from typing import Tuple
import pandas as pd
from scipy.optimize import nnls
import argparse


def filter_tfs(
        initial: sc.AnnData,
        target: sc.AnnData,
        target_expression: float,
        fold_change: float,
        tf_list: np.ndarray
) -> np.ndarray:
    """
    Filters the transcription factors based on the target expression and fold change

    TODO: handle zeros in initial and target
    """

    initial = initial[:, np.isin(initial.var_names, tf_list)]
    target = target[:, np.isin(target.var_names, tf_list)]

    mask = np.logical_and(np.isin(tf_list, target.var_names), np.isin(tf_list, initial.var_names))

    expression_condition = target.X >= target_expression
    fold_change_condition = np.log2(target.X / initial.X) >= fold_change

    return tf_list[mask][np.logical_and(expression_condition, fold_change_condition).squeeze()]


def generate_recipes(
        tf_list: np.ndarray,
        n: int
) -> list[str]:
    """
    Generates all possible combinations of transcription factors
    """

    if n > 3 or n < 1:
        raise ValueError(f'The number of transcription factors must be 1, 2, or 3. Received {n=}')

    recipes = list(itertools.combinations(tf_list, 1))
    for ii in range(2, n + 1):
        recipes.extend(itertools.combinations(tf_list, ii))
    return recipes


def preprocess_data(
        adata: sc.AnnData,
        target: sc.AnnData
) -> Tuple[sc.AnnData, sc.AnnData]:
    """
    Preprocesses the data for the DGC model
    """

    # load in steady-state value, ($\\bar{x}$ in the paper)
    steady_state = sc.read_h5ad('data/raw/fibroblast_ss.h5ad')

    # Set microRNA genes to zero
    gene_idx_to_zero = np.arange(11165, 12324)
    adata.X[:, gene_idx_to_zero] = 0.
    target.X[:, gene_idx_to_zero] = 0.

    # Collect the set of genes that do not have expression values
    nan_genes = set(target.var_names[np.isnan(target.X).any(axis=0)])
    nan_genes.update(steady_state.var_names[np.isnan(steady_state.X).any(axis=0)])
    nan_genes.update(adata.var_names[np.isnan(adata.X).any(axis=0)])

    # Set microRNA genes to zero
    steady_state.X[:, gene_idx_to_zero] = 0.

    # Remove genes that do not have expression values
    target = target[:, ~target.var_names.isin(nan_genes)].copy()
    steady_state = steady_state[:, ~steady_state.var_names.isin(nan_genes)]
    adata = adata[:, ~adata.var_names.isin(nan_genes)].copy()

    # center data
    adata.X = adata.X - steady_state.X
    target.X = target.X - steady_state.X

    return adata, target


def map_genes_to_TADs(
        adata: sc.AnnData,
        axis: str = 'var',
        func: str = 'sum'
) -> sc.AnnData:
    # Map the vectors to TAD space
    adata = sc.get.aggregate(adata, by='TAD', func=func, axis=axis)
    adata.X = adata.layers['sum']

    return adata


def thinh(txt):
    print(txt)


class DGC:

    def __init__(
            self,
            gene_list: list[str] | None = None
    ) -> None:

        self.A = None
        self.xdim = None
        self.get_B_matrix(gene_list)

    def get_B_matrix(
            self,
            gene_list: list[str] | None = None
    ) -> None:
        """
        Instantiates the B matrix for DGC as an AnnData object.
        Rows are genes and columns are transcription factors.
        Entry b_ij = 1 if TF j can influence gene i (based on data from either HuRI or STRING). Otherwise b_ij = 0. 
        """

        self.B = sc.read_h5ad('data/raw/B_matrix_2015.h5ad')
        gene_access = pd.read_csv('data/raw/gene_accessibility.csv', index_col=0)

        # subset to genes in B matrix with same order
        mask = (gene_access.loc[self.B.obs_names]).values

        self.B.X = self.B.X * mask
        self.B = self.B[gene_list]

        self.B = map_genes_to_TADs(self.B, axis='obs')

    def build_A_matrices(
            self,
            adata: sc.AnnData
    ) -> None:

        """
        Computes array of the time-varying A matrices for the DGC model and assigns it to self.A
        self.A = [A_{n-1}, A_{n-1}A_{n-2}, ..., A_{n-1}...A_1]
        """

        num_data = adata.shape[0]
        xdim = adata.X.shape[1]

        I_tad = np.identity(xdim)
        A = np.zeros((num_data - 1, xdim, xdim))

        for ii, tt in enumerate(range(num_data - 1, 0, -1)):
            # Compute A for time (tt-1)
            num = np.outer(adata.X[tt] - adata.X[tt - 1], adata.X[tt - 1])
            den = np.dot(adata.X[tt - 1], adata.X[tt - 1])

            if ii == 0:
                A[ii] = (num / den) + I_tad
            else:
                A[ii] = A[ii - 1] @ ((num / den) + I_tad)

        self.A = A
        self.xdim = xdim

    def estimate_tfs_constant(
            self,
            initial: np.ndarray,
            target: np.ndarray,
            recipe_list: list[str]
    ) -> dict[str, dict[str, np.ndarray]]:

        """
        Estimates the transcription factors according to DGC

        TODO: double check this optimization
        """

        Cbar = np.eye(self.xdim) + np.sum(self.A[1:], axis=0)
        b = target - self.A[-1] @ initial

        distances = {}
        for tfs in recipe_list:
            # select columns corresponding to the TFs
            B = self.B.X[:, self.B.var_names.isin(tfs)]
            # print(B.shape)

            # solve the non-negative least squares problem
            u, d = nnls(Cbar @ B, b)

            # save the results to the dictionary
            distances[tfs] = {'d': d, 'u': u}

        return distances


def print_scores(
        sorted_scores: dict[str, float]
) -> None:
    """
    Prints the scores from the optimization
    """

    max_recipe_len = max(len(k) for k in sorted_scores.keys())
    print(f"{'Recipe':<{max_recipe_len}}  | {'Score':>8}")

    # Print a separator line for clarity
    print("-" * (max_recipe_len + 3 + 10))
    for recipe, score in sorted_scores.items():
        print(f"{recipe:<{max_recipe_len}}  | {score:>8.2f}")



def thinh_print():
    print("do huu thinh")