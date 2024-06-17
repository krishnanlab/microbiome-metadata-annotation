"""
This script generates a samples (rows) by terms (columns) prediction score matrix given a set of
individual similarity vectors for each term (i.e. a directory of individual files for each term).

Furthermore, this script will transform the samples (rows) by terms (columns) prediction score 
matrix by one of three methods:
    - none: returns the input score matrix
    - single_z: applies a column-wise Z-score to each element (distribution across a single term)
    - double_z: applies both a column-wise and row-wise Z-score to each element, considering both
                the distributions of terms and samples. These two matrices are then summed together.


Author: Parker Hicks
Date: 2023-12
"""
from argparse import ArgumentParser
from scipy.stats import zscore
from utils import check_outdir, iterdir
from tqdm import tqdm
import numpy as np
import re


def get_weight_param(dirname: str) -> str:
    pattern = r"weight-(none|product|individual)"
    match = re.search(pattern, dirname)

    if match:
        return match.group(1)
    else:
        return None


def normalize_scores(mat: np.array, transform: str) -> np.array:
    """

    :param mat: matrix of similarity scores between each sample description and term name
    :param normalization: type of normalization method to apply to scores
        - raw: returns an unnormalized score matrix
        - single_z: applies a zscore to all rows (terms) to the score matrix
        - double_z: computes the zscore across all columns and all rows and adds them together.
                    Here, we consider the distributions of both the terms and the samples.
    """

    if transform == "none":
        normalized_mat = mat

    elif transform == "single_z":
        # Find rows and columns where all elements are not equal to 0
        row_zero_inds = np.where(np.sum(np.abs(mat), axis=1) != 0)[0]

        # Compute matrix of zscores for rows != 0
        mat_zscore_rows = np.zeros(mat.shape)
        for i in row_zero_inds:
            mat_zscore_rows[i, :] = zscore(mat[i, :])

        # Add together to get double_z score
        normalized_mat = mat_zscore_rows

    elif transform == "double_z":
        # For this method, we need to check that all elements of a given
        # row and column are not the same (usually all 0). This causes
        # the zscore function to return nans which are incompatable for
        # evaluation with auPRC.

        # Find rows and columns where all elements are not equal to 0
        row_zero_inds = np.where(np.sum(np.abs(mat), axis=1) != 0)[0]
        col_zero_inds = np.where(np.sum(np.abs(mat), axis=0) != 0)[0]

        # Get all non-all-zero row,column combinations
        rows, cols = np.ix_(row_zero_inds, col_zero_inds)

        # Compute two matrices of zscores both for rows and columns != 0
        mat_zscore_rows = np.zeros(mat.shape)
        mat_zscore_cols = np.zeros(mat.shape)
        for i in rows.ravel():
            mat_zscore_rows[i, :] = zscore(mat[i, :])
        for j in cols.ravel():
            mat_zscore_cols[:, j] = zscore(mat[:, j])

        # Add together to get double_z score
        normalized_mat = (mat_zscore_rows + mat_zscore_cols) / np.sqrt(2)

    return normalized_mat


def main(
    scores_dir: str,
    transform: str,
    outdir: str,
) -> None:
    # Get list of files from args.scores_dir
    score_files = iterdir(scores_dir)

    terms = []
    scores = []
    for i, file_ in tqdm(
        enumerate(score_files), total=len(score_files), desc="Aggregating files..."
    ):
        terms.append(file_.stem)
        data = np.load(file_, allow_pickle=True)

        # Only need to get gsms once
        if i == 0:
            gsms = data["gsms"]
            scores.append(data["similarity"])
        else:
            scores.append(data["similarity"])

    terms = np.array(terms)
    scores = np.array(scores)  # Shape (terms, samples)

    # Normalize the prediction scores
    print("Normalizing scores by %s." % (transform))
    normalized_scores = normalize_scores(scores, transform)

    # Save aggregate scores
    outdir = check_outdir(outdir)
    weight = get_weight_param(scores_dir)
    outfile = outdir / f"aggregate_prediction_scores__{transform}__weight-{weight}.npz"
    print("Saving file to %s" % (outfile))
    np.savez(outfile, scores=normalized_scores, terms=terms, gsms=gsms)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-scores_dir",
        help="/path/to/scores directory containing <term_name>.npz files with keys 'similarity' and 'gsms'. \
            These files store a similarity score for a term name and all sample descriptions.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-transform",
        help="Indicates how to transform the predictions scores. \
            'raw' just aggregates the vectors into a single matrix. \
            'single_z' computes a Z-score for each score from the distribution of scores for each individual term. \
            'double_z' adds the Z-scores from both the distribution of scores for each term and also the distribution \
                of scores for each sample for a particular d,t description term pair.",
        default="double_z",
        choices=["none", "single_z", "double_z"],
    )
    parser.add_argument(
        "-outdir",
        help="/path/to/outdir",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    main(scores_dir=args.scores_dir, transform=args.transform, outdir=args.outdir)
