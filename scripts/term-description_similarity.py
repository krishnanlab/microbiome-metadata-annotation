"""
This script computes the similarity between a term name and all sample descriptions in a corpus
all words in term name and all words in a sample description.

Example:

Say we have an ontology term of three words {t1, t2, t3} and a sample description we want to compare
it to also with 3 words {s1, s2, s3}. We compute the cosine similarity of the embedding vectors of each
word to each other to get a similarity score cos_sim[i, j] for every term word ti and description word sj.

     s1  s2  s3  | max
   ______________|____
t1|  0.5 0.4 0.1 | 0.5
  |              |
t2|  0.2 0.9 0.1 | 0.9
  |              |
t3|  0.7 0.3 0.7 | 0.7
__|______________|
max  0.7 0.9 0.7


The overall similarity between the ontology term name (t) and the sample description (s) is computed by
averaging all values of the row max values and the column max values. 

In this case scenario:
(0.7 + 0.9 + 0.7 + 0.5 + 0.9 + 0.7) / 6 = 0.73

Thus, the overall cosine similarity between the ontology term and the sample description is 0.73.

Authors: Parker Hicks
Date: 2023-12-21
"""
import sys

sys.path.insert(1, "src")

from tfidf_calculator import TfidfCalculator
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time


def array_sum(_a: np.array, _b: np.array) -> float:
    """Computes the sum of maximum values across all rows and columns of a matrix."""
    # For testing
    print(_a.shape)
    print(_b.shape)
    return np.sum(_a) + np.sum(_b)


def mask_tfidf(split_term, split_desc, tfidf_word_features, tfidf) -> np.array:
    """Get TF-IDF values for words in the sample description and term names.

    :param split_term: a list of each word in a specific ontology term
    :param split_desc: a list of each word in a particular sample description
    :param tfidf_word_features: a vector of words representing each column of a tfidf matrix
    :param tfidf: a samples (rows) by words (cols) tfidf matrix
    :returns tfidf_term: a vector of tfidf values for each word in a term name
    :returns tfidf_desc: a vector of tfidf values for each word in a sample descriptions
    """
    word_features = tfidf_word_features
    tfidf_ = tfidf

    for word in split_term:
        if word not in word_features:
            # Word not in tfidf word features, add it with value 0
            word_features = np.append(word_features, word)
            tfidf_ = np.append(tfidf_, 0)  # Append 0 to tfidf vector

    for word in split_desc:
        if word not in word_features:
            # Word not in tfidf word features, add it with value 0
            word_features = np.append(word_features, word)
            tfidf_ = np.append(tfidf_, 0)  # Append 0 to tfidf vector

    # Get a mask for the tfidf vector selecting values for words
    # that are in the description and term.
    term_mask = np.where(np.isin(word_features, split_term))[0]
    desc_mask = np.where(np.isin(word_features, split_desc))[0]

    # Use sorted indices to get sorted tfidf values
    tfidf_term = tfidf_[term_mask][np.argsort(split_term)]
    tfidf_desc = tfidf_[desc_mask][np.argsort(split_desc)]

    return tfidf_term, tfidf_desc


def avg_similarity(max_rows: np.array, max_cols: np.array) -> float:
    """Computes the average of elements across two vectors.

    :param max_rows: a vector of maximum similarity values from the
        rows of a (term words) x (description words) matrix
    :param max_cols: a vector of maximum similarity values from the
        columns of a (term words) x (description words) matrix
    """
    total = array_sum(max_rows, max_cols)
    avg_sample_sim = total / (len(max_rows) + len(max_cols))

    return avg_sample_sim


def weighted_avg_similarity(
    max_values_rows: np.array,
    max_values_cols: np.array,
    tfidf_term: np.array,
    tfidf_desc: np.array,
    weight: str,
    sample_desc_mat: np.array,
) -> float:
    if weight == "individual":
        # Multiply by the tfidf for the word that produced a max
        # value for rows and columns
        max_rows_weighted, max_cols_weighted = individual(
            max_values_row=max_values_rows,
            max_values_column=max_values_cols,
            tfidf_desc=tfidf_desc,
            tfidf_term=tfidf_term,
        )
    elif weight == "product":
        # Get the opposite axis indices for which the maximum value lies
        max_row_value_col_inds = np.argmax(sample_desc_mat, axis=1)
        max_col_value_row_inds = np.argmax(sample_desc_mat, axis=0)

        max_rows_weighted, max_cols_weighted = product(
            max_values_row=max_values_rows,
            max_values_column=max_values_cols,
            max_row_value_col_inds=max_row_value_col_inds,
            max_col_value_row_inds=max_col_value_row_inds,
            tfidf_term=tfidf_term,
            tfidf_desc=tfidf_desc,
        )

    # Average by the total tfidf for all words in sample description and term name
    total = array_sum(max_rows_weighted, max_cols_weighted)
    total_tfidf = np.sum(tfidf_term) + np.sum(tfidf_desc)
    avg_sample_sim = total / total_tfidf

    return avg_sample_sim


def product(
    max_values_row,
    max_values_column,
    max_row_value_col_inds,
    max_col_value_row_inds,
    tfidf_term,
    tfidf_desc,
):
    """Weighting strategy to multiply the maximum value for a row by the tfidf of the word representing
    that row and by the tfidf of the word representing the column for which the maximum similarity was
    found. Same procedure for columns.

    :param max_values_row: a vector of shape (len(term_words),) containing the
        maximum values across the similarity matrix rows
    :param max_values_column: a vector of shape (len(description_words),)
        containing the maximum values across the similarity matrix columns
    :param max_row_value_col_inds: a vector containing the column indices that
        gave the maximum similarity for all rows.
    :param max_col_value_row_inds: a vector containing the row indices that
        gave the maximum similarity for all columns.
    :param tfidf_term: a vector with shape (max_values_row) containing tfidf values
        to weight max_values_row
    :param tfidf_desc: a vector with shape (max_values_column) containing tfidf
        values of the description words to weight the max_values_column
    """
    max_values_row_weighted = (
        max_values_row * tfidf_term * tfidf_desc[max_row_value_col_inds]
    )
    max_values_column_weighted = (
        max_values_column * tfidf_desc * tfidf_term[max_col_value_row_inds]
    )

    return max_values_row_weighted, max_values_column_weighted


def individual(max_values_row, max_values_column, tfidf_term, tfidf_desc):
    """Weighting strategy to multiply the maximum value for a row by the tfidf of the word representing
    that row. Same procedure for columns.

    :param max_values_row: a vector of shape (len(term_words),) containing the
        maximum values across the similarity matrix rows
    :param max_values_column: a vector of shape (len(description_words),)
        containing the maximum values across the similarity matrix columns
    :param tfidf_term: a vector with shape (max_values_row) containing tfidf values
        to weight max_values_row
    :param tfidf_desc: a vector with shape (max_values_column) containing tfidf
        values of the description words to weight the max_values_column
    """
    max_cols = max_values_column * tfidf_desc
    max_rows = max_values_row * tfidf_term

    return max_rows, max_cols


def compute_similarity(
    term_name: str,
    description: str,
    words: np.array,
    similarity_mat: np.array,
    weight: str,
    tfidf=None,
    tfidf_word_features=None,
) -> float:
    """Selects word embeddings from the embedding table and computes the average similarity of \
    the embedding vectors for each i,j word combination in the term name and sample description.

    :param term_name: Name of the ontology term
    :param description: A sample description as a single string
    :param words: Array of word columns from the embedding lookup table
    :param similarity_mat: words x words matrix of similarity values between each word embedding
    :param weight: which weighting schema to use
    :param tfidf: if score is weighted, it requires a tfidf vector for a sample
    :param tfidf_word_features: words associated with each element of the tfidf vector
    """

    # Split into individual words and sort alphabetically
    split_desc = np.sort(np.unique(np.array(description.split(" "))))
    split_term = np.sort(np.unique(np.array(term_name.split(" "))))

    # Get indices of similarity matrix for (t, s) word pairs
    t_indices = np.where(np.isin(words, split_term))[0]
    s_indices = np.where(np.isin(words, split_desc))[0]

    # Get index pairs for sets
    i, j = np.meshgrid(t_indices, s_indices, indexing="ij")

    # Use the indices to extract the subset of the 2D array
    sample_desc_mat = similarity_mat[i, j]

    # Get the maximum similarity values across all rows and columns
    max_values_row = np.max(sample_desc_mat, axis=1, keepdims=False)
    max_values_column = np.max(sample_desc_mat, axis=0, keepdims=False)

    if weight == "none":
        similarity = avg_similarity(max_values_row, max_values_column)

    else:
        tfidf_term, tfidf_desc = mask_tfidf(
            split_term=split_term,
            split_desc=split_desc,
            tfidf_word_features=tfidf_word_features,
            tfidf=tfidf,
        )

        similarity = weighted_avg_similarity(
            max_values_cols=max_values_column,
            max_values_rows=max_values_row,
            tfidf_term=tfidf_term,
            tfidf_desc=tfidf_desc,
            weight=weight,
            sample_desc_mat=sample_desc_mat,
        )

    return similarity


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-term",
        help="Name of the term to compute term-corpus similarity on.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-corpus",
        help="/path/to/corpus.npz containing filtered, preprocessed, sample descriptions.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-embeddings",
        help="/path/to/embedding_lookup_table.npz storing embeddings for each word across the corpora",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-weight",
        help="Which weighting scheme to use when computing the average similarity.",
        required=False,
        default="none",
        choices=["none", "individual", "product"],
        type=str,
    )
    parser.add_argument("-outdir", help="Path to outdir", required=True, type=str)
    args = parser.parse_args()
    start = time.time()

    # Load corpus data
    data = np.load(Path(args.corpus), allow_pickle=True)
    corpus = data["corpus"]
    gsms = data["gsms"]

    # Load embedding data
    embedding_table = np.load(args.embeddings, allow_pickle=True)
    cosine_similarity = embedding_table["cosine_similarity"]
    words = embedding_table["words"]

    # Get name of the ontology term id
    term = args.term
    term_name = term

    print("Calculating similarities to %s." % (term))
    sample_similarities = np.zeros((len(corpus)))
    if args.weight == "none":
        for i, desc in tqdm(
            enumerate(corpus), total=len(corpus), desc="Generating sample rankings..."
        ):
            sample_similarities[i] = compute_similarity(
                term_name=term_name,
                description=desc,
                words=words,
                weight=args.weight,
                similarity_mat=cosine_similarity,
            )
    else:
        tfidf_calculator = TfidfCalculator(corpus)
        tfidf = tfidf_calculator.calculate_tfidf()
        word_features = tfidf_calculator.get_word_features()

        for i, desc in tqdm(
            enumerate(corpus), total=len(corpus), desc="Generating sample rankings..."
        ):
            sample_tfidf = tfidf[i, :]
            sample_similarities[i] = compute_similarity(
                term_name=term_name,
                description=desc,
                words=words,
                similarity_mat=cosine_similarity,
                tfidf=sample_tfidf,
                weight=args.weight,
                tfidf_word_features=word_features,
            )
    outfile = Path(args.outdir) / f"{term}.npz"
    print("Saving file %s" % (outfile))
    np.savez(outfile, similarity=sample_similarities, gsms=gsms)
