"""
This script generate embedding from unique words in the corpus

This script works with our t2g environment using CUDA v11.8.0 on v100 GPUs. 
If using CPUs, reinstall torch that is compatable with CPU-only machines.

Authors: Hao Yuan, Parker Hicks
Date: 2023-07
"""
from transformers import AutoTokenizer, AutoModel
from timeit import default_timer as timer
from utils import check_stem_true_false, check_lemmatize_true_false
from itertools import chain
from pathlib import Path
from tqdm import tqdm

import torch
import argparse
import numpy as np
import pandas as pd


def unique_words(corpus: np.array) -> np.array:
    """Get a np.array of the unique words in a corpus"""
    words = list()
    for sample_ind in range(0, corpus.shape[0]):
        words.extend(corpus[sample_ind].split(" "))
    words = np.unique(np.array(words))
    return words


def merge_term_and_corpus_words(corpus_words: np.array, term_words: np.array):
    """Adds an array of 'outside' words to a corpus"""
    term_names_split = [term.split(" ") for term in term_words]
    term_names_flattened = list(chain.from_iterable(term_names_split))
    words = np.unique(np.concatenate((corpus_words, term_names_flattened)))

    return words


def generate_embeddings(text, tokenizer, model, device):
    """
    Function to generate embeddings for given text
    """
    encoded_input = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
    return embeddings.cpu().numpy()


def load_language_model(model_name):
    """
    load language model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name.split("/")[1] == "BioMedLM":  # BioMedLM requires a padding token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return tokenizer, model, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-corpus",
        help="/path/to/corpus.npz with keys 'corpus' and 'gsms'",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--include_terms",
        help="Use to add tasks to the embedding/similarity matrices",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-term_names",
        help="If --include_terms flag used, add a .tsv file of preprocessed term names. \
                This file is formatted with columns id\tname where id is an ontology term \
                id and name is the preprocessed common name of the term.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-outdir",
        help="Output directory to send embedding matrices to",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-model",
        help="Model used to generate embeddings",
        default="pubmedbert",
        choices=["pubmedbert", "biomedlm", "biomed_electra"],
        type=str,
    )
    args = parser.parse_args()
    start = timer()

    # Load corpus
    print("Loading corpus.")
    corpus = np.load(args.corpus, allow_pickle=True)["corpus"]

    # get unique words in corpus
    words = unique_words(corpus)
    # Set indicator for file name
    with_terms = "False"

    if args.include_terms:
        # Update indicator for file name
        with_terms = "True"

        # Get term names and merge with corpus unique words
        names_df = pd.read_csv(args.term_names, header=0, delimiter="\t")
        term_names = names_df["name"].to_numpy()
        words = merge_term_and_corpus_words(corpus_words=words, term_words=term_names)

    # load language model
    if args.model == "pubmedbert":
        model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        tokenizer, model, device = load_language_model(model_name)

    if args.model == "biomedlm":
        model_name = "stanford-crfm/BioMedLM"
        tokenizer, model, device = load_language_model(model_name)

    if args.model == "biomed_electra":
        model_name = "microsoft/BiomedNLP-BiomedELECTRA-base-uncased-abstract"
        tokenizer, model, device = load_language_model(model_name)

    print(f"Using device: {device}")

    # Initialize an empty NumPy array to store the embeddings
    num_terms = len(words)
    embedding_size = model.config.hidden_size
    embeddings_array = np.zeros((num_terms, embedding_size))

    # generate embeddings
    for i, word in tqdm(
        enumerate(words), total=len(words), desc="generating embeddings..."
    ):
        embeddings_array[i] = generate_embeddings(word, tokenizer, model, device)

    # generate cosine similarity among words vectors
    cosine_similarity = np.dot(embeddings_array, embeddings_array.T) / np.dot(
        np.linalg.norm(embeddings_array, axis=1)[:, None],
        np.linalg.norm(embeddings_array, axis=1)[None, :],
    )

    # Set stem and lemmatize file indicators
    stem_ = check_stem_true_false(Path(args.corpus).stem)
    lemmatize_ = check_lemmatize_true_false(Path(args.corpus).stem)

    # Save embedding table
    print("Saving embeddings file.")
    outdir = Path(args.outdir)
    outfile = (
        outdir
        / f"embedding_lookup_table_{args.model}_terms-{with_terms}_stem-{stem_}_lemmatize-{lemmatize_}.npz"
    )
    np.savez_compressed(
        outfile,
        embedding=embeddings_array,
        cosine_similarity=cosine_similarity,
        words=words,
    )
    end = timer()
    print(f"Data parsed in {(end-start) / 60} mins.")
