from pathlib import Path
import numpy as np
import pandas as pd
import nltk
import string
import re


def iterdir(dir):
    return [Path(file_) for file_ in Path(dir).glob("*")]


def check_outdir(path_: str) -> Path:
    outdir = Path(path_)
    if not outdir.exists():
        outdir.mkdir(parents=True)
    return outdir


def check_stem_true_false(filename):
    pattern = r"stem-(True|False)_"
    match = re.search(pattern, filename)

    if match:
        return match.group(1)  # Returns 'True' or 'False'
    else:
        return None  # If not found


def check_lemmatize_true_false(filename):
    pattern = r"lemmatize-(True|False)"
    match = re.search(pattern, filename)

    if match:
        return match.group(1)  # Returns 'True' or 'False'
    else:
        return None  # If not found


def save_dataframe_to_npz(dataframe, filename):
    """
    Save Pandas DataFrame columns as different keys in a NumPy .npz file.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        filename (str): The name of the output .npz file.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")

    column_names = dataframe.columns
    column_data = [dataframe[column].values for column in column_names]

    np.savez(filename, **{name: data for name, data in zip(column_names, column_data)})


def remove_unencoded_text(text):
    """
    Removes characters that are not UTF-8 encodable.
    """
    return "".join([i if ord(i) < 128 else "" for i in text])


def is_allowed_word(word, stopwords, remove_numbers, min_word_len):
    """
    Checks if word is allowed based on inclusion in stopwords, presence of
    numbers, and length.
    """
    stopwords_allowed = word not in stopwords
    numbers_allowed = not (remove_numbers and contains_numbers(word))
    length_allowed = len(word) >= min_word_len
    return stopwords_allowed and numbers_allowed and length_allowed


def contains_numbers(text):
    """
    Parses text using a regular expression and returns a boolean value
    designating whether that string contains any numbers.
    """
    return bool(re.search(r"\d", text))


def preprocess(
    text,
    stopwords=set(nltk.corpus.stopwords.words("english")),
    stem=False,
    lemmatize=False,
    keep_alt_forms=False,
    remove_numbers=True,
    min_word_len=2,
):
    """
    Standardized preprocessing of a line of text. Made by Anna Yannakopoulos
    2020. Added by NTH on 21 Jan 2020.
    """

    # remove non utf-8 characters
    text = remove_unencoded_text(text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # convert all whitespace to spaces for splitting
    whitespace_pattern = re.compile(r"\s+")
    text = re.sub(whitespace_pattern, " ", text)

    # lowercase the input
    text = text.lower()

    # split into words
    words = text.split(" ")

    # stem and/or lemmatize words
    # filtering stopwords, numbers, and word lengths as required
    stemmer = nltk.stem.porter.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    if stem and lemmatize:
        words = [
            [word, stemmer.stem(word), lemmatizer.lemmatize(word)]
            for word in words
            if is_allowed_word(word, stopwords, remove_numbers, min_word_len)
        ]
    elif stem:
        words = [
            [word, stemmer.stem(word)]
            for word in words
            if is_allowed_word(word, stopwords, remove_numbers, min_word_len)
        ]
    elif lemmatize:
        words = [
            [word, lemmatizer.lemmatize(word)]
            for word in words
            if is_allowed_word(word, stopwords, remove_numbers, min_word_len)
        ]
    else:
        words = [
            word
            for word in words
            if is_allowed_word(word, stopwords, remove_numbers, min_word_len)
        ]

    if stem or lemmatize:
        if keep_alt_forms:
            # return both original and stemmed/lemmatized words
            # as long as stems/lemmas are unique
            words = [w for word in words for w in set(word)]
        else:
            # return only requested stems/lemmas
            # if both stemming and lemmatizing, return only lemmas
            words = list(zip(*words))[-1]

    words = [word for word in words if len(word) >= 2]

    return " ".join(words)
