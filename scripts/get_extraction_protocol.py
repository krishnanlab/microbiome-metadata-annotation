"""
This script is used to identify potential extraction protocols from metadata.

Notes
_________
1. Metadata fields/keys tend to contain underscores whereas other words in the metadata do not (i.e. samp_mat_process,
samp_collect_device, env_medium, isolation_source, collection_method). This of course disregards fields denoted by
single words, however the fields we were interested in contain underscores.

2. Potential fields identified in step 1 are saturated with nonsense entities like filenames (i.e.
 prw-201_s210_l001_r1_001.fastq.gz) or others with no discernible meaning (kr00325_m1). Since relevant metadata fields
 tend to not contain numbers, we remove all potential fields identified by step 1 that contain numerical characters.

3. The fields env_medium and isolation_source contain values that are variations of one another
(i.e. 'feces ethnicity caucasian' vs 'feces ethnicity' vs 'feces [envo:]' vs 'feces envo:'). Each of these values
indicate that the sample came from feces. Thus, we map each of these values to 'feces'.
---------

Author: Parker Hicks
Date: 2024-03-22
"""
from argparse import ArgumentParser
from utils import check_outdir
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import re


def samples_with_keyword(
    desc: str,
    kwd: str,
) -> int:
    """Marks descriptions that contain a keyword as 1 and 0 if the word is not present."""
    cond = 0
    if kwd in desc:
        cond = 1
    return cond


def split_by_underscore(desc: str) -> np.array:
    """Splits a string into a list of elements that contain underscores."""
    split_desc = desc.split(" ")
    mask = list()
    for i, word in enumerate(split_desc):
        if "_" in word:
            mask.append(i)

    return np.array(split_desc)[mask]


def get_unique(fields_per_sample: np.array) -> set:
    print("Finding unique fields.")
    fields = list()
    for row in fields_per_sample:
        fields.extend(row)
    unique = set(fields)
    return unique


def remove_elements_with_numbers(input_set: set) -> list:
    """Removes elements that contain numerical characters from a set of strings."""
    return list(
        {
            element
            for element in input_set
            if not any(char.isdigit() for char in element)
        }
    )


def remove_numericals_from_str(input_string: str) -> str:
    """Removes numerical characters from a string."""
    result_string = re.sub(r"\d", "", input_string)
    return result_string


def split_by_field(desc: str, fields: list) -> list:
    """Splits a description into a list where each element begins with a pattern and contains all words that lead up \
     to the next pattern.

     :param desc: a sample description
     :param fields: metadata fields to split the description by
     :return: description split into ['field1 values', 'field2 values', ..., 'fieldn values']
     """
    regex = r"(.)(?={})".format("|".join(re.escape(field) for field in fields))
    split_desc = re.split(regex, desc)
    split_desc_filled = [
        "".join(x)
        for x in itertools.zip_longest(split_desc[::2], split_desc[1::2], fillvalue="")
    ]

    return split_desc_filled


def get_items_after_substring(desc_split: list, substring: str):
    """Each metadata field for each sample is split as follows: ['field1 values', 'field2 values', ..., 'fieldn
    values']. This function extracts values for a given metadata field.

    :param desc_split: a description for a single sample split as ['field1 values', ..., 'fieldn values']
    :param substring: the name of a metadata field to extract values for
    """
    desc_split = np.array(desc_split)
    arr_mask = np.where(np.char.find(desc_split, substring) != -1)[0]
    if arr_mask.shape[0] == 0:
        return ""  # Field not present in the description
    desc_at_substring = desc_split[arr_mask].item()
    index = desc_at_substring.find(substring)
    if index == -1:
        return None  # Substring not found
    else:
        return " ".join(desc_at_substring[index + len(substring) :].split()).strip()


def merge_na(val: str, na_values: list) -> str:
    """NaN values are described by multiple entities. This function sets any values in na_values to 'Not collected'.

    :param val: the value of a particular metadata field
    :param na_values: entities that describe Nan (i.e. nan, not collected, N/A, etc.)
    """
    if val in na_values:
        return "Not collected"
    else:
        return val


def merge_by_keywords(val: str, kwd: str) -> str:
    """Assigns a word/phrase to common term if the word is in the term.

    :param val: a word or phrase
    :param kwd: the common term found between words or phrases
    :return: the common term if found in val, else val
    """
    match = re.search(kwd, val)
    if match:
        return kwd
    else:
        return val


def main(
    metadata_file: str,
    myFields_file: str,
    outdir: str,
):
    # Set paths
    meta_path = Path(metadata_file)
    myFields_path = Path(myFields_file)
    outdir = check_outdir(outdir)

    # Load data
    metadata_df = pd.read_csv(meta_path, sep="\t")
    myFields = np.loadtxt(myFields_path, dtype=str)

    # Extract potential metadata fields
    metadata_df["fields"] = metadata_df["metadatablob"].apply(split_by_underscore)
    fields_per_sample = metadata_df["fields"].to_numpy()

    # Get set of unique metadata fields
    print("Finding unique fields.")
    fields = get_unique(fields_per_sample)
    fields_num_rm = remove_elements_with_numbers(fields)

    # Collect all potential fields
    if args.sf:
        np.savetxt(outdir / "potential_fields.txt", fields_num_rm, fmt="%s")

    # Split the metadata into list of fields
    print("Splitting descriptions.")
    splitters = fields_num_rm
    metadata_df["meta_split"] = metadata_df["metadatablob"].apply(
        split_by_field, fields=splitters
    )

    # Extract values from specified fields
    for field in myFields:
        print(f"\nExtracting {field}.")
        metadata_df[field] = metadata_df["meta_split"].apply(
            get_items_after_substring, substring=field
        )

        # See Note 3
        if field in ["env_medium", "isolation_source"]:
            metadata_df[field] = metadata_df[field].apply(remove_numericals_from_str)

            kwds = [
                "feces",
                "stool",
                "fecal",
                "not applicable",
                "rectal swab",
                "hrc",
            ]  # The most informative keywords determined via manual inspection
            for kwd in kwds:
                metadata_df[field] = metadata_df[field].apply(
                    merge_by_keywords, kwd=kwd
                )

        # Change protocols where values in na_values to 'Not collected'
        na_values = [
            " ",
            "",
            "not applicable",
            "not collected",
            "none",
            "missing",
            "n/a",
            "na",
        ]
        metadata_df[f"{field}_naMerged"] = metadata_df[field].apply(
            merge_na, na_values=na_values
        )

        # Filter for unique extraction protocols for project ids
        study_df = metadata_df[["project", f"{field}_naMerged"]].drop_duplicates()

        # Get unique protocol counts and save to tsv
        print(f"Saving file for {field}.")
        sample_outfile = outdir / f"sample_{field}.tsv"
        study_outfile = outdir / f"study_{field}.tsv"

        sample_protocol_counts = metadata_df[f"{field}_naMerged"].value_counts()
        sample_protocol_counts.to_csv(sample_outfile, sep="\t")

        study_protocol_counts = study_df[f"{field}_naMerged"].value_counts()
        study_protocol_counts.to_csv(study_outfile, sep="\t")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-metadata",
        help="/path/to/metadata as a tab separated txt or tsv file with header names",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-myFields",
        help="/path/to/file.txt containing fields for which to extract values from the metadata",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-outdir",
        help="Path to save dataframes in",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--sf",
        help="Use to save the extracted fields as a txt file",
        action="store_true",
    )
    args = parser.parse_args()

    main(
        args.metadata,
        args.myFields,
        args.outdir,
    )
