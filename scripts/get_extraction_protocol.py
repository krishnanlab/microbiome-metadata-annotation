from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import re


def df_from_txt(_file: Path) -> pd.DataFrame:
    """Opens a tab-separated txt file as a pandas dataframe"""
    lines = list()
    with open(_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            split_line = line.strip().split('\t')
            if i == 0:
                col_names = split_line
            else:
                lines.append(split_line)
    df = pd.DataFrame(lines, columns=col_names)

    return df


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
    split_desc = desc.split(' ')
    mask = list()
    for i, word in enumerate(split_desc):
        if '_' in word:
            mask.append(i)

    return np.array(split_desc)[mask]


def remove_elements_with_numbers(input_set):
    """Removes elements that contain numerical characters from a set of strings"""
    return {element for element in input_set if not any(char.isdigit() for char in element)}


def remove_numericals_from_str(input_string):
    # Use a regular expression to remove numerical characters
    result_string = re.sub(r'\d', '', input_string)
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
        "".join(x) for x in itertools.zip_longest(
            split_desc[::2],
            split_desc[1::2],
            fillvalue=''
        )
    ]

    return split_desc_filled


def get_items_after_substring(desc_split: list, substring: str):
    """Extracts """
    desc_split = np.array(desc_split)
    arr_mask = np.where(np.char.find(desc_split, substring) != -1)[0]
    if arr_mask.shape[0] == 0:
        return ''
    desc_at_substring = desc_split[arr_mask].item()
    index = desc_at_substring.find(substring)
    if index == -1:
        return None  # Substring not found
    else:
        return " ".join(desc_at_substring[index + len(substring):].split()).strip()


def merge_na(protocol: str, na_values: list) -> str:
    if protocol in na_values:
        return "Not collected"
    else:
        return protocol


def merge_by_keywords(protocol: str, keyword: str) -> str:
    pattern = keyword
    match = re.search(pattern, protocol)
    if match:
        return keyword
    else:
        return protocol


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-metadata',
        help='/path/to/metadata as a tab separated txt or tsv file with header names',
        required=True,
        type=str,
    )
    parser.add_argument(
        '-outdir',
        help="Path to save dataframes in",
        required=True,
        type=str,
    )
    parser.add_argument(
        '--sf',
        help="Use to save the extracted fields as a txt file",
        action='store_true',
    )
    args = parser.parse_args()

    # Set outdir
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # Load data
    meta_path = Path(args.metadata)
    metadata_df = df_from_txt(meta_path)

    # Extract potential metadata fields
    metadata_df["fields"] = metadata_df['metadatablob'].apply(split_by_underscore)
    fields_per_sample = metadata_df["fields"].to_numpy()

    # Get set of unique metadata fields
    print("Finding unique fields.")
    fields = list()
    for row in fields_per_sample:
        fields.extend(row)
    fields = set(fields)
    fields_num_rm = remove_elements_with_numbers(fields)

    # Collect all potential fields
    if args.sf:
        with open(outdir / 'potential_fields.txt', 'w') as f:
            for field in fields_num_rm:
                f.write(f'{field}\n')

    # Split the metadata into list of fields
    print("Splitting descriptions.")
    splitters = list(fields_num_rm)
    metadata_df["meta_split"] = metadata_df["metadatablob"].apply(
        split_by_field,
        fields=splitters
    )

    keys_to_look_for = [
        "samp_mat_process",
        "samp_collect_device",
        "env_medium",
        "isolation_source",
        "collection_method"
    ]

    for key in keys_to_look_for:
        print(f"Extracting {key}.")
        metadata_df[key] = metadata_df["meta_split"].apply(
            get_items_after_substring,
            substring=key
        )

        # Change protocols where values in na_values to 'Not collected'
        na_values = [" ", "", "not applicable", "not collected", "none", "missing", 'n/a', 'na']
        metadata_df[f"{key}_naMerged"] = metadata_df[key].apply(
            merge_na,
            na_values=na_values
        )

        if key in ['env_medium', 'isolation_source']:
            metadata_df[f'{key}_naMerged'] = metadata_df[f'{key}_naMerged'].apply(remove_numericals_from_str)

            kwds = ['feces ethnicity', 'stool ethnicity', 'fecal material', 'not applicable', 'rectal swab', 'hrc']
            for kwd in kwds:
                metadata_df[f'{key}_naMerged'] = metadata_df[f'{key}_naMerged'].apply(merge_by_keywords, keyword=kwd)

        study_df = metadata_df[["project", f"{key}_naMerged"]].drop_duplicates()

        # Get unique protocol counts and save to tsv
        print(f"Saving file for {key}.")
        sample_outfile = outdir / f'sample_{key}_counts.tsv'
        study_outfile = f'study_{key}_counts.tsv'

        sample_protocol_counts = metadata_df[f"{key}_naMerged"].value_counts()
        sample_protocol_counts.to_csv(sample_outfile, sep='\t')

        study_protocol_counts = study_df[f"{key}_naMerged"].value_counts()
        study_protocol_counts.to_csv(study_outfile, sep='\t')
