# microbiome-annotation
This repository includes scripts for performing semantic search to identify and annotate text snippets (from PubMed manuscripts) with a predefined set of terms (in this case, microbiome extraction kits).

## Method
We developed a semantic matching approach to identify the presence of the names of extraction kits within these descriptions. Given an extraction kit name (with n words) and a study description (from the literature; with m words), for each kit word ei in {e1, e2, ..., en}, we recorded its similarity to the closest study word among {d1, d2, ..., dm}. Similarly, for each study word dj, we recorded its similarity to the closest kit word. Similarity between a pair of words was defined as the cosine similarity between their word embeddings generated using BioMedBERT. The overall similarity of the kit-study pair was calculated by averaging these best word-pair similarities, weighted by each word’s “informativeness” (quantified using its term-frequency inverse document frequency; TF-IDF).  Finally, we used the Stouffer’s z-score method to correct the similarity score for each kit-study pair to account for background signals. Specifically, the corrected kit-study score is a combination of two z-scores of the original similarity score calculated based on the 𝜇 and 𝜎 of two distributions: that kit’s similarity to all studies and that study’s similarity to all kits. Finally, each study description was annotated to the extraction kit with which it had the highest corrected similarity score.
### Example
Consider an extraction kit with three words `{e1, e2, e3}` and a study description with three words `{d1, d2, d3}`. We compute the cosine similarity of the embedding vectors for each pair of words to create the following similarity matrix:

|       | **d1** | **d2** | **d3** | **max** |
|-------|--------|--------|--------|---------|
| **e1** | 0.5    | 0.4    | 0.1    | 0.5     |
| **e2** | 0.2    | 0.9    | 0.1    | 0.9     |
| **e3** | 0.7    | 0.3    | 0.7    | 0.7     |
| **max**| 0.7    | 0.9    | 0.7    |         |

To compute the overall similarity between the ontology term and the sample description, we average the maximum values from both rows and columns:

In this case scenario:
(0.7 + 0.9 + 0.7 + 0.5 + 0.9 + 0.7) / 6 = 0.73



## 0. Identify potential extraction protocols from sample metadata

```python
python get_extraction_protocol.py \
       -metadata ../data/metadata_blobs.txt \
       -myFields ../data/myFields.txt \
       -outdir ../data/counts \
       --sf
```

#### Inputs:
- metadata
  - a tab separated file with columns `sample`, `project`,`metadatablob`.
    - `sample`: a sample ID
    - `project`: a project or study ID
    - `metadatablob`: a description for a given sample ID
- myFields
  - a `.txt` file defining metadata fields that could contain extraction protocols
- outdir
- sf
  - if this flag is used, all potential metadata fields will be saved to `data/potential_fields.txt`

#### Outputs:
`outdir/<sample or study>_<field>.csv` for field in `myFields.txt`

## 1. Generating descriptions

## 2. Generation embeddings
```
python embedding_lookup_table.py\
          -outdir ../data/
```

## 3. Calculating similarities

## 4. Aggregating results
```
python aggregate_similarities.py\
       -scores_dir ../data/output\
       -transform double_z\
       -outdir ../data/
```
