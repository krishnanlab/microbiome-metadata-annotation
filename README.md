# microbiome-annotation
This repository includes scripts for performing semantic search to identify and annotate text snippets (from PubMed manuscripts) with a predefined set of terms (in this case, microbiome extraction kits).

## Method
we developed a semantic matching approach to compute similarity scores between each description and a list of known extraction kits. We began by generating word embeddings using PubMedBERT [cite] for both the extraction kits and descriptions. A similarity matrix was constructed where rows correspond to the words in the extraction kit and columns correspond to the words in the description. For each matrix, we calculated the maximum cosine similarity value across all rows and columns. These maximum values were then weighted by the TF-IDF scores of the corresponding words to account for the importance of each word in the context of the overall corpus, ensuring that more relevant terms had a greater impact on the similarity score. The weighted maximum values were averaged to produce a single similarity score for each extraction kit-description pair, resulting in a weighted similarity matrix. To ensure comparability across all extraction kits and descriptions, we applied z-transformation across both columns and rows of this weighted similarity matrix. Each description was then annotated with the extraction kit that had the highest normalized similarity score. To ensure accuracy, we manually reviewed the descriptions with very low similarity scores by sorting and examining the last 10-20 percent of the least similar cases.

### Example
Consider an ontology term with three words `{t1, t2, t3}` and a sample description with three words `{s1, s2, s3}`. We compute the cosine similarity of the embedding vectors for each pair of words to create the following similarity matrix:

|       | **s1** | **s2** | **s3** | **max** |
|-------|--------|--------|--------|---------|
| **t1** | 0.5    | 0.4    | 0.1    | 0.5     |
| **t2** | 0.2    | 0.9    | 0.1    | 0.9     |
| **t3** | 0.7    | 0.3    | 0.7    | 0.7     |
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
