# microbiome-annotation
Training text-based ML models to predict protocols for microbiome samples

## Project notes from collaborators
Thanks again for meeting with us yesterday. The data file we discussed is attached. The columns, separated by tabs:
sample – the BioSample accession for the sample described in that row
project – the accession of the BioProject associated with the sample
instrument – the sequencing instrument field reported to BioSample
pubdate – the date the data was released
amplicon – the inferred hypervariable region sequenced for the project
avglength – The mean length of all ASVs identified in the project
metadatablob – a string of all the BioSample's metadata key/value pairs concatenated together. Each value follows its respective key, separated by a space.
Each key/value pair is also separated by a space, but I can change this to another character if it would be helpful.
 
We do still have the other fields I mentioned—library source, library strategy, etc.—but I forgot that we used those for filtering which samples to process, so those values are all identical.
 
You also asked which factors we would want to hunt for. After talking it over again, we only really have two specific items:
extraction kit – This one was mentioned by reviewers. Tags like "samp_mat_process" do have some of this available, with values like "mo-bio power soil kit" and "dneasy powerlyzer powersoil kit (qiagen)".
collection method – Probably lower priority than the one mentioned by the reviewers, but this factor might have more impact on composition. This is where the microbiota were obtained from—values like "fecal" are probably the most common, followed by "swab" and "biopsy."
Tags that may have the most information on this are "samp_collect_device", "env_medium", "isolation_source", and "collection_method".
 
Eventually, it would also be very helpful to try extracting host phenotypic information—in that case, even labeling the training data with a harmonized field for "age" or "sex" would be very useful even without inferring anything for unlabeled samples. A few ideas:
age – available from tags such as "host_age", "age", "age (months)", etc. Inferring broad categories here ("adult", "juvenile") could be useful too, or even just a flag for "infant" vs "non-infant".
sex – available via "sex", "host_sex", "gender_cat", etc.
study group – we have a few different ideas for how to label a "case" and a "control" in each project, and tags like "control", "study_arm" and "treatmentgroup" do have some of this already.
Ideally, one day we hope to label samples with more specific information about disease status—whether the host has diabetes/cancer/IBS and so on.


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