#!/bin/bash

# Output directory
output_dir="../data/pubtator_fulltext/"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# List of PMIDs from the third column of the CSV file
pmids=($(awk -F',' '{print $3}' ../data/project_pmids.csv | tail -n +2))  # Skip header

# Total number of PMIDs
total_pmids=${#pmids[@]}
echo "Total PMIDs: $total_pmids"
for pmid in "${pmids[@]}"; do
    echo "Downloading data for PMID: $pmid"
    # Check if PMID is not empty
    if [ -n "$pmid" ]; then
        # Print the current iteration's PMID
        echo "Current PMID: $pmid"

        # Run wget command for each PMID
        wget "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids=$pmid&full=true" -O "$output_dir/$pmid.xml"
    fi
done

echo "Download completed. Files are saved in '$output_dir' directory."
