import os
import re
import csv
import xml.etree.ElementTree as ET


# The directory containing the XML files
folder_path = '../data/publication_data/pubtator_fulltext'
# The directory containing output TSV file
output_dir = '../data/publication_data/'
output_file = os.path.join(output_dir, 'extraction_kit_description.tsv')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the words to search for
required_word = re.compile(r'\bkit(s)?\b', re.IGNORECASE)
additional_words = re.compile(r'\b(extraction|dna|genome)\b', re.IGNORECASE)


# Function to extract extraction kit text snippets
def extract_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant_sentences = [
        sentence for sentence in sentences
        if required_word.search(sentence) and additional_words.search(sentence)
    ]
    return relevant_sentences


# write to a tsv output file
with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(['PMID', 'Extracted Sentences'])  # Write header

    # Iterate over the XML files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            pmid = filename.split('.')[0]  # Extract PMID from filename
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                text = ''.join(root.itertext())
                sentences = extract_sentences(text)

                if sentences:
                    extracted_text = ' '.join(sentences)
                    writer.writerow([pmid, extracted_text])
                    print(f"Extracted text from {pmid} saved.")
                else:
                    print(f"No relevant sentences found in {pmid}.")

            except ET.ParseError as e:
                print(f"Error parsing {filename}: {e}")

print("Processing complete. Results saved to:", output_file)
