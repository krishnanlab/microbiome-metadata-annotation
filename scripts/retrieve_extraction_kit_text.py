import os
import re
import csv
import xml.etree.ElementTree as ET

# path to the xml and output file
folder_path = '../data/pubtator_fulltext'
output_dir = '../data/'
output_file = os.path.join(output_dir, 'extraction_kit_description.tsv')

# Define the words to search for
required_word = re.compile(r'\bkit(s)?\b', re.IGNORECASE)
additional_words = re.compile(r'\b(dna |genome|extraction)\b', re.IGNORECASE)

# Function to extract relevant sentences from text
def extract_relevant_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    relevant_sentences = [
        sentence for sentence in sentences
        if required_word.search(sentence) and additional_words.search(sentence)
    ]
    return relevant_sentences

# Function to extract passage text
def extract_passage_text(passage):
    text_element = passage.find('text')
    if text_element is not None:
        passage_text = text_element.text
        if passage_text and not passage_text.endswith('.'):
            passage_text += '.'  # Ensure a period at the end to avoid concatenating redundant sentences
        return passage_text
    return ''

# Prepare the output file
os.makedirs(output_dir, exist_ok=True)

# Write the result to a TSV output file
with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(['PMID', 'Extracted Sentences'])  # Write header

    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            pmid = filename.split('.')[0]  # Extract PMID from filename
            file_path = os.path.join(folder_path, filename)
            try:
                # Load and parse the XML file
                tree = ET.parse(file_path)
                root = tree.getroot()

                extracted_texts = []

                # Iterate through documents
                for document in root.findall('document'):
                    # Extract and print each passage
                    for passage in document.findall('passage'):
                        passage_text = extract_passage_text(passage)

                        # Extract relevant sentences
                        relevant_sentences = extract_relevant_sentences(passage_text)

                        if relevant_sentences:
                            extracted_texts.extend(relevant_sentences)

                # Prepare the output
                if extracted_texts:
                    extracted_text = ' '.join(extracted_texts)
                else:
                    extracted_text = 'no description found'

                # Write to TSV file
                writer.writerow([pmid, extracted_text])
                print(f"Processed {pmid}: {extracted_text}")

            except ET.ParseError as e:
                print(f"Error parsing {filename}: {e}")
                writer.writerow([pmid, 'error parsing file'])

print("Processing complete. Results saved to:", output_file)

