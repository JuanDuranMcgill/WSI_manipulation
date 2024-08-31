import os
import json
import pandas as pd

# File paths
json_file_path = 'svs_patient_map_DFS.json'
embeddings_folder = 'TCGA-HNSC-embeddings-flatten'
clinical_file_path = 'clinical.tsv'

# Load the JSON metadata
with open(json_file_path, 'r') as f:
    metadata = json.load(f)

# Load the clinical.tsv file
clinical_data = pd.read_csv(clinical_file_path, sep='\t')

# Get the list of case_ids from the clinical data
clinical_case_ids = set(clinical_data['case_id'].str.lower())

# Prepare to count the matches
count_json_matches = 0
count_clinical_matches = 0
unique_clinical_matches = set()

for entry in metadata:
    if pd.isnull(entry['censoring']) or pd.isnull(entry['time_to_event']):
        continue

    # Replace .svs with _flatten.pt in the file name
    file_name = entry['file_name'].replace('.svs', '_flatten.pt')
    file_path = os.path.join(embeddings_folder, file_name)
    
    if os.path.exists(file_path):  # Check if file exists
        count_json_matches += 1
        # Check if the case_id is also in clinical data
        if entry['case_id'].lower() in clinical_case_ids:
            count_clinical_matches += 1
            unique_clinical_matches.add(entry['case_id'].lower())

print("Total number of matching files from JSON metadata:", count_json_matches)
total_files_in_embeddings_folder = len(os.listdir(embeddings_folder))
print("Total number of files in embeddings_folder:", total_files_in_embeddings_folder)
print("Total number of matching case_ids in clinical.tsv and embeddings folder:", count_clinical_matches)
print("Total number of unique matching case_ids in clinical.tsv and embeddings folder:", len(unique_clinical_matches))
total_entries_in_clinical = len(clinical_data)
print("Total number of entries in clinical.tsv:", total_entries_in_clinical)

print("@@@@@@")

# Define file paths
embeddings_folder = 'TCGA-HNSC-embeddings-flatten'
final_patients_file = 'final_patients.txt'

# Load the list of filenames from final_patients.txt
with open(final_patients_file, 'r') as f:
    patient_filenames = [filename.lower() for filename in f.read().splitlines()]

# Get a list of all files in the embeddings folder, in lowercase
embeddings_files = {filename.lower() for filename in os.listdir(embeddings_folder)}

# Initialize counters
total_files = len(patient_filenames)
matches = 0
missing_files = []

# Check for matches in the embeddings folder
for filename in patient_filenames:
    if filename in embeddings_files:
        matches += 1
    else:
        missing_files.append(filename)

# Print results
print(f"Total number of files in final_patients.txt: {total_files}")
print(f"Total number of matching files in the embeddings folder: {matches}")
print(f"Total number of missing files: {total_files - matches}")
print("Total number of files in embeddings_folder:", total_files_in_embeddings_folder)
