import json
import pandas as pd

def merge_survival_data(json_file, tsv_file, output_file):
    # Load the TSV file into a DataFrame
    survival_df = pd.read_csv(tsv_file, sep='\t')
    
    # Handle duplicates by keeping the first occurrence
    survival_df = survival_df.drop_duplicates(subset=['case_id'])
    
    # Load the JSON file
    with open(json_file, 'r') as jf:
        svs_data = json.load(jf)
    
    # Create a dictionary for quick lookup of survival data by case_id
    survival_dict = survival_df.set_index('case_id').to_dict('index')
    
    # Add survival data to the JSON data
    for record in svs_data:
        case_id = record['case_id']
        if case_id in survival_dict:
            record['days_to_death'] = survival_dict[case_id].get('days_to_death')
            record['days_to_last_follow_up'] = survival_dict[case_id].get('days_to_last_follow_up')
        else:
            record['days_to_death'] = None
            record['days_to_last_follow_up'] = None
    
    # Save the updated JSON data to the new file
    with open(output_file, 'w') as of:
        json.dump(svs_data, of, indent=4)

# Example usage:
json_file = 'svs_patient_map.json'  # Replace with your actual JSON file path
tsv_file = 'clinical_survival.tsv'  # Replace with your actual TSV file path
output_file = 'svs_patient_map_survival_2.json'
merge_survival_data(json_file, tsv_file, output_file)

