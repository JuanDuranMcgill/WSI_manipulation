import pandas as pd

def extract_survival_data(input_file, output_file):
    # Load the TSV file into a DataFrame
    df = pd.read_csv(input_file, sep='\t')
    
    # Extract the specified columns
    columns_to_extract = ['case_id', 'days_to_death', 'days_to_last_follow_up']
    
    # Check if all required columns are present
    missing_columns = [col for col in columns_to_extract if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return
    
    # Extract the required columns
    df_extracted = df[columns_to_extract]
    
    # Save the extracted data to a new TSV file
    df_extracted.to_csv(output_file, sep='\t', index=False)
    print(f"Data saved to {output_file}")

# Example usage:
input_file = 'clinical.tsv'  # Replace with your actual input file path
output_file = 'clinical_survival.tsv'
extract_survival_data(input_file, output_file)
