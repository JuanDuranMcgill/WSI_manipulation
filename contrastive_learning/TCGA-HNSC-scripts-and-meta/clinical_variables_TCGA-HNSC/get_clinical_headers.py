import pandas as pd

def extract_non_empty_headers(file_path):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(file_path, sep='\t')
    
    # Initialize an empty list to store headers with non-empty values
    non_empty_headers = []

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if any value in the column is not '--'
        if (df[column] != "'--").any():
            non_empty_headers.append(column)
    
    # Write the list of non-empty headers to a txt file
    with open('clinical_headers_list_out.txt', 'w') as f:
        for header in non_empty_headers:
            f.write(f"{header}\n")

# Call the function with the input file name
extract_non_empty_headers('clinical.tsv')