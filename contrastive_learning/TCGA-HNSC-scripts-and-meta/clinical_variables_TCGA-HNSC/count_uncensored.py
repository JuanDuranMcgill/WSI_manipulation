import pandas as pd

# Load the data
df = pd.read_csv('clinical.tsv', sep='\t')

# Initialize the output DataFrame
output_df = pd.DataFrame()

# Copy the case_id column
output_df['case_id'] = df['case_id']

# Create the censoring column
output_df['censoring'] = df['days_to_death'].apply(lambda x: '1' if x != "'--" else '0')

# Create the survival_time column
def get_survival_time(value):
    if value != "'--":
        return int(value)
    else:
        return 0

output_df['survival_time'] = df['days_to_death'].apply(get_survival_time)

# Save the output DataFrame to a TSV file
output_df.to_csv('TCGA-HNSC-SURVIVAL-DATA.tsv', sep='\t', index=False)
