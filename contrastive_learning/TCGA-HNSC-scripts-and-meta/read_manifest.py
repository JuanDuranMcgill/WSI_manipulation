import pandas as pd

# Replace with the actual path to your manifest file
# file_path = "gdc_manifest.2024-03-26.txt"
file_path = "gdc_manifest.2024-04-12.txt"

# Load the data into a Pandas DataFrame (assumes tab-separated values)
df = pd.read_csv(file_path, sep='\t')

# Calculate the total file size in bytes
total_size_bytes = df['size'].sum()

# Convert bytes to gigabytes
total_size_gb = total_size_bytes / (1024 ** 3)

print("The total size of all files is: {:.2f} GB".format(total_size_gb)) 
print("The total size of ")