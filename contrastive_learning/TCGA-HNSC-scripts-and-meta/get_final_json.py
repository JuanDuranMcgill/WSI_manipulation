import json

# Load the input JSON file
input_file = 'svs_patient_map_survival_2.json'
output_file = 'svs_patient_map_survival_complete.json'

# Read the input JSON data
with open(input_file, 'r') as file:
    data = json.load(file)

# Process the data
for entry in data:
    days_to_death = entry['days_to_death']
    
    if days_to_death == "'--":
        entry['time'] = 0
        entry['death'] = 0
    else:
        entry['time'] = int(days_to_death)
        entry['death'] = 1
    
    if entry['days_to_last_follow_up'] == "'--":
        entry['days_to_last_follow_up'] = ""
    else:
        entry['days_to_last_follow_up'] = int(float(entry['days_to_last_follow_up']))

    # Remove the 'days_to_death' field
    del entry['days_to_death']

# Write the modified data to the output JSON file
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Processed data saved to {output_file}")
