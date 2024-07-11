import json
import csv

# Function to read JSON file from local directory
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to extract threat information and save it to CSV
def extract_threat_info(json_data, csv_file):
    # Open the CSV file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write the header
        writer.writerow(['threat_name', 'threat_description', 'threat_type'])
        
        # Extract and write the data
        for item in json_data['objects']:
            threat_name = item.get('name', '').strip()
            threat_description = item.get('description', '').strip()
            threat_type = item.get('type', '').strip()
            
            # Check if any field is empty, skip the row if true
            if not threat_name or not threat_description or not threat_type:
                continue
            
            writer.writerow([threat_name, threat_description, threat_type])

# Example usage
json_file_path = r'D:\transformer\datasets\mobile-attack.json'  # Path to your local JSON file
csv_file = 'cleandata.csv'

# Read JSON data from local file
json_data = read_json_file(json_file_path)

# Extract threat information and save to CSV
extract_threat_info(json_data, csv_file)

print(f"CSV file '{csv_file}' created successfully.")
