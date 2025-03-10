import os
import json
import re

def extract_section_headers():
    # Directory path
    dir_path = "anushka/ups_regulations"
    
    # Pattern for matching source_to_destination.json files
    pattern = re.compile(r'.*_to_.*\.json$')
    
    # Dictionary to store results: filename -> list of section headers
    results = {}
    
    # Iterate through files in the directory
    for filename in os.listdir(dir_path):
        if pattern.match(filename) and filename.endswith('.json'):
            file_path = os.path.join(dir_path, filename)
            
            try:
                # Read and parse JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract section headers
                section_headers = []
                
                # Handle both list and dictionary formats based on the structure shown
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "Section" in item:
                            section_headers.append(item["Section"])
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict) and "Section" in value:
                            section_headers.append(value["Section"])
                
                # Store results
                results[filename] = section_headers
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    # Write results to output file
    with open("section_headers.txt", "w", encoding='utf-8') as f:
        for filename, headers in results.items():
            f.write(f"File: {filename}\n")
            for header in headers:
                f.write(f"  - {header}\n")
            f.write("\n")
    
    print(f"Section headers extracted and saved to section_headers.txt")

if __name__ == "__main__":
    extract_section_headers()