import json

def extract_country_names(json_file, output_file):
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract country names (keys)
    country_names = list(data.keys())
    
    # Save as an array in a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(country_names))
    
    print(f"Extracted {len(country_names)} country names and saved to {output_file}")

# Example usage
extract_country_names('fedex_country_restrictions2.json', 'countries.txt')
