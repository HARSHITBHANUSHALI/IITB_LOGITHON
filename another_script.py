import re

def extract_mapping_from_file(file_path):
    # Dictionary to store source -> destination mappings
    mappings = {}
    
    try:
        # Read the text file
        with open(file_path, 'r') as file:
            content = file.read()
            
            # Regular expression to match filenames like "country1_to_country2.json"
            pattern = r'([a-zA-Z]+)_to_([a-zA-Z]+)\.json'
            
            # Find all matches in the content
            matches = re.findall(pattern, content)
            
            # Process each match
            for source, destination in matches:
                # Convert to lowercase for consistency
                source = source.lower()
                destination = destination.lower()
                
                # If source already exists, append to its list of destinations
                if source in mappings:
                    if destination not in mappings[source]:
                        mappings[source].append(destination)
                else:
                    mappings[source] = [destination]
                    
        return mappings
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return {}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {}

def print_mappings(mappings):
    # Pretty print the mappings
    print("\nSource to Destination Mappings:")
    print("-----------------------------")
    for source, destinations in mappings.items():
        print(f"{source} → {', '.join(destinations)}")

# Example usage
if __name__ == "__main__":
    # Assuming your text file is named 'shipping_info.txt'
    file_path = "section_headers.txt"
    
    # Extract mappings
    mappings = extract_mapping_from_file(file_path)
    
    # Print results
    print_mappings(mappings)
    
    # Optional: Print in dictionary format
    print("\nRaw Dictionary:")
    print(mappings)
