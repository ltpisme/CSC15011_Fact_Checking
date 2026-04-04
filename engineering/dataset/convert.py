import json

def convert_json(input_file, output_file):
    try:
        # Load the original JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_data = []

        for entry in data:
            # Create the new dictionary structure
            if entry.get("status") == "HIGH_CONFIDENCE":
                new_entry = {
                    "claim": entry.get("claim"),
                    "label": entry.get("final_label"), # Renamed from final_label
                    #"status": entry.get("status"),
                    # Extract only the 'text' field from each evidence object
                    "contexts": [ev.get("text") for ev in entry.get("evidence", [])]
                }
                new_data.append(new_entry)

        # Write the simplified data to a new file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully converted data to {output_file}")

    except FileNotFoundError:
        print("Error: The input file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
if __name__ == "__main__":
    # Replace 'input.json' and 'output.json' with your actual filenames
    convert_json('OG_output_dataset.json', 'HIGH_CONFIDENCE_converted_dataset.json')