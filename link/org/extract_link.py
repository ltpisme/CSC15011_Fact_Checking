import json

# Load the original data
with open('org_TG.json', 'r', encoding='utf-8') as file:
    data_list = json.load(file)

# Open a new file in 'write' mode ('w')
with open('org_TG_link.txt', 'w', encoding='utf-8') as output_file:
    for item in data_list:
        if "url" in item:
            # Write the URL followed by a newline character
            output_file.write(item["url"] + "\n")

print("Successfully saved links to links_output.txt")