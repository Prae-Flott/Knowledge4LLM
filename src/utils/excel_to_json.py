import pandas as pd

# Read the Excel file. Replace 'your_file.xlsx' with your Excel file name.
df = pd.read_excel('./labeled_data/battery_status.xlsx')  # adjust sheet_name if needed

# Convert the DataFrame to JSON format
# The 'orient' parameter can be adjusted according to your needs.
json_data = df.to_json(orient='records', force_ascii=False, indent=4)

# Save the JSON output to a file
with open('./labeled_data/battery_status.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

print("Conversion complete! JSON data saved to output.json")
