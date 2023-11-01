import json
import csv

csv_path = "/home/max/datasets/hse_lab_nlp/train.csv"
json_path = "/home/max/datasets/hse_lab_nlp/train.json"

result = []

with open(csv_path, newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')

    for row in reader:
        result.append({"label": int(row[0]) - 1, "text": ' '.join([row[1], row[2]])})

with open(json_path, 'w') as f:
    json.dump(result, f, indent=4)
