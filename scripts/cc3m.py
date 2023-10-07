import os
import json

# Read Train_GCC-training.tsv
with open('../data/T-X_pair_data/cc3m/GCC-training.tsv', 'r') as f:
    lines = f.readlines()

# List image files
image_files = sorted([f for f in os.listdir() if f.endswith('.jpg')])

data = []

for line, image_file in zip(lines, image_files):
    url, caption = line.strip().split('\t')
    data.append({
        "caption": caption,
        "image_name": image_file
    })

# Save to cc3m.json
with open('../data/T-X_pair_data/cc3m/cc3m.json', 'w') as f:
    json.dump(data, f, indent=4)