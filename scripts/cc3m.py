import os
import json
from tqdm import tqdm
# Read Train_GCC-training.tsv
with open('data/T-X_pair_data/cc3m/GCC-training.tsv', 'r') as f:
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
with open('data/T-X_pair_data/cc3m/cc3m.json', 'a') as f:
    for d in tqdm(data):
        json.dump(d, f, indent=4)