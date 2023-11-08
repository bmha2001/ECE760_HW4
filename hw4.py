import os
import math
from collections import defaultdict

# Step 1: Load and preprocess the dataset
data_path = "languageID"  # Adjust the path to the dataset folder
training_files = {"e": [], "j": [], "s": []}
vocabulary = set("abcdefghijklmnopqrstuvwxyz ")  # 27 character types

for root, dirs, files in os.walk(data_path):
    for filename in files:
        lang = filename[0]
        num = int(filename.replace("e", "").replace(".txt", "").replace("j", "").replace("s", ""))
        if lang in training_files and num < 10:
            with open(os.path.join(root, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                training_files[lang].append(text)

# Step 2: Estimate prior probabilities with additive smoothing (parameter 1/2)
num_classes = len(training_files)

total_training_docs = sum(len(training_files[lang]) for lang in training_files)

prior_probabilities = {lang: math.log(len(training_files[lang]) + 0.5) / math.log(total_training_docs + num_classes * 0.5) for lang in training_files}
for lang, prior_prob in prior_probabilities.items():
    print(f'P(y={lang}) = {prior_prob:.4f}')

# Step 4: Implement the Naive Bayes classifier
conditional_probabilities = defaultdict(lambda: defaultdict(float))

# Initialize variables
smooth_param = 0.5  # Additive smoothing parameter
english_counts = defaultdict(int)

# Calculate counts of characters in English documents
for doc in training_files['e']:
    for char in doc:
        if char in vocabulary:
            english_counts[char] += 1

# Calculate the class conditional probabilities for English
theta_e = [0] * len(vocabulary)
for i, char in enumerate(vocabulary):
    theta_e[i] = (english_counts[char] + smooth_param) / (sum(english_counts.values()) + smooth_param * len(vocabulary))

# Print θe (the class conditional probabilities for English)
print("θe:", theta_e)




