import os
import random
import csv

words_folder = "words"

split_ratio = 0.5

train_files = []
test_files = []

for word_folder in os.listdir(words_folder):
    word_path = os.path.join(words_folder, word_folder)

    files = os.listdir(word_path)
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)

    train_files.extend([(os.path.join(word_folder, file), word_folder) for file in files[:split_index]])
    test_files.extend([(os.path.join(word_folder, file), word_folder) for file in files[split_index:]])

with open("train_files.csv", "w", newline="", encoding="utf-8") as train_csv:
    writer = csv.writer(train_csv)
    writer.writerow(["filename", "folder"])
    for filename, folder in train_files:
        writer.writerow([filename, folder])

with open("test_files.csv", "w", newline="", encoding="utf-8") as test_csv:
    writer = csv.writer(test_csv)
    writer.writerow(["filename", "folder"])
    for filename, folder in test_files:
        writer.writerow([filename, folder])

print("CSV files created successfully.")
