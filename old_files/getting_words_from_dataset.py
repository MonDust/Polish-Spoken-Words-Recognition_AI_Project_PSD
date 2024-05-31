import os
import string

def extract_words(folder_path, output_file):
    all_words = set()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                txt_file = os.path.join(root, file)
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text_no_punctuation = text.translate(str.maketrans('', '', string.punctuation))
                    words = text_no_punctuation.split()
                    for word in words:
                        all_words.add(word.lower())
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in all_words:
            f.write(word + '\n')

folder_paths = ["C:\\Users\\..."] # path to the folder in which the txt files are located

output_file = "words.txt"
for folder_path in folder_paths:
    extract_words(folder_path, output_file)

