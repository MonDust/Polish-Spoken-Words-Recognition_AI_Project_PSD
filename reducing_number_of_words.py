import os

def count_files_in_directory(directory_path):
    file_count = {}
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            num_files = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
            file_count[subdir] = num_files
    return file_count

def find_words_with_more_than__files(file_count_dict):
    words = [word for word, count in file_count_dict.items() if count > 100]
    return words

def write_words_to_file(words_list, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for word in words_list:
            file.write(f"{word}\n")

# Gett needed words - put appropriate file directory.
words_dir = "...\\words"
output_file = "...\\words_useful.txt"

file_counts = count_files_in_directory(words_dir)

words_with_more_than__files = find_words_with_more_than__files(file_counts)

write_words_to_file(words_with_more_than__files, output_file)
