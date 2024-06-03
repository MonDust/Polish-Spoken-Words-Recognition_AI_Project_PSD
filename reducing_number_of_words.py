import os

def count_files_in_directory(directory_path):
    """
    Count the number of files in each subdirectory of the given directory.

    Parameters:
    directory_path (str): The path of the main directory containing subdirectories.

    Returns:
    dict: A dictionary where keys are subdirectory names and values are the number of files in each subdirectory.
    """
    file_count = {}
    for subdir in os.listdir(directory_path):
        subdir_path = os.path.join(directory_path, subdir)
        if os.path.isdir(subdir_path):
            num_files = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
            file_count[subdir] = num_files
    return file_count

def find_words_with_more_than__files(file_count_dict):
    """
    Find all keys (subdirectory names) in the dictionary that have values greater than __.

    Parameters:
    file_count_dict (dict): A dictionary where keys are subdirectory names and values are the number of files.

    Returns:
    list: A list of subdirectory names that have more than __ files.
    """
    words = [word for word, count in file_count_dict.items() if count > 100]
    return words

def write_words_to_file(words_list, output_file_path):
    """
    Write the list of words to a text file.

    Parameters:
    words_list (list): A list of words to write to the file.
    output_file_path (str): The path of the output file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for word in words_list:
            file.write(f"{word}\n")

# Define the directory path and output file path
words_dir = "C:\\Users\\maria\\Desktop\\PG\\SEM_IV\\Artificial_Intelligence\\Project\\PSD\\words"
output_file = "C:\\Users\\maria\\Desktop\\PG\\SEM_IV\\Artificial_Intelligence\\Project\\PSD\\words_useful2.txt"

# Count files in each subdirectory
file_counts = count_files_in_directory(words_dir)

# Find words with more than 8 files
words_with_more_than__files = find_words_with_more_than__files(file_counts)

# Write the words to the output file
write_words_to_file(words_with_more_than__files, output_file)
