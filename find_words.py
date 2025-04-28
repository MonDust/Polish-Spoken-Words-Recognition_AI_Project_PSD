def save_word_folders_to_txt(words_folder, output_file="words.txt"):
    """
    Save the names of word folders in the specified directory to a text file.

    Args:
    words_folder (str): Path to the folder containing word folders.
    output_file (str): Name of the text file to save the word folder names. Default is "words.txt".
    """
    folder_names = [folder for folder in os.listdir(words_folder) if os.path.isdir(os.path.join(words_folder, folder))]

    with open(output_file, "w") as file:
        for folder_name in folder_names:
            file.write(folder_name + "\n")

words_folder = "words"
save_word_folders_to_txt(words_folder)
