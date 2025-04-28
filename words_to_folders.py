# -*- coding: utf-8 -*-
"""
Artificial neural network classifier trained by backpropagation for polish spoken words recognition (PSD dataset)
-----
"""

"""1.Getting ready to train the model
-------
All the functions and variables needed.

1.1 Importing.
"""
import os
import string

from pydub import AudioSegment

"""1.2 File paths."""

# Putting here the folder path where the database is located.
folder_path = " "
#folder_path = "/content/drive/MyDrive/all_data/1-500/105"
#word_list_path = '/content/drive/MyDrive/words.txt'

"""1.21 Clean the text from punctuation."""

def clean_text(text):
  ''' Cleaning the text (in the form of str) from punctuation '''
  words = text.lower().split(' ')
  cleaned_words = []
  for word in words:
    while word[-1] in string.punctuation:
      word = word[:-1]
      if len(word) == 0: break
    if len(word) > 0: cleaned_words.append(word)
  return cleaned_words

def clean_word(word):
    ''' Cleaning the text (in the form of str) from punctuation '''
    cleaned_word = ''
    for char in word:
        if char not in string.punctuation:
            cleaned_word += char
    return cleaned_word

"""1.3 Getting TexGrid file timestamps."""

def load_timestamps(textgrid_file):
  ''' Load timestamps '''
  timestamps = []
  text = []

  with open(textgrid_file, 'r', encoding='utf-8') as txtgrid_file:
    lines = txtgrid_file.readlines()
    line_check = 0
    for line in lines:
      if line.strip().startswith('intervals:'):
        line_check = 1
      if line.strip() == 'name = "phones"':
        return timestamps, text
      if line_check == 1:
        if line.strip().startswith('xmin'):
          xmin = float(line.split('=')[-1].strip())
        if line.strip().startswith('xmax'):
          xmax = float(line.split('=')[-1].strip())
          timestamps.append((xmin, xmax))
        if line.strip().startswith('text'):
          txt_ = line.split('=')[-1].strip().strip('"')
          text.append(txt_)
  return timestamps, text

def add_text_to_timestamps(timestamps, text):
  aligned_text = []
  for word, (start_time, end_time) in zip(text, timestamps):
    aligned_text.append((word, start_time, end_time))
  return aligned_text

def load_texgrid_transcription(textgrid_file):
  timestamps, text = load_timestamps(textgrid_file)
  aligned_text = add_text_to_timestamps(timestamps, text)
  return aligned_text

# aligned_text = load_texgrid_transcription('/content/drive/MyDrive/all_data/1-500/105/100005_2.TextGrid')
# print(aligned_text)

"""1.4 Separating files (sentences) into words."""

def get_next_file_number(directory):
    files = os.listdir(directory)
    numbers = []
    for file in files:
        parts = file.split('_')
        if len(parts) >= 2:
            number_part = parts[1].split('.')[0]
            if number_part.isdigit():
                numbers.append(int(number_part))
    max_number = max(numbers) if numbers else 0
    return max_number + 1

def split_audio_by_words(wav_file, transcription):

  # Putting here the folder where words are located.
  words_dir = ""
  os.makedirs(words_dir, exist_ok=True)
  audio = AudioSegment.from_wav(wav_file)

  for i, (word, start, end) in enumerate(transcription):
    if not word:
      continue
    word = clean_word(word)
    if word.isdigit():
      continue
    if not word:
      continue

    word_audio = audio[int(start * 1000):int(end * 1000)]

    word_dir = os.path.join(words_dir, word)
    os.makedirs(word_dir, exist_ok=True)

    file_number = get_next_file_number(word_dir)

    word_file = os.path.join(word_dir, f"{word}_{file_number}.wav")
    word_audio.export(word_file, format="wav")

"""1.5 Split all audio files."""

def split_all_audio_files(folder_path):
  for root, dirs, files in os.walk(folder_path):
    for file in files:
      if file.endswith(".TextGrid"):
        textgrid_file = os.path.join(root, file)
        transcription = load_texgrid_transcription(textgrid_file)
        wav_file = textgrid_file.replace('.TextGrid', '.wav')
        split_audio_by_words(wav_file,transcription)
        print("Split: ", file)

split_all_audio_files(folder_path)
