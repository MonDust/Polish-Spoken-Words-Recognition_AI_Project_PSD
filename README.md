# AI_Project_PSD
Artificial neural network classifier trained by backpropagation for polish spoken words recognition (PSD dataset).

Links:
- Shared information: https://pgedupl-my.sharepoint.com/:w:/g/personal/s193507_student_pg_edu_pl/ERdnZW877kRHh-C_ZeyAGzwB1uRteGOSYwk5hAic-0g5Tw?e=NQnaa8
- dataset: https://drive.google.com/drive/folders/1rypKZAU2qBCL2A7KVw71hSc7J6Y17PZl?usp=sharing
- extracted words: https://drive.google.com/drive/folders/1AustmyTC9NP_kj5B3plggsSKEe5uiJug?usp=drive_link

Files:
- words_final.txt - all words after extracting them and not performing additional actions

Notes:
- 50/50 (pula do szkolenia/pula do sprawdzania) + cross validacja,
- splotowa + rekurencujna najlepiej(?) - 25 ms + przesuwanie o 10 ms,
- jeśli pojedyńcza (nie rekurencujna) - stała wielkość
- najlepiej wyszukać słów wcześniej

Not used:
- Google collab - words: https://colab.research.google.com/drive/1044ulkEg6t_PzotxHb7TdyOdJJ2BQF94?usp=sharing
- [OLD] Google collab - sentences: https://colab.research.google.com/drive/1vee1-ccTDdoTZCsFuRu4I3Dwd0E3wGWW?usp=sharing

[OLD] Google collab - sentences: messy, doesn't work, is build weirdly.
Google collab - words: starting from the start, building for only words with already splitted files into words.

Old files:
- words.txt - all words in the dataset PSD,
- getting_words_from_dataset.py - script to get all the words from dataset,
- split2.py - splitting the dataset into testing and training (getting the names into csv files)
- csv files - the dataset split into testing and training
- find_words.py - find final words (after extracting the words from files)
- words_to_folders.py - script used for extracting words into folders

## Authors
O. Paszkiewicz (MonDust) - [GitHub](https://github.com/MonDust)
Kaosek - [GitHub](https://github.com/Kaosek)
