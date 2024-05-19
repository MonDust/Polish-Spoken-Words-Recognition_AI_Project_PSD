import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Load the Data
words_dir = r'C:\Users\maria\Desktop\PG\SEM_IV\Artificial_Intelligence\Project\PSD\words'
words = os.listdir(words_dir)[:10]  # Select 20 words
spectrograms = []
labels = []

max_length = 1000  # Choose a maximum length for spectrograms
label_encoder = LabelEncoder()

# Modify the loop to pad or truncate spectrograms
for word in words:
    word_dir = os.path.join(words_dir, word)
    wav_files = os.listdir(word_dir)
    for wav_file in wav_files:
        wav_path = os.path.join(word_dir, wav_file)
        y, sr = librosa.load(wav_path)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Pad or truncate spectrogram to max_length
        if spectrogram.shape[1] < max_length:
            pad_width = max_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        elif spectrogram.shape[1] > max_length:
            spectrogram = spectrogram[:, :max_length]

        spectrograms.append(spectrogram)
        labels.append(word)

# Convert lists to numpy arrays
spectrograms = np.array(spectrograms)
labels_encoded = label_encoder.fit_transform(labels)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels_encoded, test_size=0.5, random_state=42)

# Step 4: Build the Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(spectrogram.shape[0], spectrogram.shape[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(words), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train the Model
X_train = np.array(X_train)
X_train = X_train[..., np.newaxis]
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Step 6: Evaluate the Model
X_test = np.array(X_test)
X_test = X_test[..., np.newaxis]
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
