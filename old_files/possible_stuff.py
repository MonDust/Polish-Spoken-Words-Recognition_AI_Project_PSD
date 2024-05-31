import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Function to load and preprocess audio files
def load_and_preprocess_audio(file_path, max_length=1000, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate spectrogram to max_length
    if mel_spec.shape[1] < max_length:
        pad_width = max_length - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec.shape[1] > max_length:
        mel_spec = mel_spec[:, :max_length]

    return mel_spec

# Load data and preprocess it
words_dir = "words"
words = os.listdir(words_dir)[:20]  # Select 20 words
spectrograms = []
labels = []

for word in words:
    word_dir = os.path.join(words_dir, word)
    wav_files = os.listdir(word_dir)
    for wav_file in wav_files:
        wav_path = os.path.join(word_dir, wav_file)
        mel_spec = load_and_preprocess_audio(wav_path)
        spectrograms.append(mel_spec)
        labels.append(word)

# Convert lists to numpy arrays
spectrograms = np.array(spectrograms)
labels = np.array(labels)

# One-hot encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels_one_hot, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(spectrograms.shape[1], spectrograms.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(words), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(X_train[..., np.newaxis], y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test[..., np.newaxis], y_test)
print('Test accuracy:', test_acc)
