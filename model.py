import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Normalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Function to load and preprocess audio files
def load_and_preprocess_audio(file_path, output_sequence_length=16000):
    audio_binary = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)

    if tf.shape(waveform)[0] < output_sequence_length:
        padding = output_sequence_length - tf.shape(waveform)[0]
        waveform = tf.pad(waveform, paddings=[[0, padding]])
    else:
        waveform = waveform[:output_sequence_length]

    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]

    return spectrogram

# Function to load dataset
def load_dataset(words_dir, words, output_sequence_length=16000):
    file_paths = []
    labels = []

    for word in words:
        word_dir = os.path.join(words_dir, word)
        wav_files = os.listdir(word_dir)
        for wav_file in wav_files:
            wav_path = os.path.join(word_dir, wav_file)
            file_paths.append(wav_path)
            labels.append(word)

    return file_paths, labels

# Function to preprocess dataset using tf.data
def preprocess_dataset(file_paths, labels, batch_size=64):
    def load_and_preprocess(file_path, label):
        spectrogram = load_and_preprocess_audio(file_path)
        return spectrogram, label

    file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((file_paths_ds, labels_ds))
    dataset = dataset.map(lambda file_path, label: (load_and_preprocess_audio(file_path), label), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

# Load words
with open(r'C:\Users\maria\Desktop\PG\SEM_IV\Artificial_Intelligence\Project\PSD\words_useful2.txt', 'r', encoding='utf-8') as file:
    words = file.read().splitlines()

# Directory containing words
words_dir = r'C:\Users\maria\Desktop\PG\SEM_IV\Artificial_Intelligence\Project\PSD\words'
maximum_number_of_words = len(words)
number_of_words = 50
if maximum_number_of_words < number_of_words:
    number_of_words = maximum_number_of_words
words = words[:number_of_words]

# Load data
file_paths, labels = load_dataset(words_dir, words)

# One-hot encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(file_paths, labels_one_hot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize the data
norm_layer = Normalization()
dummy_spectrogram = load_and_preprocess_audio(file_paths[0])
dummy_spectrogram = tf.expand_dims(dummy_spectrogram, axis=0)
norm_layer.adapt(dummy_spectrogram)

# Prepare datasets
train_dataset = preprocess_dataset(X_train, y_train)
val_dataset = preprocess_dataset(X_val, y_val)
test_dataset = preprocess_dataset(X_test, y_test)

# Define the model architecture
model = Sequential()
model.add(Input(shape=(dummy_spectrogram.shape[1], dummy_spectrogram.shape[2], 1)))
model.add(norm_layer)
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(len(words), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(train_dataset, epochs=40, validation_data=val_dataset, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)
