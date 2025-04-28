# AI_Project_PSD
## Description
An artificial neural network classifier trained by backpropagation for polish spoken words recognition (PSD dataset).

## Project Goal
The aim of this project is to develop a neural network classifier capable of recognizing spoken Polish words. The system processes recorded audio, extracts spectrogram features, and trains using backpropagation to achieve high accuracy.

## Dataset
**Source**: PSD dataset
**Structure**: 
- For each sentence, the audio track is in a *.wav file, the transcription is in a *.txt file, and the start and end times of each word and phoneme are in a *.TextGrid file. 
- Individual sentences are numbered from 1 to 3000. The data is grouped into sections containing 500 sentences from different authors.
- Full number of files: 12,000 (the same number for each file type).

## Libraries used
- `os`: File and directory operations
- `tensorflow`: Audio processing and neural network training
- `tensorflow.signal`: Generating spectrograms
- `tensorflow.io`: Reading audio files
- `tensorflow.data`: Efficient data pipelines
- `tensorflow.keras.callbacks`: EarlyStopping and ModelCheckpoint
- `scikit-learn (sklearn)`: Preprocessing, label encoding, model evaluation
- `keras`: Model building, training, and evaluation
- `keras.layers`: Defining model layers (Conv, Dense, Pooling, Normalization)

## Data preparation
- **Words Preparation**:
  - Sentences split into individual words
  - Words sorted alphabetically
  - Words with at least 100 instances selected
- **Audio Processing**:
  - Read audio files (`tf.io.read_file`)
  - Decode (`tf.audio.decode_wav`)
  - Pad/truncate to 16,000 samples
- **Feature Extraction**:
  - Spectrogram generation using Short-Time Fourier Transform (STFT)
  - Normalize spectrograms
- **Label Encoding**:
  - Labels numerically encoded and one-hot encoded
- **Dataset Splitting**:
  - 70% training, 15% validation, 15% testing
  
### Model Architecture
- **Sequential Model** using Keras
- **Convolutional Layers**:
  - 3 Conv layers + MaxPooling
  - ReLU activation
  - L2 kernel regularization
- **Flatten and Dense Layers**:
  - Flatten layer
  - Dense layer (256 units, ReLU)
  - Dropout layer (rate = 0.5)
- **Output Layer**:
  - Dense layer with Softmax activation
  
## Training
- **Data Handling**:
  - Batching and prefetching using `tf.data.Dataset`
- **Normalization**:
  - Adapt normalization layer to dataset
- **Compilation**:
  - Loss: Categorical Cross-Entropy
  - Optimizer: Adam
  - Metric: Accuracy
- **Callbacks**:
  - EarlyStopping (patience = 4)
  - ModelCheckpoint (save best model)
- **Training Details**:
  - Up to 40 epochs (early stopping applied)
  
## Testing
- Final evaluation performed on the test dataset.
- 10 training loops to assess consistency.

## Result
The result was on average 80% accuracy while being trained on 50 words (18919 files 
containing one word, and one word having at least 100 examples).

## Authors
O. Paszkiewicz (MonDust) - [GitHub](https://github.com/MonDust)

Krzysztof Ostrzycki - [GitHub](https://github.com/KrzyszOst)

