# Speaker-Identification
```markdown
# Deep Learning-Based Speaker Identification System

## Overview

This project implements a comprehensive speaker identification system using multiple deep learning architectures including LSTM, CNN-LSTM, Transformer, and attention-based models. The system achieves up to **97% accuracy** on a 20-speaker identification task using MFCC features extracted from audio signals.

## Authors

- **Khoi Nguyen Ta** - Department of Artificial Intelligence, FPT University, Danang
- **Quoc Hoan Doan Van** - Department of Artificial Intelligence, FPT University, Danang
- **Van Hon Nguyen** - Department of Artificial Intelligence, FPT University, Danang
- **Luong Vuong Nguyen** (Corresponding Author) - Department of Artificial Intelligence, FPT University, Danang

**Contact:** vuongnl3@fe.edu.vn

## Table of Contents

- [Features](#features)
- [Model Architectures](#model-architectures)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Feature Extraction Pipeline](#feature-extraction-pipeline)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Training Configuration](#training-configuration)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- 🎯 Multiple neural network architectures for comparative analysis
- 🎵 MFCC-based audio feature extraction with STFT and Mel filterbank
- 📊 Data augmentation with Gaussian white noise for robustness
- 📈 Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- 📉 Confusion matrix visualization for detailed error analysis
- 📊 Training history plots for model performance tracking
- 🔄 Early stopping and learning rate scheduling

## Model Architectures

### 1. Baseline LSTM Model
Two-layer LSTM with batch normalization and dropout for temporal sequence modeling.
- **Architecture:** 2 stacked LSTM layers (128, 64 units)
- **Test Accuracy:** 87.00%
- **Precision:** 88.23%
- **Recall:** 87.00%
- **F1-Score:** 85.93%
- **Parameters:** ~128K

### 2. CNN-LSTM Hybrid Model ⭐
Combines convolutional layers for spectral feature extraction with LSTM for temporal modeling.
- **Architecture:** 3 Conv2D layers + 2 LSTM layers
- **Test Accuracy:** 97.00%
- **Precision:** 97.74%
- **Recall:** 97.00%
- **F1-Score:** 96.91%
- **Parameters:** ~279K

### 3. Transformer Model ⭐
Self-attention based architecture using Transformer encoder blocks for global relationship modeling.
- **Architecture:** Transformer encoder with multi-head attention (4 heads)
- **Test Accuracy:** 97.00%
- **Precision:** 97.50%
- **Recall:** 97.00%
- **F1-Score:** 96.83%
- **Parameters:** ~95K

### 4. LSTM + Multi-Head Attention ⭐
Enhanced LSTM with multi-head attention mechanism to focus on salient speech segments.
- **Architecture:** 2 LSTM layers + Multi-head attention
- **Test Accuracy:** 97.00%
- **Precision:** 97.50%
- **Recall:** 97.00%
- **F1-Score:** 96.97%
- **Parameters:** ~240K

### 5. Feature Fusion Model
Parallel CNN and LSTM branches with feature fusion at latent representation level.
- **Architecture:** Parallel CNN + LSTM with fusion layer
- **Test Accuracy:** 84.00%
- **Precision:** 89.11%
- **Recall:** 84.00%
- **F1-Score:** 82.59%

## Dataset

- **Source:** LibriSpeech corpus (clean speech subset)
- **Speakers:** 20 (9 male, 11 female)
- **Training Samples:** 2,198 clean utterances + 2,198 augmented (total 4,396)
- **Validation Split:** 10% of training data
- **Test Samples:** 100 utterances (5 per speaker)
- **Audio Format:** 16 kHz mono WAV files
- **Augmentation:** Gaussian white noise at 3% noise level (SNR)

## Requirements

```txt
tensorflow>=2.10.0
numpy>=1.21.0
librosa>=0.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/speaker-identification.git
cd speaker-identification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Feature Extraction Pipeline

The system uses a standard MFCC (Mel-Frequency Cepstral Coefficients) extraction pipeline:

### Pipeline Steps

1. **Pre-processing and Framing**
   - Frame length: 25ms (400 samples at 16 kHz)
   - Hop length: 10ms (160 samples)
   - Window function: Hamming window

2. **Short-Time Fourier Transform (STFT)**
   - Converts time-domain signal to frequency domain
   - Computes magnitude spectrogram

3. **Mel Filterbank**
   - 23 triangular filters
   - Frequency range: 125 Hz - 7.5 kHz
   - Maps linear frequencies to perceptual Mel scale

4. **Log Compression**
   - Applies logarithm to compress dynamic range
   - Produces log-Mel spectrogram

5. **Discrete Cosine Transform (DCT)**
   - Decorrelates Mel coefficients
   - Extracts 13 MFCC coefficients per frame

6. **Sequence Padding/Truncation**
   - Fixed length: 500 frames (5 seconds)
   - Zero-padding for shorter utterances
   - Truncation for longer utterances

**Final Feature Shape:** 500 × 13 (time steps × MFCC features)

### Mathematical Formulation

Hamming Window:
```
w(n) = 0.54 - 0.46 * cos(2πn/(N-1)), n = 0, 1, ..., N-1
```

Mel Scale:
```
f_Mel = 1127 * log(1 + f/700)
```

MFCC Feature Vector:
```
MFCC_t = [c1(t), c2(t), ..., c13(t)]
```

## Usage

### Data Preparation

```python
import librosa
import numpy as np
from feature_extraction import audio_to_mfcc, pad_or_trim_mfcc

# Load audio file
audio, sr = librosa.load('path/to/audio.wav', sr=16000, mono=True)

# Extract MFCC features
mfcc = audio_to_mfcc(audio, sampling_rate=16000)

# Pad or trim to fixed length
mfcc_padded = pad_or_trim_mfcc(mfcc, max_frames=500)
```

### Training Models

```python
from models import build_lstm_model, build_cnn_lstm_model, build_transformer_model
from tensorflow import keras

# Load preprocessed data
X_train, y_train = load_training_data()
X_val, y_val = load_validation_data()

# Build CNN-LSTM model
model = build_cnn_lstm_model(
    input_shape=(500, 13),
    num_classes=20
)

# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=20,
        monitor='val_accuracy',
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

### Evaluating Models

```python
from evaluation import full_evaluation

# Load test data
X_test, y_test = load_test_data()

# Evaluate model
accuracy, precision, recall, f1 = full_evaluation(
    model=model,
    X_test=X_test,
    y_test=y_test,
    label_to_index=label_to_index,
    model_name="CNN-LSTM"
)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Prediction

```python
# Predict speaker for new audio
def predict_speaker(audio_path, model, label_to_index):
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    mfcc = audio_to_mfcc(audio)
    mfcc_padded = pad_or_trim_mfcc(mfcc, max_frames=500)
    
    # Reshape for model input
    mfcc_input = np.expand_dims(mfcc_padded, axis=0)
    
    # Predict
    prediction = model.predict(mfcc_input)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Get speaker name
    index_to_label = {v: k for k, v in label_to_index.items()}
    speaker_name = index_to_label[predicted_class]
    
    return speaker_name, prediction[predicted_class]

# Example usage
speaker, confidence = predict_speaker('test_audio.wav', model, label_to_index)
print(f"Predicted Speaker: {speaker} (Confidence: {confidence:.2%})")
```

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| LSTM (2-layer) | 87.00% | 88.23% | 87.00% | 85.93% | ~128K |
| **CNN-LSTM (hybrid)** | **97.00%** | **97.74%** | **97.00%** | **96.91%** | ~279K |
| **Transformer** | **97.00%** | **97.50%** | **97.00%** | **96.83%** | ~95K |
| **LSTM + Multi-Head Attention** | **97.00%** | **97.50%** | **97.00%** | **96.97%** | ~240K |
| Feature Fusion (CNN + LSTM) | 84.00% | 89.11% | 84.00% | 82.59% | ~450K |

### Model Performance Insights

- **Top 3 models** (CNN-LSTM, Transformer, LSTM+Attention) achieve 97% accuracy, misclassifying only 3 out of 100 test utterances
- **10% improvement** over baseline LSTM when using hybrid architectures or attention mechanisms
- **Transformer model** achieves competitive performance with fewer parameters (~95K vs ~279K for CNN-LSTM)
- **Feature Fusion model** underperforms expectations, suggesting challenges in naive feature combination

## Key Findings

1. **Hybrid Architectures Dominate**
   - CNN-LSTM, Transformer, and LSTM+Attention significantly outperform baseline LSTM (97% vs 87%)
   - Combining spectral and temporal modeling yields superior results

2. **Attention Mechanisms Are Crucial**
   - Attention enables models to focus on speaker-discriminative speech segments
   - Multi-head attention effectively captures global relationships in sequences

3. **Data Augmentation Improves Robustness**
   - Adding Gaussian noise during training enhances generalization
   - Models trained with augmentation perform well on clean test data

4. **CNN Feature Extraction Is Effective**
   - Convolutional layers extract robust spectral features (formant patterns, pitch contours)
   - CNN front-end + LSTM back-end leverages both local and temporal information

5. **Model Convergence at 97%**
   - Three different architectures converge to same accuracy ceiling
   - Suggests 97% may be near maximum achievable with current features and dataset

## Project Structure

```
speaker-identification/
├── notebooks/
│   └── Copy_of_SLP_Speaker_Identification-1.ipynb  # Main implementation
├── paper/
│   └── Deep_Learning_Based_Speaker_Identification_Using_Hybrid_Neural_Architectures_and_Feature_Fusion_Techniques-3.pdf
├── src/
│   ├── models.py                    # Model architectures
│   ├── feature_extraction.py        # MFCC extraction pipeline
│   ├── data_augmentation.py         # Noise augmentation
│   ├── evaluation.py                # Evaluation utilities
│   └── utils.py                     # Helper functions
├── models/                          # Saved model checkpoints
├── data/                            # Dataset directory
│   ├── train/
│   └── test/
├── results/                         # Training results and plots
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── LICENSE                          # License file
```

## Training Configuration

### Hyperparameters

- **Optimizer:** Adam
- **Learning Rate:** 1e-4 (0.0001)
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Batch Size:** 32
- **Epochs:** 100 (with early stopping)
- **Dropout Rate:** 0.2-0.3 (varies by model)

### Callbacks

- **EarlyStopping:** Patience = 20 epochs, monitors validation accuracy
- **ModelCheckpoint:** Saves best model based on validation accuracy
- **ReduceLROnPlateau:** Reduces learning rate by 0.5x after 3 epochs without improvement

### Data Split

- **Training:** 89.9% (~3,956 samples)
- **Validation:** 10.1% (~440 samples)
- **Test:** 100 samples (5 per speaker, held out)

## Future Work

- 🔍 **Larger Datasets:** Explore datasets with more speakers (100+) and diverse acoustic conditions
- 🧠 **Advanced Fusion Strategies:** Investigate attention-based fusion mechanisms
- 🎯 **Pre-trained Models:** Experiment with wav2vec 2.0, Whisper, HuBERT embeddings
- 🚀 **Real-time Deployment:** Implement streaming inference for live speaker identification
- 🌐 **Noisy Environments:** Test robustness on real-world noisy audio
- 📱 **Mobile Optimization:** Deploy lightweight models for edge devices
- 🔊 **Multi-speaker Scenarios:** Handle overlapping speech and speaker diarization
- 🌍 **Cross-lingual Testing:** Evaluate on multilingual speaker datasets

## References

1. Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., Khudanpur, S. (2018). "X-vectors: Robust DNN embeddings for speaker recognition." *Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP)*, pp. 5329-5333.

2. Prachi, N.N., Nahiyan, F.M., Habibullah, M., Khan, R. (2022). "Deep learning based speaker recognition system with CNN and LSTM techniques." *Interdisciplinary Research in Technology and Management (IRTM)*, pp. 1-6.

3. Wang, R., Ao, J., Zhou, L., Liu, S., Wei, Z., Ko, T., Li, Q., Zhang, Y. (2022). "Multi-view self-attention based transformer for speaker recognition." *ICASSP 2022*, pp. 6732-6736.

4. Saritha, B., Laskar, R.H., Choudhury, M., K, A.M. (2024). "Optimizing speaker identification through SincSquareNet and SincNet fusion with attention mechanism." *Procedia Computer Science*, 233, 215-225.

5. Hassanzadeh, H., Qadir, J.A., Omer, S.M., Ahmed, M.H., Khezri, E. (2024). "Deep learning for speaker recognition: A comparative analysis of 1D-CNN and LSTM models using diverse datasets." *4th Interdisciplinary Conference on Electrics and Computer (INTCEC)*, pp. 1-8.

6. Emon, J.I., Salek, M.A., Alam, K.T. (2025). "Whisper speaker identification: Leveraging pre-trained multilingual transformers for robust speaker embeddings." *arXiv preprint arXiv:2503.10446*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{ta2026speaker,
  title={Enhancing Speaker Identification with Hybrid Deep Learning Models and Multi-Feature Fusion},
  author={Ta, Khoi Nguyen and Van, Quoc Hoan Doan and Nguyen, Van Hon and Nguyen, Luong Vuong},
  journal={Department of Artificial Intelligence, FPT University},
  year={2026}
}
```

## Acknowledgments

This research was conducted at the **Department of Artificial Intelligence, FPT University, Danang, Vietnam**. We acknowledge:

- The creators of the **LibriSpeech corpus** for providing the audio dataset
- The **TensorFlow** and **Librosa** development teams for their excellent tools
- Our advisors and colleagues for valuable feedback and support

## Contact

For questions, collaborations, or issues, please contact:

- **Email:** vuongnl3@fe.edu.vn
- **Institution:** Department of Artificial Intelligence, FPT University, Danang, 550000, Vietnam

---

**⭐ Star this repository if you find it helpful!**
```
