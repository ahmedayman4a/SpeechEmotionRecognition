# Speech Emotion Recognition Project Configuration

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 1. Data Parameters
DATA_DIR = "data/Crema"  # Path to the CREMA-D dataset containing .wav files
NUM_CLASSES = 6  # SAD, ANG, DIS, FEA, HAP, NEU (for CREMA-D)
# If using a combined dataset like EmoDSC with 7 classes, update this and labels below
EMOTION_LABELS = {
    'SAD': 0,
    'ANG': 1,
    'DIS': 2,
    'FEA': 3,
    'HAP': 4,
    'NEU': 5
}

# 2. Audio Preprocessing Parameters (for AudioPreprocessor)
TARGET_SAMPLE_RATE = 16000 # Hz, common for speech processing
VAD_MODE = 0 # Voice Activity Detection mode (0: off, 1-3: webrtcvad aggressiveness)
NORMALIZE_AUDIO = True # Whether to normalize audio waveform (z-score)
FRAME_MS_VAD = 30 # Frame duration in ms for VAD

# 3. Feature Extraction Parameters

# For Path A: 1D Spectral Features (target vector size: 162)
# These are parameters for underlying Librosa/Torchaudio calls if customized
N_FFT_COMMON = 512        # FFT window size (e.g., for MFCC, MelSpec, Chroma)
HOP_LENGTH_COMMON = 160   # Hop length for STFT
N_MFCC = 13               # Number of MFCCs to compute before mean
N_MELS_FOR_MELSPEC = 64   # Number of Mel bands for Mel spectrogram (before mean)
N_MELS_FOR_1D_FEAT = 135  # Number of Mel bands for 1D features (before mean)
# ZCR: 1 feature (mean)
# Chroma STFT: 12 features (mean)
# MFCCs: N_MFCC features (mean)
# RMS: 1 feature (mean)
# Mel-Spectrogram (values): N_MELS_FOR_MELSPEC features (mean)
# The paper mentions 162 distinct values. We'll need to ensure our feature extraction logic for Path A sums to this.
# For example, if using deltas for MFCCs, that would increase N_MFCC * (1+delta_order)
# The paper summary states "162 distinct values per audio file" after processing.


# For Path B: 2D Spectrogram Images
SPECTROGRAM_TYPE = 'melspectrogram' # 'melspectrogram' or 'spectrogram'
N_MELS_IMG = 64          # Number of Mel bands for the 2D image (becomes height)
N_FFT_IMG = 1024         # FFT window size for 2D spectrogram image
HOP_LENGTH_IMG = 256     # Hop length for 2D spectrogram image (affects width)
IMG_HEIGHT = 64          # Target height of the spectrogram image (same as N_MELS_IMG)
# IMG_WIDTH = 64           # REMOVED: Target width is no longer fixed, spectrogram width is variable
LOG_SPECTROGRAM_IMG = True # Apply log to the spectrogram image values
FMAX_IMG = 8000          # Maximum frequency for Mel spectrogram calculation (sr/2 for 16kHz)

# Activation functions are set in model instantiation, can be nn.ReLU(inplace=True) or nn.SiLU()

# 5. Training Parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
NUM_EPOCHS = 150
OPTIMIZER = 'Adam' # 'Adam', 'SGD', etc.
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5
MIN_LR_FACTOR = 0.0001

# For DataLoader
NUM_WORKERS = 4
PIN_MEMORY = True
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False

# 6. Other
DEVICE = 'cuda' # 'cuda' or 'cpu'
RANDOM_SEED = 42

# Placeholder for paths if needed later for saving models, logs etc.
MODEL_SAVE_DIR = "trained_models"
LOG_DIR = "logs"

# --- Dataset Normalization Stats ---
# Path to the file where dataset mean/std are saved
# Run utils/dataset_stats.py to generate this file
DATASET_STATS_FILE = os.path.join(os.path.dirname(__file__), 'dataset_stats.pt')

# --- Augmentation Config ---
AUGMENTATION = {
    'apply_noise': True,
    'noise_factor': 0.01, # Amount of Gaussian noise to add (relative to std deviation)
    
    'apply_volume': True,
    'volume_range': (0.7, 1.3), # Randomly scale volume between 80% and 120%
    
    'apply_specaugment': True,
    'specaugment_freq_mask_param': 27, # Max width of frequency mask (paper default: 27 for LibriSpeech)
    'specaugment_time_mask_param': 70, # Max width of time mask (paper default: 70 for LibriSpeech)
    'specaugment_num_freq_masks': 3,   # Number of frequency masks
    'specaugment_num_time_masks': 3    # Number of time masks
}

# --- Model Config ---
MODEL_PARAMS = {
    'num_classes': NUM_CLASSES,
    # Input dims (derived from feature extraction)
    'cnn1d_input_channels': 1,
    'cnn1d_num_features_dim': 162, # Output dim of _extract_1d_features 
    'cnn2d_input_channels': 1,
    
    # Configuration for new ResNet-like models
    'cnn1d_initial_out_channels': 16, 
    'cnn2d_initial_out_channels': 16,
    
    # Dropout Rates
    'cnn_dropout_rate': 0.4, 
    'mlp_hidden_units': 128, # Hidden units in the final MLP head
    'mlp_dropout_rate': 0.6,
    
    'activation_name': 'relu', # 'relu' or 'silu' (swish)
    'layers': [2, 2, 2, 2]
}
