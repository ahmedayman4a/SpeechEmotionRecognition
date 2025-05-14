# Speech Emotion Recognition Project Configuration

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
    # 'SUP': 6, # Add Surprise if using 7 classes
}

# 2. Audio Preprocessing Parameters (for AudioPreprocessor)
TARGET_SAMPLE_RATE = 16000 # Hz, common for speech processing
VAD_MODE = 0 # Voice Activity Detection mode (0: off, 1-3: webrtcvad aggressiveness)
NORMALIZE_AUDIO = True # Whether to normalize audio waveform (z-score)
FRAME_MS_VAD = 30 # Frame duration in ms for VAD

# 3. Feature Extraction Parameters

# For Path A: 1D Spectral Features (target vector size: 162)
# These are parameters for underlying Librosa/Torchaudio calls if customized
# Default values from FeatureExtractor or typical values:
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
# The feature_extractor_1d should produce this fixed size vector.

# For Path B: 2D Spectrogram Images
SPECTROGRAM_TYPE = 'melspectrogram' # 'melspectrogram' or 'spectrogram'
N_MELS_IMG = 64          # Number of Mel bands for the 2D image (becomes height)
N_FFT_IMG = 1024         # FFT window size for 2D spectrogram image
HOP_LENGTH_IMG = 256     # Hop length for 2D spectrogram image (affects width)
IMG_HEIGHT = 64          # Target height of the spectrogram image
IMG_WIDTH = 64           # Target width of the spectrogram image (may require padding/truncating time axis)
LOG_SPECTROGRAM_IMG = True # Apply log to the spectrogram image values
UPPER_FREQ_LIMIT_KHZ = 10 # As per paper for spectrogram images (10000 Hz)

# 4. Model Parameters
# CNN1D specific (defaults match CombinedModel if not overridden there)
CNN1D_INPUT_CHANNELS = 27
CNN1D_NUM_FEATURES_DIM = 162 # This must match the output of 1D feature extraction

# CNN2D specific (defaults match CombinedModel if not overridden there)
CNN2D_INPUT_CHANNELS = 1
CNN2D_IMG_HEIGHT = IMG_HEIGHT # Should match the processed image height
CNN2D_IMG_WIDTH = IMG_WIDTH   # Should match the processed image width

# CombinedModel general type
MODEL_TYPE = "1d"  # Options: "combined", "1d", "2d"

# Shared for CNNs in CombinedModel
CNN_DROPOUT_RATE = 0.3

# MLP Head in CombinedModel
MLP_DROPOUT_RATE = 0.5
# Activation functions are set in model instantiation, can be nn.ReLU(inplace=True) or nn.SiLU()

# 5. Training Parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 150 # Or use early stopping
OPTIMIZER = 'Adam' # 'Adam', 'SGD', etc.
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
MIN_LR_FACTOR = 0.001

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