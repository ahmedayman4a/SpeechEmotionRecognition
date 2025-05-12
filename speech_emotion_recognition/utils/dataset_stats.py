import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import os
import sys
from torch.utils.data import Dataset, DataLoader
import argparse

# Add project root to path to allow sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from speech_emotion_recognition.config import config
from speech_emotion_recognition.data_loader.dataset import CremaDataset
from speech_emotion_recognition.preprocessing.audio_preprocessor import AudioPreprocessor
from speech_emotion_recognition.preprocessing.feature_extractor import PaperCombinedFeatureExtractor



# --- Simple Dataset for Stats Calculation ---
class StatsCalculationDataset(Dataset):
    """A simple dataset to load audio files for statistics calculation."""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        print(f"Initialized StatsCalculationDataset with {len(file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            return waveform, sample_rate, file_path # Include filepath for debugging maybe
        except Exception as e:
            # print(f"Warning: Error loading {file_path}: {e}. Skipping file.")
            # Return None placeholders which will be filtered in collate_fn
            return None, None, file_path

# --- Collate function for Stats Calculation ---
def create_stats_collate_fn(preprocessor, feature_extractor):
    """Collate function to preprocess audio and extract features for stats calculation."""
    def collate_fn(batch):
        # Filter out samples that failed to load
        valid_batch = [(wf, sr, fp) for wf, sr, fp in batch if wf is not None]
        
        if not valid_batch:
            # print("Warning: Skipping batch because all files failed to load.")
            return None, None # Indicate empty batch

        waveforms = [item[0] for item in valid_batch]
        sample_rates = [item[1] for item in valid_batch]
        # filenames = [item[2] for item in valid_batch] # Can be used for debugging

        # 1. Preprocess audio using the provided preprocessor instance
        # The preprocessor's __call__ expects a list of (waveform, sr) tuples
        preprocessor_input = list(zip(waveforms, sample_rates))
        padded_waveforms, _ = preprocessor(preprocessor_input) # Padded shape: [B, C, T]

        # Handle potentially empty waveforms after VAD
        if padded_waveforms.numel() == 0:
            # print("Warning: Skipping batch due to empty waveforms after preprocessing (VAD?).")
            return None, None

        # 2. Extract features using the provided feature extractor instance
        # Note: This feature extractor should NOT apply the final dataset normalization itself.
        # It should only perform the base feature extraction (1D vector, 2D image).
        features_1d, features_2d = feature_extractor(padded_waveforms) # [B, D_1d], [B, C_img, H, W]

        return features_1d, features_2d

    return collate_fn

# --- Main Calculation Function ---
def calculate_mean_std(data_dir, save_path, batch_size=32, num_workers=4):
    """Calculates and saves the mean and standard deviation for 1D and 2D features."""
    print(f"Starting dataset statistics calculation for directory: {data_dir}")
    print(f"Using Batch Size: {batch_size}, Num Workers: {num_workers}")

    # Find all wav files recursively
    all_files = []
    print("Scanning for .wav files...")
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(".wav"):
                all_files.append(os.path.join(root, filename))

    if not all_files:
        print(f"Error: No .wav files found in {data_dir} or its subdirectories.")
        return False

    print(f"Found {len(all_files)} audio files.")

    # Create dataset and dataloader
    stats_dataset = StatsCalculationDataset(all_files)

    # Initialize Preprocessor and Feature Extractor using config
    preprocessor = AudioPreprocessor(
        target_sample_rate=config.TARGET_SAMPLE_RATE,
        vad_mode=config.VAD_MODE,
        normalize_audio=config.NORMALIZE_AUDIO,
        frame_ms=config.FRAME_MS_VAD
    )
    feature_extractor = PaperCombinedFeatureExtractor(
        sr=config.TARGET_SAMPLE_RATE,
        n_fft_1d=config.N_FFT_COMMON, hop_length_1d=config.HOP_LENGTH_COMMON,
        n_mfcc_1d=config.N_MFCC, n_mels_for_1d_feat=config.N_MELS_FOR_1D_FEAT,
        n_fft_2d=config.N_FFT_IMG, hop_length_2d=config.HOP_LENGTH_IMG, n_mels_2d=config.N_MELS_IMG,
        log_spec_img=config.LOG_SPECTROGRAM_IMG, fmax_spec_img=config.FMAX_IMG)

    collate_fn = create_stats_collate_fn(preprocessor, feature_extractor)
    stats_loader = DataLoader(
        stats_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False # Process all samples
    )

    # Variables to accumulate stats (use float64 for precision)
    # Determine feature dimensions from config or dummy extraction
    dim_1d = config.CNN1D_NUM_FEATURES_DIM
    
    sum_1d = torch.zeros(dim_1d, dtype=torch.float64)
    sum_sq_1d = torch.zeros(dim_1d, dtype=torch.float64)
    count_1d = 0 # Total number of 1D feature vectors processed

    sum_2d = torch.zeros(1, dtype=torch.float64) # Global sum for 2D features
    sum_sq_2d = torch.zeros(1, dtype=torch.float64) # Global sum of squares for 2D
    count_2d_pixels = 0 # Total number of pixels across all 2D features processed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for accumulation")
    sum_1d = sum_1d.to(device)
    sum_sq_1d = sum_sq_1d.to(device)
    sum_2d = sum_2d.to(device)
    sum_sq_2d = sum_sq_2d.to(device)

    print("Iterating through dataset to calculate statistics...")
    for batch_idx, (features_1d, features_2d) in enumerate(tqdm(stats_loader, desc="Calculating Stats")):
        if features_1d is None or features_2d is None:
            # print(f"Skipping batch {batch_idx+1} due to loading/processing errors.")
            continue
            
        # Move features to the accumulation device
        features_1d = features_1d.to(device, non_blocking=True).squeeze() # Shape: [B, D_1d]
        features_2d = features_2d.to(device, non_blocking=True).squeeze() # Shape: [B, C=1, H, W]

        # Update 1D stats
        # Ensure calculations are done in float64
        current_batch_size = features_1d.shape[0]
        if current_batch_size > 0:
            sum_1d += torch.sum(features_1d, dim=0).to(torch.float64)
            sum_sq_1d += torch.sum(features_1d.to(torch.float64)**2, dim=0)
            count_1d += current_batch_size

        # Update 2D stats (globally across all pixels)
        num_pixels_in_batch = features_2d.numel()
        if num_pixels_in_batch > 0:
            sum_2d += torch.sum(features_2d).to(torch.float64)
            sum_sq_2d += torch.sum(features_2d.to(torch.float64)**2)
            count_2d_pixels += num_pixels_in_batch

    if count_1d == 0 or count_2d_pixels == 0:
        print("Error: No features were successfully processed. Cannot calculate statistics.")
        print("Please check data loading, preprocessing (VAD settings?), and feature extraction steps.")
        return False

    print(f"Finished iteration. Processed {count_1d} samples for 1D stats and {count_2d_pixels} pixels for 2D stats.")

    # Calculate mean and std (still on device)
    mean_1d = sum_1d / count_1d
    # Add epsilon before sqrt for numerical stability
    std_1d = torch.sqrt((sum_sq_1d / count_1d) - (mean_1d ** 2) + 1e-9) 

    mean_2d = sum_2d / count_2d_pixels
    # Add epsilon before sqrt for numerical stability
    std_2d = torch.sqrt((sum_sq_2d / count_2d_pixels) - (mean_2d ** 2) + 1e-9)

    # Move final results to CPU for saving
    mean_1d = mean_1d.cpu()
    std_1d = std_1d.cpu()
    mean_2d = mean_2d.cpu() # Scalar tensor
    std_2d = std_2d.cpu() # Scalar tensor

    # Optional: Clamp std dev to avoid near-zero values if necessary, though epsilon helps
    std_1d = torch.clamp(std_1d, min=1e-6)
    std_2d = torch.clamp(std_2d, min=1e-6)

    print("--- Dataset Statistics ---")
    print(f"1D Features ({mean_1d.shape[0]} dims):")
    # Convert to numpy for cleaner printing if desired
    print(f"  Mean: {mean_1d.numpy().round(6)}")
    print(f"  Std Dev: {std_1d.numpy().round(6)}")
    print("2D Features (Global):")
    print(f"  Mean: {mean_2d.item():.6f}")
    print(f"  Std Dev: {std_2d.item():.6f}")
    print("-------------------------")

    # Ensure the directory for the save path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save stats to the specified file path
    try:
        torch.save({
            'mean_1d': mean_1d.float(), # Save as float32
            'std_1d': std_1d.float(),
            'mean_2d': mean_2d.float(),
            'std_2d': std_2d.float()
        }, save_path)
        print(f"Statistics successfully saved to: {save_path}")
        return True
    except Exception as e:
        print(f"Error saving statistics to {save_path}: {e}")
        return False


# --- Main execution block to run the calculation ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and save dataset normalization statistics.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help="Path to the base directory containing audio files (e.g., CREMA-D/AudioWAV)."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=config.DATASET_STATS_FILE, # Use path from config by default
        help="Path to save the calculated statistics (.pt file)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, os.cpu_count() // 2 if os.cpu_count() else 1), # Sensible default
        help="Number of worker processes for data loading."
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
         print(f"Error: Data directory not found at {args.data_dir}")
         print("Please provide a valid path using --data_dir or ensure config.DATA_DIR is correct.")
    else:
        success = calculate_mean_std(
            data_dir=args.data_dir,
            save_path=args.save_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        if success:
            print("\\nRecommendation: Ensure the DATASET_STATS_FILE path in config.py is correct:")
            print(f"DATASET_STATS_FILE = \'{os.path.relpath(args.save_path, config.BASE_DIR)}\'")
            print("Then, update DATASET_MEAN_1D, etc., in config.py to `None` if you want train.py to load them automatically,")
            print("or manually copy the printed values into the config if preferred.")
        else:
            print("\\nStatistics calculation failed.") 