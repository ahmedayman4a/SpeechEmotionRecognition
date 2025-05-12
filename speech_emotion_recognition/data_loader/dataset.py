import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

try:
    from ..preprocessing.audio_preprocessor import AudioPreprocessor
    from ..preprocessing.feature_extractor import PaperCombinedFeatureExtractor
except ImportError:
    print("Attempting to import preprocessors from sibling 'preprocessing' directory for standalone testing of dataset.py.")
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent dir to path
    from preprocessing.audio_preprocessor import AudioPreprocessor
    from preprocessing.feature_extractor import PaperCombinedFeatureExtractor


class CremaDataset(Dataset):
    def __init__(self, 
                 file_paths_list: list, 
                 labels_list: list,
                 emotion_labels_map: dict,
                 transform_audio_func=None):
        """
        Args:
            file_paths_list (list): List of full paths to the .wav files for this dataset split.
            labels_list (list): List of integer labels corresponding to file_paths_list.
            emotion_labels_map (dict): Mapping from emotion string (e.g., 'SAD') to integer label.
            transform_audio_func (callable, optional): Optional transform to be applied on a raw waveform.
        """
        self.file_paths_list = file_paths_list
        self.labels_list = labels_list
        self.emotion_labels_map = emotion_labels_map # Still useful for num_classes or other metadata
        self.transform_audio_func = transform_audio_func
        
        if len(self.file_paths_list) != len(self.labels_list):
            raise ValueError("file_paths_list and labels_list must have the same length.")
        if not self.file_paths_list:
            print(f"Warning: file_paths_list is empty.")

        self.num_classes = len(emotion_labels_map)

    def __len__(self):
        return len(self.file_paths_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_paths_list[idx]
        label = self.labels_list[idx]
        file_name = os.path.basename(file_path)
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return torch.zeros((1,16000)), 16000, label, file_name # Return provided label

        if self.transform_audio_func:
            waveform = self.transform_audio_func(waveform)

        return waveform, sample_rate, label, file_name


def create_collate_fn(preprocessor: AudioPreprocessor, 
                      feature_extractor: PaperCombinedFeatureExtractor,
                      augmentation_config: dict, # Pass augmentation settings dict
                      is_training: bool = False):
    """Collate function to load, preprocess, augment (if training), and extract features."""
    
    # Initialize SpecAugment transform if needed
    spec_augment = None
    if is_training and augmentation_config.get('apply_specaugment', False):
        spec_augment = torchaudio.transforms.SpecAugment(
            freq_mask_param=augmentation_config.get('specaugment_freq_mask_param', 27),
            time_mask_param=augmentation_config.get('specaugment_time_mask_param', 70),
            iid_masks=True, # Apply masks independently
            n_freq_masks=augmentation_config.get('specaugment_num_freq_masks', 2),
            n_time_masks=augmentation_config.get('specaugment_num_time_masks', 2)
        )
        print("SpecAugment enabled for training batches.")

    def collate_fn(batch):
        # batch is a list of tuples: (waveform, sample_rate, label, filename)
        waveforms = [item[0] for item in batch]
        sample_rates = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        filenames = [item[3] for item in batch]

        # --- Apply Waveform Augmentations (only during training) ---
        augmented_waveforms = []
        if is_training:
            for wf in waveforms:
                temp_wf = wf.clone()
                # 1. Noise Injection
                if augmentation_config.get('apply_noise', False) and augmentation_config.get('noise_factor', 0) > 0:
                    noise = torch.randn_like(temp_wf) * augmentation_config['noise_factor'] # Use dict directly
                    temp_wf += noise
                
                # 2. Volume Perturbation
                if augmentation_config.get('apply_volume', False):
                    min_vol, max_vol = augmentation_config.get('volume_range', (1.0, 1.0))
                    scale = torch.rand(1).item() * (max_vol - min_vol) + min_vol
                    temp_wf *= scale
                
                # Clip to avoid potential issues from large noise/scaling
                temp_wf = torch.clamp(temp_wf, -1.0, 1.0)
                augmented_waveforms.append(temp_wf)
        else:
            augmented_waveforms = waveforms # Use original waveforms if not training

        # Prepare input for preprocessor
        preprocessor_input = list(zip(augmented_waveforms, sample_rates))

        # 1. Preprocess audio using the provided preprocessor instance
        # This handles resampling, VAD, normalization etc.
        # Output: [B, C, T_padded]
        try:
            padded_waveforms, _ = preprocessor(preprocessor_input)
        except Exception as e:
            print(f"Error during audio preprocessing: {e}")
            # Handle error: skip batch or return dummy data? Returning None for now.
            print(f"Problematic filenames (sample): {filenames[:2]}")
            return None, None, None, None # Indicate error

        # Handle potentially empty waveforms after VAD in preprocessor
        if padded_waveforms is None or padded_waveforms.numel() == 0:
            # print("Warning: Skipping batch due to empty waveforms after preprocessing (VAD?).")
            return None, None, None, None # Indicate empty batch

        # 2. Extract features using the provided feature extractor instance
        # The feature extractor handles dataset normalization internally if configured.
        # Output: features_1d [B, D_1d], features_2d [B, C_img, H, W]
        try:
            features_1d, features_2d = feature_extractor(padded_waveforms)
        except Exception as e:
             print(f"Error during feature extraction: {e}")
             print(f"Padded waveform shape: {padded_waveforms.shape}")
             print(f"Problematic filenames (sample): {filenames[:2]}")
             return None, None, None, None
        
        # --- Apply Spectrogram Augmentation (SpecAugment, only during training) ---
        if spec_augment is not None: # Check if transform was initialized
            # SpecAugment expects [.., freq, time]
            features_2d_permuted = features_2d.permute(0, 1, 3, 2) # [B, C, W, H]
            # Apply SpecAugment
            features_2d_augmented = spec_augment(features_2d_permuted)
            # Permute back: [B, C, H, W]
            features_2d = features_2d_augmented.permute(0, 1, 3, 2)
            # print("Applied SpecAugment") # Debug print

        return features_1d, features_2d, labels, filenames

    return collate_fn


def get_data_loader(file_paths_list: list, 
                    labels_list: list, 
                    emotion_labels_map: dict,
                    batch_size: int, 
                    audio_preprocessor: AudioPreprocessor, 
                    feature_extractor: PaperCombinedFeatureExtractor, 
                    is_training: bool = False,
                    shuffle: bool = True, 
                    num_workers: int = 4, 
                    pin_memory: bool = True,
                    augmentation_config: dict = None):
    """Creates a DataLoader for the CREMA-D dataset.

    Args:
        file_paths_list (list): List of full paths to .wav files.
        labels_list (list): List of integer labels corresponding to file paths.
        emotion_labels_map (dict): Mapping from emotion string to integer label.
        batch_size (int): Number of samples per batch.
        audio_preprocessor (AudioPreprocessor): Instance for audio preprocessing.
        feature_extractor (PaperCombinedFeatureExtractor): Instance for feature extraction.
        is_training (bool): If True, applies augmentations and drops the last batch if incomplete.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): If True, copies tensors into pinned memory before returning.
        augmentation_config (dict): Dictionary containing augmentation parameters.

    Returns:
        DataLoader: The configured DataLoader instance.
    """

    dataset = CremaDataset(
        file_paths_list=file_paths_list, 
        labels_list=labels_list,
        emotion_labels_map=emotion_labels_map
    )
    
    collate_fn = create_collate_fn(
        preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor,
        augmentation_config=augmentation_config, # Pass the dict here
        is_training=is_training
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training
    )
    
    print(f"Created DataLoader with {len(dataset)} samples. is_training={is_training}, shuffle={shuffle}")
    return data_loader


if __name__ == '__main__':
    # Example usage - this test needs to be updated to reflect the new Dataset structure
    print("--- Testing CremaDataset and get_data_loader with pre-defined file lists ---")
    # --- Configuration (mirroring parts of config.py for this test) ---
    TEST_DATA_ROOT_DIR = "data/Crema"  # Root directory of data
    TARGET_SR_TEST = 16000
    EMOTION_LABELS_TEST = { 'SAD': 0, 'ANG': 1, 'DIS': 2, 'FEA': 3, 'HAP': 4, 'NEU': 5 }
    
    N_MFCC_1D_TEST = 13
    N_MELS_FOR_1D_FEAT_TEST = 135
    N_MELS_2D_TEST = 64
    CNN1D_NUM_FEATURES_DIM_TEST = 162 # Match 1D feature output dim

    # Dummy augmentation config for testing
    AUG_CONFIG_TEST = {
        'apply_noise': False,
        'noise_factor': 0.001,
        'apply_volume': False,
        'volume_range': (0.9, 1.1),
        'apply_specaugment': False,
        'specaugment_freq_mask_param': 10,
        'specaugment_time_mask_param': 20,
        'specaugment_num_freq_masks': 1,
        'specaugment_num_time_masks': 1
    }

    # Create dummy data directory and files if they don't exist
    if not os.path.exists(TEST_DATA_ROOT_DIR):
        print(f"Test data directory {TEST_DATA_ROOT_DIR} does not exist. Creating dummy files...")
        os.makedirs(TEST_DATA_ROOT_DIR, exist_ok=True)

    dummy_files_info = [
        ("1091_TIE_SAD_XX.wav", EMOTION_LABELS_TEST['SAD']),
        ("1001_DFA_ANG_XX.wav", EMOTION_LABELS_TEST['ANG']),
        ("1002_WSI_DIS_XX.wav", EMOTION_LABELS_TEST['DIS']),
        ("1003_IWL_FEA_XX.wav", EMOTION_LABELS_TEST['FEA']),
    ]
    

    test_file_paths = [os.path.join(TEST_DATA_ROOT_DIR, fname) for fname, _ in dummy_files_info]
    test_labels = [label for _, label in dummy_files_info]

    print(f"Attempting to load data using predefined lists from: {os.path.abspath(TEST_DATA_ROOT_DIR)}")

    # Instantiate Preprocessor and Feature Extractor
    audio_proc = AudioPreprocessor(target_sample_rate=TARGET_SR_TEST, vad_mode=0)
    # Load dummy stats or set to None if feature extractor can handle it
    dummy_mean_1d = torch.randn(CNN1D_NUM_FEATURES_DIM_TEST)
    dummy_std_1d = torch.rand(CNN1D_NUM_FEATURES_DIM_TEST) + 1e-6
    dummy_mean_2d = torch.randn(1)
    dummy_std_2d = torch.rand(1) + 1e-6

    feature_ext = PaperCombinedFeatureExtractor(
        sr=TARGET_SR_TEST, n_mfcc_1d=N_MFCC_1D_TEST, n_mels_for_1d_feat=N_MELS_FOR_1D_FEAT_TEST,
        n_mels_2d=N_MELS_2D_TEST,
        fmax_spec_img= TARGET_SR_TEST // 2,
        hop_length_2d=256,
        n_fft_2d=1024,
        dataset_mean_1d=dummy_mean_1d, dataset_std_1d=dummy_std_1d, 
        dataset_mean_2d=dummy_mean_2d, dataset_std_2d=dummy_std_2d,
        log_spec_img=True # Add other necessary params from config if needed by extractor
    )

    try:
        # Test with is_training = False (no augmentation)
        print("\n--- Testing DataLoader (is_training=False) ---")
        test_dataloader_noaug = get_data_loader(
            file_paths_list=test_file_paths,
            labels_list=test_labels,
            emotion_labels_map=EMOTION_LABELS_TEST,
            batch_size=2,
            audio_preprocessor=audio_proc,
            feature_extractor=feature_ext,
            augmentation_config=AUG_CONFIG_TEST, # Pass test config
            is_training=False, # Test without aug
            shuffle=False, num_workers=0, pin_memory=False # Easier for local test
        )
        
        print(f"Successfully created DataLoader (no aug) with {len(test_dataloader_noaug.dataset)} samples.")
        for i, batch_data in enumerate(test_dataloader_noaug):
            if batch_data[0] is None: # Skip if collate_fn returned error
                print(f"Skipping batch {i+1} due to preprocessing/feature extraction error.")
                continue
            feats_1d, feats_2d, lbls, fnames = batch_data
            print(f"Batch {i+1} (no aug): 1D={feats_1d.shape}, 2D={feats_2d.shape}, Lbls={lbls}")
            # Add assertions if needed
        print("No-Augmentation DataLoader test completed.")

        # Test with is_training = True (augmentation)
        print("\n--- Testing DataLoader (is_training=True) ---")
        AUG_CONFIG_TEST['apply_specaugment'] = True # Enable for this test
        test_dataloader_aug = get_data_loader(
            file_paths_list=test_file_paths,
            labels_list=test_labels,
            emotion_labels_map=EMOTION_LABELS_TEST,
            batch_size=2,
            audio_preprocessor=audio_proc,
            feature_extractor=feature_ext,
            augmentation_config=AUG_CONFIG_TEST, # Pass test config
            is_training=True, # Test with aug
            shuffle=False, num_workers=0, pin_memory=False
        )

        print(f"Successfully created DataLoader (aug) with {len(test_dataloader_aug.dataset)} samples.")
        for i, batch_data in enumerate(test_dataloader_aug):
            if batch_data[0] is None:
                print(f"Skipping batch {i+1} due to preprocessing/feature extraction error.")
                continue
            feats_1d, feats_2d, lbls, fnames = batch_data
            print(f"Batch {i+1} (aug): 1D={feats_1d.shape}, 2D={feats_2d.shape}, Lbls={lbls}")
            # Add assertions if needed
        print("Augmentation DataLoader test completed.")
        
    except Exception as e:
        print(f"Error during DataLoader test: {e}")
        import traceback
        traceback.print_exc() 