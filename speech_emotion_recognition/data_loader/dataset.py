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
    def __init__(self, data_dir, emotion_labels_map, transform_audio_func=None):
        """
        Args:
            data_dir (string): Directory with all the .wav files.
            emotion_labels_map (dict): Mapping from emotion string (e.g., 'SAD') to integer label.
            transform_audio_func (callable, optional): Optional transform to be applied on a raw waveform.
                                                      This is more for simple waveform transforms.
                                                      Heavy preprocessing/feature extraction is better in collate_fn.
        """
        self.data_dir = data_dir
        self.emotion_labels_map = emotion_labels_map
        self.transform_audio_func = transform_audio_func
        
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        if not self.file_list:
            print(f"Warning: No .wav files found in directory: {data_dir}")

        self.num_classes = len(emotion_labels_map)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        parts = file_name.split('_')
        emotion_str = 'NEU' # Default
        if len(parts) > 2:
            potential_emotion_str = parts[2]
            if potential_emotion_str in self.emotion_labels_map:
                emotion_str = potential_emotion_str
            else:
                print(f"Warning: Parsed emotion '{potential_emotion_str}' from {file_name} not in emotion_labels_map. Using NEU.")
        else:
            print(f"Warning: Could not parse emotion from filename {file_name}. Using NEU.")

        label = self.emotion_labels_map.get(emotion_str)
        # This should always find a label now due to the check above, but as a safeguard:
        if label is None: 
            print(f"Critical Warning: Emotion string '{emotion_str}' (from {file_name}) resulted in None label. Defaulting to NEU's ID.")
            label = self.emotion_labels_map.get('NEU', 0)


        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            # Return dummy data or raise error to be caught by DataLoader
            return torch.zeros((1,16000)), 16000, self.emotion_labels_map.get('NEU',0), file_name

        if self.transform_audio_func:
            waveform = self.transform_audio_func(waveform)

        return waveform, sample_rate, label, file_name


def create_collate_fn(audio_preprocessor: AudioPreprocessor, 
                        feature_extractor: PaperCombinedFeatureExtractor,
                        target_sample_rate: int):
    """
    Creates a collate_fn for the DataLoader.
    This function will:
    1. Take a batch of (waveform, sample_rate, label, file_name) tuples.
    2. Apply audio_preprocessor to waveforms (resample, VAD, normalize).
    3. Apply feature_extractor to get 1D and 2D features.
    4. Pad features if necessary (though feature_extractor should handle batching internally now).
    5. Return batched 1D features, 2D features, and labels.
    """
    def collate_fn(batch):
        # batch is a list of tuples: (waveform, sample_rate, label, file_name)
        waveforms_orig, sample_rates_orig, labels, file_names = zip(*batch)

        # 1. Preprocess audio waveforms
        # The AudioPreprocessor's __call__ method expects a list of (waveform, orig_sr) tuples
        # and returns (padded_waveforms_batch, processed_sample_rates_batch)
        # Ensure waveforms are suitable for preprocessor (e.g. [C,T] or [T])
        
        batch_for_preprocessing = []
        for wf, sr in zip(waveforms_orig, sample_rates_orig):
            # Ensure waveform tensor, handle potential issues from __getitem__ if load failed
            if not isinstance(wf, torch.Tensor): # Should not happen with current __getitem__
                print(f"Warning: Waveform is not a tensor. Creating dummy waveform.")
                wf = torch.zeros((1, target_sample_rate)) # Dummy waveform
            batch_for_preprocessing.append((wf, sr))

        # preprocessed_waveforms_batch: [B, 1, Time_processed]
        # processed_srs_batch: [B] (all should be target_sample_rate)
        preprocessed_waveforms_batch, _ = audio_preprocessor(batch_for_preprocessing)

        # 2. Extract features using PaperCombinedFeatureExtractor
        # Its __call__ method takes a batch of preprocessed waveforms [B, 1, T_proc] or [B, T_proc]
        # and returns (batch_features_1d, batch_features_2d)
        batch_features_1d, batch_features_2d = feature_extractor(preprocessed_waveforms_batch)
        
        # batch_features_1d: (B, 1, 162)
        # batch_features_2d: (B, 1, img_H, img_W)
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return batch_features_1d, batch_features_2d, labels_tensor, list(file_names)

    return collate_fn


def get_data_loader(data_dir, emotion_labels_map, batch_size, 
                    audio_preprocessor: AudioPreprocessor, 
                    feature_extractor: PaperCombinedFeatureExtractor,
                    target_sample_rate: int, # Passed to collate_fn for safety with dummy data
                    shuffle=True, num_workers=4, pin_memory=True):
    
    dataset = CremaDataset(data_dir=data_dir, emotion_labels_map=emotion_labels_map)
    
    collate_function = create_collate_fn(
        audio_preprocessor=audio_preprocessor,
        feature_extractor=feature_extractor,
        target_sample_rate=target_sample_rate
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_function
    )
    return dataloader


if __name__ == '__main__':
    # Example usage:
    # This __main__ block requires config values. Ideally, load from config.py
    # For simplicity here, define them directly or ensure config.py is importable.
    
    # --- Configuration (mirroring parts of config.py for this test) ---
    TEST_DATA_DIR = "../../data/Crema"  # Adjust this path to your actual or dummy data
    # If TEST_DATA_DIR does not exist, this test will fail unless dummy files are created.
    
    TARGET_SR_TEST = 16000
    EMOTION_LABELS_TEST = { 'SAD': 0, 'ANG': 1, 'DIS': 2, 'FEA': 3, 'HAP': 4, 'NEU': 5 }
    
    # Params for PaperCombinedFeatureExtractor to get 162 1D-features
    # ZCR(1) + Chroma(12) + MFCC(13) + RMS(1) + MelSpec(135) = 162
    N_MFCC_1D_TEST = 13
    N_MELS_FOR_1D_FEAT_TEST = 135
    IMG_H_TEST, IMG_W_TEST = 64, 64
    N_MELS_2D_TEST = 64 # n_mels for the 2D spectrogram before resizing to IMG_H_TEST

    # Create dummy data directory and files if they don't exist
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Test data directory {TEST_DATA_DIR} does not exist. Creating it.")
        os.makedirs(TEST_DATA_DIR, exist_ok=True)

    if not os.listdir(TEST_DATA_DIR):
        print(f"Test data directory {TEST_DATA_DIR} is empty. Creating dummy .wav files for testing.")
        dummy_files = [
            "1001_AAA_SAD_XX.wav", "1002_BBB_ANG_XX.wav", "1003_CCC_DIS_XX.wav",
            "1004_DDD_FEA_XX.wav", "1005_EEE_HAP_XX.wav", "1006_FFF_NEU_XX.wav",
            "1007_HHH_SAD_YY.wav", "1008_III_ANG_ZZ.wav" # Ensure at least batch_size files
        ]
        for fname in dummy_files:
            try:
                waveform_dummy = torch.zeros((1, TARGET_SR_TEST // 10)) # 0.1 sec silent audio
                torchaudio.save(os.path.join(TEST_DATA_DIR, fname), waveform_dummy, TARGET_SR_TEST)
            except Exception as e:
                print(f"Could not create dummy file {fname}: {e}. Ensure sox is installed or torchaudio backend is set.")
                print("Skipping DataLoader test.")
                exit()
        print(f"Created {len(dummy_files)} dummy files in {TEST_DATA_DIR}")

    print(f"Attempting to load data from: {os.path.abspath(TEST_DATA_DIR)}")

    # Instantiate preprocessors
    audio_proc = AudioPreprocessor(target_sample_rate=TARGET_SR_TEST, vad_mode=0)
    feature_ext = PaperCombinedFeatureExtractor(
        sr=TARGET_SR_TEST,
        n_mfcc_1d=N_MFCC_1D_TEST,
        n_mels_for_1d_feat=N_MELS_FOR_1D_FEAT_TEST,
        img_height=IMG_H_TEST, 
        img_width=IMG_W_TEST, 
        n_mels_2d=N_MELS_2D_TEST,
        fmax_spec_img= TARGET_SR_TEST // 2 # Use Nyquist for fmax if not specified like 10kHz
    )

    try:
        test_dataloader = get_data_loader(
            data_dir=TEST_DATA_DIR,
            emotion_labels_map=EMOTION_LABELS_TEST,
            batch_size=2,
            audio_preprocessor=audio_proc,
            feature_extractor=feature_ext,
            target_sample_rate=TARGET_SR_TEST, # pass for collate_fn safety
            shuffle=False, # No shuffle for deterministic test
            num_workers=0 # Often better for debugging data loading
        )
        
        print(f"Successfully created DataLoader with {len(test_dataloader.dataset)} samples.")
        
        for i, batch_data in enumerate(test_dataloader):
            feats_1d, feats_2d, lbls, fnames = batch_data
            print(f"Batch {i+1}:")
            print(f"  1D Features shape: {feats_1d.shape}") # Expected: [batch_size, 1, 162]
            print(f"  2D Features shape: {feats_2d.shape}") # Expected: [batch_size, 1, IMG_H_TEST, IMG_W_TEST]
            print(f"  Labels: {lbls}")
            print(f"  File names: {fnames}")
            
            assert feats_1d.shape == (test_dataloader.batch_size, 1, 162)
            assert feats_2d.shape == (test_dataloader.batch_size, 1, IMG_H_TEST, IMG_W_TEST)
            assert lbls.shape == (test_dataloader.batch_size,)
            
            if i == 0: # Print details for the first batch only and break
                break
        print("DataLoader test completed successfully.")
        
    except Exception as e:
        print(f"Error during DataLoader test: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure TEST_DATA_DIR exists and contains valid .wav files or that dummy file creation succeeded.")
        print("If you see import errors for preprocessing modules, check your PYTHONPATH or run from project root.") 