import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from audiomentations import Compose

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
                 transform_audio_func=None,
                 augmentations: Compose = None):
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
        self.augmentations = augmentations
        
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
            
        if self.augmentations:
            wf_np = waveform.squeeze().numpy()
            augmented = self.augmentations(samples=wf_np, sample_rate=sample_rate)
            waveform = torch.from_numpy(augmented).unsqueeze(0)

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

        batch_for_preprocessing = []
        for wf, sr in zip(waveforms_orig, sample_rates_orig):
            if not isinstance(wf, torch.Tensor): 
                wf = torch.zeros((1, target_sample_rate))
            batch_for_preprocessing.append((wf, sr))

        preprocessed_waveforms_batch, _ = audio_preprocessor(batch_for_preprocessing)
        batch_features_1d, batch_features_2d = feature_extractor(preprocessed_waveforms_batch)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return batch_features_1d, batch_features_2d, labels_tensor, list(file_names)

    return collate_fn


def get_data_loader(file_paths_list: list, 
                    labels_list: list,
                    emotion_labels_map: dict,
                    batch_size: int, 
                    audio_preprocessor: AudioPreprocessor, 
                    feature_extractor: PaperCombinedFeatureExtractor,
                    target_sample_rate: int,
                    shuffle=True, num_workers=4, pin_memory=True,
                    augmentations=None):
    
    dataset = CremaDataset(
        file_paths_list=file_paths_list,
        labels_list=labels_list,
        emotion_labels_map=emotion_labels_map,
        augmentations=augmentations
    )
    
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
    # Example usage - this test needs to be updated to reflect the new Dataset structure
    print("--- Testing CremaDataset and get_data_loader with pre-defined file lists ---")
    # --- Configuration (mirroring parts of config.py for this test) ---
    TEST_DATA_ROOT_DIR = "../../data/Crema"  # Root directory of data
    TARGET_SR_TEST = 16000
    EMOTION_LABELS_TEST = { 'SAD': 0, 'ANG': 1, 'DIS': 2, 'FEA': 3, 'HAP': 4, 'NEU': 5 }
    
    N_MFCC_1D_TEST = 13
    N_MELS_FOR_1D_FEAT_TEST = 135
    IMG_H_TEST, IMG_W_TEST = 64, 64
    N_MELS_2D_TEST = 64

    # Create dummy data directory and files if they don't exist
    if not os.path.exists(TEST_DATA_ROOT_DIR):
        print(f"Test data directory {TEST_DATA_ROOT_DIR} does not exist. Creating it.")
        os.makedirs(TEST_DATA_ROOT_DIR, exist_ok=True)

    dummy_files_info = [
        ("1001_AAA_SAD_XX.wav", EMOTION_LABELS_TEST['SAD']),
        ("1002_BBB_ANG_XX.wav", EMOTION_LABELS_TEST['ANG']),
        ("1003_CCC_DIS_XX.wav", EMOTION_LABELS_TEST['DIS']),
        ("1004_DDD_FEA_XX.wav", EMOTION_LABELS_TEST['FEA']),
    ]
    
    # Ensure enough dummy files exist for the test
    if not all(os.path.exists(os.path.join(TEST_DATA_ROOT_DIR, fname)) for fname, _ in dummy_files_info):
        print(f"Test data directory {TEST_DATA_ROOT_DIR} is missing some files. Recreating dummy .wav files.")
        for fname, _ in dummy_files_info:
            fpath = os.path.join(TEST_DATA_ROOT_DIR, fname)
            try:
                waveform_dummy = torch.zeros((1, TARGET_SR_TEST // 10))
                torchaudio.save(fpath, waveform_dummy, TARGET_SR_TEST)
            except Exception as e:
                print(f"Could not create dummy file {fpath}: {e}.")
                exit()
        print(f"Created/Verified {len(dummy_files_info)} dummy files in {TEST_DATA_ROOT_DIR}")

    test_file_paths = [os.path.join(TEST_DATA_ROOT_DIR, fname) for fname, _ in dummy_files_info]
    test_labels = [label for _, label in dummy_files_info]

    print(f"Attempting to load data using predefined lists from: {os.path.abspath(TEST_DATA_ROOT_DIR)}")

    audio_proc = AudioPreprocessor(target_sample_rate=TARGET_SR_TEST, vad_mode=0)
    feature_ext = PaperCombinedFeatureExtractor(
        sr=TARGET_SR_TEST, n_mfcc_1d=N_MFCC_1D_TEST, n_mels_for_1d_feat=N_MELS_FOR_1D_FEAT_TEST,
        img_height=IMG_H_TEST, img_width=IMG_W_TEST, n_mels_2d=N_MELS_2D_TEST,
        fmax_spec_img= TARGET_SR_TEST // 2
    )

    try:
        test_dataloader = get_data_loader(
            file_paths_list=test_file_paths,
            labels_list=test_labels,
            emotion_labels_map=EMOTION_LABELS_TEST,
            batch_size=2,
            audio_preprocessor=audio_proc,
            feature_extractor=feature_ext,
            target_sample_rate=TARGET_SR_TEST,
            shuffle=False, num_workers=0
        )
        
        print(f"Successfully created DataLoader with {len(test_dataloader.dataset)} samples.")
        
        for i, batch_data in enumerate(test_dataloader):
            feats_1d, feats_2d, lbls, fnames = batch_data
            print(f"Batch {i+1}:")
            print(f"  1D Features shape: {feats_1d.shape}")
            print(f"  2D Features shape: {feats_2d.shape}")
            print(f"  Labels: {lbls}")
            print(f"  File names: {fnames}")
            
            assert feats_1d.shape[0] <= 2 # Batch size
            assert feats_1d.shape[1:] == (1, 162)
            assert feats_2d.shape[0] <= 2
            assert feats_2d.shape[1:] == (1, IMG_H_TEST, IMG_W_TEST)
            assert lbls.shape[0] <= 2
            if i == 0: break
        print("DataLoader test with predefined lists completed successfully.")
        
    except Exception as e:
        print(f"Error during DataLoader test: {e}")
        import traceback
        traceback.print_exc() 