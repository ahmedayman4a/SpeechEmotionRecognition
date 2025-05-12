import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
import skimage.transform

# --- Original Feature Extractor (for reference or other models) ---
class TorchaudioFeatureExtractor:
    def __init__(self, sample_rate=16000, feature_type='mfcc', n_mfcc=40, n_mels=128, 
                 n_fft=400, hop_length=160, log_mels=True, 
                 delta_order=0, delta_window_size=5):
        self.sample_rate = sample_rate
        self.feature_type = feature_type.lower()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_mels = log_mels
        self.delta_order = delta_order
        self.delta_window_size = delta_window_size

        if self.feature_type == 'melspectrogram':
            self.mel_spectrogram_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0
            )
        elif self.feature_type == 'mfcc':
            self.mfcc_transform = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels,
                    'power': 2.0
                }
            )
        elif self.feature_type == 'spectrogram':
            self.spectrogram_transform = T.Spectrogram(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=2.0
            )
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}.")

    def _compute_deltas(self, features):
        if self.delta_order >= 1:
            deltas = T.ComputeDeltas(win_length=self.delta_window_size)(features)
            features = torch.cat((features, deltas), dim=-2)
        if self.delta_order >= 2:
            delta_deltas = T.ComputeDeltas(win_length=self.delta_window_size)(deltas) 
            features = torch.cat((features, delta_deltas), dim=-2)
        return features

    def extract_features(self, waveform_batch):
        if waveform_batch.ndim == 3 and waveform_batch.shape[1] == 1:
            waveform_batch = waveform_batch.squeeze(1)
        elif waveform_batch.ndim != 2:
            raise ValueError(f"Expected 2D [batch, time] or 3D [batch, 1, time], got {waveform_batch.shape}")

        if self.feature_type == 'melspectrogram':
            features = self.mel_spectrogram_transform(waveform_batch)
            if self.log_mels:
                features = torch.log(features + 1e-6)
        elif self.feature_type == 'mfcc':
            features = self.mfcc_transform(waveform_batch)
        elif self.feature_type == 'spectrogram':
            features = self.spectrogram_transform(waveform_batch)
            if self.log_mels:
                features = torch.log(features + 1e-6)
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        features = self._compute_deltas(features)
        return features

    def __call__(self, waveform_batch):
        return self.extract_features(waveform_batch)

# --- Feature Extractor for the Paper's Combined Model ---
class PaperCombinedFeatureExtractor:
    def __init__(self, sr=16000, 
                 # Params for 1D features
                 n_fft_1d=512, hop_length_1d=160, 
                 n_mfcc_1d=13, n_mels_for_1d_feat=135, # 1+12+13+1+135 = 162 features
                 # Params for 2D image features
                 n_fft_2d=1024, hop_length_2d=256, n_mels_2d=64, 
                 log_spec_img=True, fmax_spec_img=10000): # Removed img_height, img_width
        
        self.sr = sr
        # 1D feature params
        self.n_fft_1d = n_fft_1d
        self.hop_length_1d = hop_length_1d
        self.n_mfcc_1d = n_mfcc_1d
        self.n_mels_for_1d_feat = n_mels_for_1d_feat # To make total 162 features

        # 2D image feature params
        self.n_fft_2d = n_fft_2d
        self.hop_length_2d = hop_length_2d
        self.n_mels_2d = n_mels_2d # This is the height of the spectrogram
        # self.img_height = img_height # Removed
        # self.img_width = img_width   # Removed
        self.log_spec_img = log_spec_img
        self.fmax_spec_img = fmax_spec_img if fmax_spec_img is not None else sr // 2

        # Torchaudio transform for 2D Mel spectrogram (Path B)
        self.mel_spectrogram_transform_2d = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft_2d,
            hop_length=self.hop_length_2d,
            n_mels=self.n_mels_2d, # This is height before resize
            f_max=self.fmax_spec_img,
            power=2.0
        )

    def _extract_1d_features(self, waveform_np):
        """ Extracts the 162-feature vector using Librosa. Waveform is a mono numpy array. """
        # 1. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(waveform_np, frame_length=self.n_fft_1d, hop_length=self.hop_length_1d)
        zcr_mean = np.mean(zcr, axis=1)
        
        # 2. Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=waveform_np, sr=self.sr, n_fft=self.n_fft_1d, hop_length=self.hop_length_1d)
        chroma_mean = np.mean(chroma_stft, axis=1)
        
        # 3. MFCCs
        mfccs = librosa.feature.mfcc(y=waveform_np, sr=self.sr, n_mfcc=self.n_mfcc_1d, n_fft=self.n_fft_1d, hop_length=self.hop_length_1d)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # 4. RMS Energy
        rms = librosa.feature.rms(y=waveform_np, frame_length=self.n_fft_1d, hop_length=self.hop_length_1d)
        rms_mean = np.mean(rms, axis=1)
        
        # 5. Mel-Spectrogram (values, then mean)
        mel_spec_vals = librosa.feature.melspectrogram(y=waveform_np, sr=self.sr, n_fft=self.n_fft_1d, 
                                                       hop_length=self.hop_length_1d, n_mels=self.n_mels_for_1d_feat)
        mel_spec_mean = np.mean(librosa.power_to_db(mel_spec_vals, ref=np.max), axis=1)
        
        # Concatenate all mean features
        # Expected sizes: zcr (1), chroma (12), mfcc (self.n_mfcc_1d), rms (1), mel_spec (self.n_mels_for_1d_feat)
        features_1d = np.concatenate((zcr_mean, chroma_mean, mfccs_mean, rms_mean, mel_spec_mean), axis=0)
        return torch.tensor(features_1d, dtype=torch.float32).unsqueeze(0) # Shape [1, 162]

    def _extract_2d_image_features(self, waveform_tensor):
        """ Extracts 2D spectrogram image. Waveform is a_i mono tensor [1, T] or [T]. """
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0) # Ensure [1, T]
        
        # Use torchaudio for Mel spectrogram generation
        mel_spec_2d = self.mel_spectrogram_transform_2d(waveform_tensor.squeeze(0)) # Expects [T], output [n_mels_2d, time_frames]

        if self.log_spec_img:
            mel_spec_2d = torch.log(mel_spec_2d + 1e-6)

        # Normalize to [0, 1] - this is a common practice for images
        min_val = mel_spec_2d.min()
        max_val = mel_spec_2d.max()
        if max_val > min_val:
            mel_spec_2d = (mel_spec_2d - min_val) / (max_val - min_val)
        else: # Avoid division by zero if flat
            mel_spec_2d = torch.zeros_like(mel_spec_2d)
            
        # RESIZING REMOVED
        # mel_spec_np = mel_spec_2d.cpu().numpy() 
        # resized_spec_np = skimage.transform.resize(
        #     mel_spec_np, 
        #     (self.img_height, self.img_width), 
        #     anti_aliasing=True, 
        #     mode='reflect'
        # )
        # resized_spec_tensor = torch.tensor(resized_spec_np, dtype=torch.float32).unsqueeze(0)
        # return resized_spec_tensor # Shape [1, img_height, img_width]

        # Add channel dimension: (1, H, W) for PyTorch CNNs
        # H is n_mels_2d, W is variable (time_frames)
        return mel_spec_2d.unsqueeze(0) # Shape [1, n_mels_2d, time_frames]

    def __call__(self, waveform_batch_preprocessed):
        """ 
        Processes a batch of preprocessed audio waveforms.
        Args:
            waveform_batch_preprocessed (torch.Tensor): Batch of preprocessed mono waveforms.
                                                      Shape [batch_size, 1, time] or [batch_size, time].
                                                      Assumes audio is at self.sr.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - batch_features_1d: (batch_size, 1, 162)
                - batch_features_2d: (batch_size, 1, n_mels_2d, variable_time_frames)
        """
        if waveform_batch_preprocessed.ndim == 3 and waveform_batch_preprocessed.shape[1] == 1:
            waveform_batch_mono = waveform_batch_preprocessed.squeeze(1) # [batch_size, time]
        elif waveform_batch_preprocessed.ndim == 2:
            waveform_batch_mono = waveform_batch_preprocessed # [batch_size, time]
        else:
            raise ValueError("Expected waveform_batch to be 2D [B, T] or 3D [B, 1, T]")

        batch_features_1d_list = []
        batch_features_2d_list = []

        for i in range(waveform_batch_mono.shape[0]):
            waveform_single_tensor = waveform_batch_mono[i, :] # [time]
            waveform_single_np = waveform_single_tensor.cpu().numpy()
            
            # Path A: 1D features
            features_1d = self._extract_1d_features(waveform_single_np) # [1, 162]
            batch_features_1d_list.append(features_1d)
            
            # Path B: 2D image features
            # Pass tensor for torchaudio transforms
            features_2d = self._extract_2d_image_features(waveform_single_tensor) # [1, H, W]
            batch_features_2d_list.append(features_2d)
            
        batch_features_1d = torch.stack(batch_features_1d_list, dim=0) # [B, 1, 162]
        batch_features_2d = torch.stack(batch_features_2d_list, dim=0) # [B, 1, H, W]
        
        return batch_features_1d, batch_features_2d

if __name__ == '__main__':
    print("Testing PaperCombinedFeatureExtractor...")
    sr_test = 16000
    batch_s = 2
    duration_s = 3
    dummy_waveforms_batch_tensor = torch.randn(batch_s, sr_test * duration_s) # Batch of mono waveforms

    # Initialize extractor with parameters that should lead to 162 1D features
    # ZCR(1) + Chroma(12) + MFCC(13) + RMS(1) + MelSpec(135) = 162
    paper_feat_extractor = PaperCombinedFeatureExtractor(
        sr=sr_test,
        n_mfcc_1d=13, 
        n_mels_for_1d_feat=135, # This makes it 1+12+13+1+135 = 162
        n_mels_2d=64, # img_height and img_width removed
        fmax_spec_img=8000,
        n_fft_2d = 1024,
        hop_length_2d = 256
    )

    features_1d_batch, features_2d_batch = paper_feat_extractor(dummy_waveforms_batch_tensor)

    print(f"Input waveform batch shape: {dummy_waveforms_batch_tensor.shape}")
    print(f"Output 1D features batch shape: {features_1d_batch.shape}") # Expected: [batch_s, 1, 162]
    print(f"Output 2D features batch shape: {features_2d_batch.shape}") # Expected: [batch_s, 1, 64, VariableWidth]

    assert features_1d_batch.shape == (batch_s, 1, 162), f"Error in 1D feature shape! Got {features_1d_batch.shape}"
    # For 2D, check batch, channels, and height (n_mels_2d). Width is variable.
    assert features_2d_batch.shape[0] == batch_s, f"Error in 2D feature batch size! Got {features_2d_batch.shape[0]}"
    assert features_2d_batch.shape[1] == 1, f"Error in 2D feature channels! Got {features_2d_batch.shape[1]}"
    assert features_2d_batch.shape[2] == 64, f"Error in 2D feature height (n_mels_2d)! Got {features_2d_batch.shape[2]}"
    print("PaperCombinedFeatureExtractor test completed.")

    # Test the original TorchaudioFeatureExtractor (renamed from FeatureExtractor)
    print("\nTesting TorchaudioFeatureExtractor (MFCCs with deltas)...")
    # Example: 20 MFCCs + deltas + delta-deltas = 60 features
    # Adjust n_mels and n_fft here to avoid the warning in the test.
    # n_freqs = 512 // 2 + 1 = 257. n_mels=60 is fine for n_mfcc=20.
    ta_feat_extractor = TorchaudioFeatureExtractor(
        sample_rate=sr_test, 
        feature_type='mfcc', 
        n_mfcc=20, 
        delta_order=2,
        n_mels=60, # Was default 128, changed for test
        n_fft=512  # Was default 400, changed for test
    )
    # Ensure input is [B, T]
    if dummy_waveforms_batch_tensor.ndim == 3 and dummy_waveforms_batch_tensor.shape[1] == 1:
        test_waveforms_for_ta = dummy_waveforms_batch_tensor.squeeze(1)
    else:
        test_waveforms_for_ta = dummy_waveforms_batch_tensor
    
    mfcc_delta_features = ta_feat_extractor(test_waveforms_for_ta)
    print(f"Torchaudio MFCC features shape: {mfcc_delta_features.shape}") # Expected e.g. [batch_s, 60, time_frames]
    assert mfcc_delta_features.shape[0] == batch_s
    assert mfcc_delta_features.shape[1] == 20 * 3 # n_mfcc * (1+delta_order)
    print("TorchaudioFeatureExtractor test completed.") 