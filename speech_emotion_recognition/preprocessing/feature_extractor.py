import torch
import librosa
import numpy as np
import skimage.transform
import torchaudio.transforms as T

# --- Feature Extractor for the Paper's Combined Model ---
class PaperCombinedFeatureExtractor:
    def __init__(self, sr=16000, 
                 # Params for 1D features
                 n_fft_1d=512, hop_length_1d=160, 
                 n_mfcc_1d=13, n_mels_for_1d_feat=135,
                 # Params for 2D image features
                 n_fft_2d=1024, hop_length_2d=256, n_mels_2d=64, 
                 img_height=64, img_width=64, log_spec_img=True, fmax_spec_img=10000):
        
        self.sr = sr
        # 1D feature params
        self.n_fft_1d = n_fft_1d
        self.hop_length_1d = hop_length_1d
        self.n_mfcc_1d = n_mfcc_1d
        self.n_mels_for_1d_feat = n_mels_for_1d_feat # To make total 162 features

        # 2D image feature params
        self.n_fft_2d = n_fft_2d
        self.hop_length_2d = hop_length_2d
        self.n_mels_2d = n_mels_2d
        self.img_height = img_height
        self.img_width = img_width
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
        mel_spec_2d = self.mel_spectrogram_transform_2d(waveform_tensor.squeeze(0)) # Expects [T]

        if self.log_spec_img:
            mel_spec_2d = torch.log(mel_spec_2d + 1e-6)

        # Normalize to [0, 1] - this is a common practice for images
        # Paper: "Scale pixel values from [0, 255] to [0, 1]" - implies initial image was uint8
        # Here, we have float tensor. Min-max scale to [0,1]
        min_val = mel_spec_2d.min()
        max_val = mel_spec_2d.max()
        if max_val > min_val:
            mel_spec_2d = (mel_spec_2d - min_val) / (max_val - min_val)
        else: # Avoid division by zero if flat
            mel_spec_2d = torch.zeros_like(mel_spec_2d)
            
        # Resize to target IMG_HEIGHT x IMG_WIDTH (e.g., 64x64)
        # Convert to numpy for skimage.transform.resize
        # Input for resize: (H, W) or (H, W, C)
        mel_spec_np = mel_spec_2d.cpu().numpy() # Shape (n_mels_2d, time_frames)
        
        # skimage.transform.resize expects image_ndim >= channel_ndim.
        # (H, W) is fine. Output is float64 by default.
        resized_spec_np = skimage.transform.resize(
            mel_spec_np, 
            (self.img_height, self.img_width), 
            anti_aliasing=True, 
            mode='reflect' # or other mode like 'edge', 'constant'
        )
        
        # Add channel dimension: (1, H, W) for PyTorch CNNs
        resized_spec_tensor = torch.tensor(resized_spec_np, dtype=torch.float32).unsqueeze(0)
        return resized_spec_tensor # Shape [1, img_height, img_width]

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
                - batch_features_2d: (batch_size, 1, img_height, img_width)
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

    paper_feat_extractor = PaperCombinedFeatureExtractor(
        sr=sr_test,
        n_mfcc_1d=13, 
        n_mels_for_1d_feat=135,
        img_height=64, img_width=64, n_mels_2d=64, fmax_spec_img=8000
    )

    features_1d_batch, features_2d_batch = paper_feat_extractor(dummy_waveforms_batch_tensor)

    print(f"Input waveform batch shape: {dummy_waveforms_batch_tensor.shape}")
    print(f"Output 1D features batch shape: {features_1d_batch.shape}") # Expected: [batch_s, 1, 162]
    print(f"Output 2D features batch shape: {features_2d_batch.shape}") # Expected: [batch_s, 1, 64, 64]

    assert features_1d_batch.shape == (batch_s, 27, 301), f"Error in 1D feature shape! Got {features_1d_batch.shape}"
    assert features_2d_batch.shape == (batch_s, 1, 64, 64), f"Error in 2D feature shape! Got {features_2d_batch.shape}"
    print("PaperCombinedFeatureExtractor test completed.")