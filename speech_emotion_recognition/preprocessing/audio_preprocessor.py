import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import webrtcvad

class AudioPreprocessor:
    def __init__(self, target_sample_rate=16000, frame_ms=30, vad_mode=0, normalize_audio=True):
        self.target_sample_rate = target_sample_rate
        self.frame_ms = frame_ms
        self.vad_mode = vad_mode # 0: no VAD, 1-3: webrtcvad aggressiveness
        self.normalize_audio = normalize_audio
        self.resamplers = {}

    def get_resampler(self, orig_freq):
        if orig_freq not in self.resamplers:
            self.resamplers[orig_freq] = Resample(orig_freq=orig_freq, new_freq=self.target_sample_rate)
        return self.resamplers[orig_freq]

    def convert_to_16bit_mono(self, waveform, sample_rate):
        """Converts audio to 16-bit PCM mono if not already.
           Assumes waveform is a PyTorch tensor.
        """
        if waveform.shape[0] > 1: # If stereo, convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize to [-1, 1] if it's not already (e.g. from float32 to int16 range)
        # torchaudio.load loads as float32 in [-1, 1] by default.
        # If it were, for example, int16, we'd need to divide by 32768.0
        # For webrtcvad, data needs to be 16-bit PCM bytes.
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = self.get_resampler(sample_rate)
            waveform = resampler(waveform)
            current_sample_rate = self.target_sample_rate
        else:
            current_sample_rate = sample_rate
            
        # Convert to 16-bit integer values for VAD
        # webrtcvad expects mono 16-bit PCM audio data.
        # waveform is [-1.0, 1.0], needs to be scaled to [-32768, 32767]
        audio_int16 = (waveform * 32767).to(torch.int16)
        return audio_int16, current_sample_rate

    def vad_filter_waveform(self, waveform, sample_rate):
        if not webrtcvad or self.vad_mode == 0:
            return waveform # Return original waveform if VAD is disabled or unavailable

        if waveform.ndim > 1 and waveform.shape[0] == 1: # Ensure it's [1, T]
            waveform_mono = waveform.squeeze(0) # Get [T]
        elif waveform.ndim == 1: # Already [T]
            waveform_mono = waveform
        else:
            print("Warning: VAD filter expects mono audio. Returning original waveform.")
            return waveform

        # Convert to 16-bit PCM bytes for webrtcvad
        # The audio needs to be in a format webrtcvad understands: mono, 16-bit, 8/16/32 kHz
        # Let's assume target_sample_rate is one of these (e.g., 16000 Hz)
        if sample_rate not in [8000, 16000, 32000, 48000]: # webrtcvad supported rates
             print(f"Warning: VAD filter may not work well with sample rate {sample_rate}. Expected 8k, 16k, 32k, or 48k Hz.")
             # Attempt to resample to a VAD-compatible rate if not already target_sample_rate
             if sample_rate != self.target_sample_rate and self.target_sample_rate in [8000, 16000, 32000, 48000]:
                 resampler_for_vad = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                 waveform_mono = resampler_for_vad(waveform_mono.unsqueeze(0)).squeeze(0)
                 sample_rate = self.target_sample_rate
             else: # If target_sample_rate is also not compatible, VAD cannot be reliably applied
                 return waveform
        
        audio_int16, _ = self.convert_to_16bit_mono(waveform_mono.unsqueeze(0), sample_rate) # Process as [1,T]
        audio_bytes = audio_int16.squeeze(0).numpy().tobytes()
        
        vad = webrtcvad.Vad(self.vad_mode)
        
        frame_length = int(sample_rate * (self.frame_ms / 1000.0))
        num_bytes_per_frame = frame_length * 2 # 16-bit = 2 bytes
        
        voiced_frames_data = []
        for i in range(0, len(audio_bytes) - num_bytes_per_frame + 1, num_bytes_per_frame):
            frame = audio_bytes[i:i+num_bytes_per_frame]
            if len(frame) == num_bytes_per_frame: # Ensure full frame
                if vad.is_speech(frame, sample_rate):
                    voiced_frames_data.append(frame)
        
        if not voiced_frames_data:
            return torch.tensor([], dtype=waveform.dtype) # Return empty tensor if no voice activity
            
        voiced_audio_bytes = b''.join(voiced_frames_data)
        voiced_audio_np = np.frombuffer(voiced_audio_bytes, dtype=np.int16)
        
        # Convert back to tensor, normalize from int16 range if needed
        # Ensure the numpy array is writable before converting
        voiced_waveform = torch.from_numpy(voiced_audio_np.copy()).float()
        if voiced_waveform.max() > 1.0: # Heuristic check if it's still int16
            voiced_waveform = voiced_waveform / 32768.0 # Normalize int16 range to [-1, 1]
        
        return voiced_waveform.unsqueeze(0) # Return as [1, T]

    def process_waveform(self, waveform, orig_sample_rate):
        """Process a single audio waveform tensor."""
        # Ensure waveform is at least 2D [channels, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # 1. Resample to target_sample_rate
        if orig_sample_rate != self.target_sample_rate:
            resampler = self.get_resampler(orig_sample_rate)
            waveform = resampler(waveform)
        current_sample_rate = self.target_sample_rate

        # 2. Convert to mono (if not already)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Apply VAD (optional)
        if self.vad_mode > 0 and webrtcvad is not None:
            # VAD expects float waveform in [-1,1] range, and will handle conversion to int16 internally
            filtered_waveform = self.vad_filter_waveform(waveform.clone(), current_sample_rate) # Use clone to avoid modifying original
            if filtered_waveform.numel() > 0: # Check if VAD returned non-empty audio
                waveform = filtered_waveform
            else:
                # print("Warning: VAD resulted in empty audio. Using original waveform.")
                pass # Keep original if VAD removes everything

        # 4. Normalize audio (optional)
        if self.normalize_audio:
            if waveform.numel() > 0: # Check if waveform is not empty
                waveform_mean = waveform.mean()
                waveform_std = waveform.std()
                if waveform_std > 1e-6: # Avoid division by zero or very small std
                    waveform = (waveform - waveform_mean) / waveform_std
                else:
                    waveform = waveform - waveform_mean # Just center if std is too small
            # else: waveform is empty, do nothing

        return waveform, current_sample_rate

    def __call__(self, batch):
        """Process a batch of audio data. 
           Input batch: list of (waveform_tensor, orig_sample_rate) tuples
        """
        processed_waveforms_for_padding = [] # List of [Time_i, Channels]
        processed_sample_rates = []
        
        for waveform, orig_sr in batch:
            # process_waveform returns [Channels, Time_processed], e.g., [1, T_i]
            processed_wf, processed_sr = self.process_waveform(waveform, orig_sr) 
            
            # Transpose to [Time_processed, Channels] for pad_sequence
            processed_waveforms_for_padding.append(processed_wf.transpose(0, 1))
            processed_sample_rates.append(processed_sr)
        
        # pad_sequence on list of [Time_i, Channels] with batch_first=True gives [Batch, Time_max, Channels]
        padded_waveforms_transposed = torch.nn.utils.rnn.pad_sequence(
            processed_waveforms_for_padding, 
            batch_first=True, 
            padding_value=0.0
        )
        
        # Transpose back to [Batch, Channels, Time_max]
        padded_waveforms = padded_waveforms_transposed.transpose(1, 2)
        
        return padded_waveforms, torch.tensor(processed_sample_rates, dtype=torch.long)


if __name__ == '__main__':
    # Example Usage and Test
    # Create a dummy waveform (e.g. stereo, 44100 Hz)
    sr_orig = 44100
    duration = 2 # seconds
    dummy_stereo_waveform = torch.sin(2 * torch.pi * torch.arange(0, duration, 1/sr_orig).unsqueeze(0).repeat(2,1) * 440)
    dummy_mono_waveform = torch.mean(dummy_stereo_waveform, dim=0, keepdim=True)
    
    print(f"Original stereo waveform shape: {dummy_stereo_waveform.shape}, SR: {sr_orig}")
    print(f"Original mono waveform shape: {dummy_mono_waveform.shape}, SR: {sr_orig}")

    # Initialize preprocessor
    # Set vad_mode > 0 to test VAD if webrtcvad is installed.
    # On Kaggle, webrtcvad might not be available by default.
    preprocessor_no_vad = AudioPreprocessor(target_sample_rate=16000, vad_mode=0)
    preprocessor_with_vad = AudioPreprocessor(target_sample_rate=16000, vad_mode=1 if webrtcvad else 0)

    # Process single waveforms
    processed_wf_no_vad, new_sr_no_vad = preprocessor_no_vad.process_waveform(dummy_stereo_waveform.clone(), sr_orig)
    print(f"Processed (no VAD) waveform shape: {processed_wf_no_vad.shape}, SR: {new_sr_no_vad}")

    if webrtcvad:
        processed_wf_with_vad, new_sr_with_vad = preprocessor_with_vad.process_waveform(dummy_mono_waveform.clone(), sr_orig)
        print(f"Processed (with VAD) waveform shape: {processed_wf_with_vad.shape}, SR: {new_sr_with_vad}")
    else:
        print("Skipping VAD test as webrtcvad is not available.")

    # Test batch processing
    # Create another dummy waveform, possibly shorter and different original SR
    sr_orig2 = 22050
    duration2 = 1.5
    dummy_mono_waveform2 = torch.cos(2 * torch.pi * torch.arange(0, duration2, 1/sr_orig2).unsqueeze(0) * 220)
    print(f"Second original mono waveform shape: {dummy_mono_waveform2.shape}, SR: {sr_orig2}")

    batch_to_process = [
        (dummy_stereo_waveform.clone(), sr_orig),
        (dummy_mono_waveform2.clone(), sr_orig2)
    ]

    print("\nTesting batch processing (no VAD)...")
    padded_batch_no_vad, batch_srs_no_vad = preprocessor_no_vad(batch_to_process)
    print(f"Padded batch (no VAD) shape: {padded_batch_no_vad.shape}") # Expected: [batch_size, 1, max_time_after_processing]
    print(f"Batch sample rates (no VAD): {batch_srs_no_vad}")

    if webrtcvad:
        print("\nTesting batch processing (with VAD)...")
        # For VAD testing, ensure inputs are mono or handled appropriately if VAD expects mono
        batch_for_vad = [
            (dummy_mono_waveform.clone(), sr_orig),
            (dummy_mono_waveform2.clone(), sr_orig2) # VAD works on mono
        ]
        padded_batch_with_vad, batch_srs_with_vad = preprocessor_with_vad(batch_for_vad)
        print(f"Padded batch (with VAD) shape: {padded_batch_with_vad.shape}")
        print(f"Batch sample rates (with VAD): {batch_srs_with_vad}") 