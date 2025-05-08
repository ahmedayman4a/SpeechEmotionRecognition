import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split # Added for splitting data

# Project specific imports - assuming this script is run from the project root 
# or the speech_emotion_recognition package is in PYTHONPATH.
from speech_emotion_recognition.config import config
from speech_emotion_recognition.data_loader.dataset import get_data_loader
from speech_emotion_recognition.preprocessing.audio_preprocessor import AudioPreprocessor
from speech_emotion_recognition.preprocessing.feature_extractor import PaperCombinedFeatureExtractor
from speech_emotion_recognition.models.combined_model import CombinedModel
# from speech_emotion_recognition.utils.utils import set_seed, save_checkpoint, load_checkpoint # TODO: Implement utils

def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, num_epochs,
                 model_save_dir, model_name="combined_ser_model.pth"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.model_save_path = os.path.join(model_save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)

        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (features_1d, features_2d, labels, _) in enumerate(progress_bar):
            features_1d = features_1d.to(self.device)
            features_2d = features_2d.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features_1d, features_2d)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.2f}")

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch_idx, (features_1d, features_2d, labels, _) in enumerate(progress_bar):
                features_1d = features_1d.to(self.device)
                features_2d = features_2d.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features_1d, features_2d)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                progress_bar.set_postfix(loss=loss.item(), acc=f"{(predicted == labels).sum().item()/labels.size(0):.2f}")

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def train(self):
        print(f"Starting training for {self.num_epochs} epochs on device: {self.device}")
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            print(f"Epoch {epoch+1}/{self.num_epochs}:\n" \
                  f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n" \
                  f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"  New best model saved to {self.model_save_path} (Val Loss: {self.best_val_loss:.4f})")
        print("Training finished.")

def parse_label_from_filename(filename, emotion_map):
    """Helper function to extract label from CREMA-D filename."""
    try:
        parts = os.path.basename(filename).split('_')
        if len(parts) > 2:
            emotion_str = parts[2]
            if emotion_str in emotion_map:
                return emotion_map[emotion_str]
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
    return emotion_map.get('NEU', 0) # Default to NEU if parsing fails or emotion unknown

def main():
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to the dataset directory (default: {config.DATA_DIR} from config.py)')
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the data_dir from command line arguments or config
    current_data_dir = args.data_dir

    # --- Data Splitting --- 
    print(f"Scanning data directory: {current_data_dir}")
    all_wav_files = [os.path.join(current_data_dir, f) 
                     for f in os.listdir(current_data_dir) 
                     if f.endswith('.wav')]
    
    if not all_wav_files:
        print(f"Error: No .wav files found in {current_data_dir}. Please check the path.")
        return

    print(f"Found {len(all_wav_files)} .wav files.")
    
    # Extract labels for stratification
    all_labels = [parse_label_from_filename(f, config.EMOTION_LABELS) for f in all_wav_files]
    
    # First split: 70% Train+Val, 30% Test
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_wav_files, all_labels, 
        test_size=0.30, 
        random_state=config.RANDOM_SEED, 
        stratify=all_labels
    )
    
    # Second split: Train vs Validation (from the 70% pool)
    # Validation is 5% OF THE TRAIN_VAL SET
    val_split_proportion = 0.05 # 5% of the train_val data for validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=val_split_proportion, 
        random_state=config.RANDOM_SEED, 
        stratify=train_val_labels
    )

    print(f"Total files: {len(all_wav_files)}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    # --- End Data Splitting ---

    # 1. Initialize AudioPreprocessor and FeatureExtractor
    audio_preprocessor = AudioPreprocessor(
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
        img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH, 
        log_spec_img=config.LOG_SPECTROGRAM_IMG, fmax_spec_img=config.UPPER_FREQ_LIMIT_KHZ * 1000 if config.UPPER_FREQ_LIMIT_KHZ else None
    )

    # 2. Create DataLoaders using the split file lists and labels
    print("Preparing DataLoaders...")
    train_loader = get_data_loader(
        file_paths_list=train_files, 
        labels_list=train_labels,
        emotion_labels_map=config.EMOTION_LABELS,
        batch_size=config.BATCH_SIZE, 
        audio_preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor,
        target_sample_rate=config.TARGET_SAMPLE_RATE,
        shuffle=config.SHUFFLE_TRAIN, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    val_loader = get_data_loader(
        file_paths_list=val_files, 
        labels_list=val_labels,
        emotion_labels_map=config.EMOTION_LABELS,
        batch_size=config.BATCH_SIZE, 
        audio_preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor,
        target_sample_rate=config.TARGET_SAMPLE_RATE,
        shuffle=config.SHUFFLE_VAL, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    print("DataLoaders created.")
    
    # Test loader can be created similarly if needed for final evaluation
    # test_loader = get_data_loader(file_paths_list=test_files, labels_list=test_labels, ...)

    # 3. Initialize Model
    # Determine activation functions based on config or use defaults
    # For this example, we'll use ReLU as per paper summary, which is default in CombinedModel
    # If SiLU is desired, pass activation_module_cnn=nn.SiLU(), activation_module_mlp=nn.SiLU()
    model = CombinedModel(
        num_classes=config.NUM_CLASSES,
        cnn1d_input_channels=config.CNN1D_INPUT_CHANNELS,
        cnn1d_num_features_input_dim=config.CNN1D_NUM_FEATURES_DIM,
        cnn2d_input_channels=config.CNN2D_INPUT_CHANNELS,
        cnn2d_img_height=config.CNN2D_IMG_HEIGHT,
        cnn2d_img_width=config.CNN2D_IMG_WIDTH,
        dropout_rate_cnn=config.CNN_DROPOUT_RATE,
        dropout_rate_mlp=config.MLP_DROPOUT_RATE
        # activation_module_cnn=nn.SiLU(), # Example for SiLU
        # activation_module_mlp=nn.SiLU()  # Example for SiLU
    ).to(device)
    print("Model initialized.")

    # 4. Optimizer and Loss Function
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
    
    criterion = nn.CrossEntropyLoss()
    print(f"Optimizer: {config.OPTIMIZER}, Loss: CrossEntropyLoss")

    # 5. Training
    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device, 
                      config.NUM_EPOCHS, config.MODEL_SAVE_DIR)
    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 