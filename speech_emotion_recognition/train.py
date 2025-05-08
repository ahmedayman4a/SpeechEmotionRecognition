import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

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

def main():
    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        n_mfcc_1d=config.N_MFCC, n_mels_for_1d_feat=config.N_MELS_FOR_1D_FEAT, # Ensure these sum up for 162 for Path A
        n_fft_2d=config.N_FFT_IMG, hop_length_2d=config.HOP_LENGTH_IMG, n_mels_2d=config.N_MELS_IMG,
        img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH, 
        log_spec_img=config.LOG_SPECTROGRAM_IMG, fmax_spec_img=config.UPPER_FREQ_LIMIT_KHZ * 1000 if config.UPPER_FREQ_LIMIT_KHZ else None
    )

    # 2. Create DataLoaders (assuming a train/val split of data_dir is handled externally or use subset)
    # For simplicity, using the same data_dir for train and val here.
    # In a real scenario, you'd split your data into train and validation sets.
    print("Preparing DataLoader(s)...")
    # TODO: Implement proper train/validation split. For now, using full dataset for both.
    # You might need to prepare separate train_dir and val_dir.
    # As a quick test, one could use a subset for validation if a split is not ready.
    # For now, let's assume config.DATA_DIR is the training data and we'll use it as val too.
    
    # Adjust N_MELS_FOR_1D_FEAT in config.py if needed to ensure 1D features are 162
    # e.g. 1 (ZCR) + 12 (Chroma) + 13 (MFCC) + 1 (RMS) + X (MelSpec) = 162 => X = 135
    # If config.N_MELS_FOR_1D_FEAT is not 135, the 1D features might not be 162.
    # This should be checked against the PaperCombinedFeatureExtractor's logic.
    
    # Check for data directory
    if not os.path.isdir(config.DATA_DIR):
        print(f"Error: Data directory {config.DATA_DIR} not found. Please check config.py.")
        print("As a placeholder, a dummy dataset might be created by data_loader/dataset.py if run directly with its __main__.")
        print("For this train.py script to work, the actual data directory is expected.")
        return
    if not os.listdir(config.DATA_DIR):
        print(f"Error: Data directory {config.DATA_DIR} is empty. Please populate it with .wav files.")
        return
        
    train_loader = get_data_loader(
        data_dir=config.DATA_DIR, 
        emotion_labels_map=config.EMOTION_LABELS,
        batch_size=config.BATCH_SIZE, 
        audio_preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor,
        target_sample_rate=config.TARGET_SAMPLE_RATE,
        shuffle=config.SHUFFLE_TRAIN, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    # Create a validation loader - using the same data for now, ideally a separate split
    val_loader = get_data_loader(
        data_dir=config.DATA_DIR, # Replace with actual validation data path
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