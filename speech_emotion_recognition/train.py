import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split # Added for splitting data
import wandb # Added
import torchmetrics # Added
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR # Added
import matplotlib.pyplot as plt # Added for confusion matrix plot
import io # Added for logging plot to wandb
import PIL.Image # Added for logging plot to wandb

# Project specific imports - assuming this script is run from the project root 
# or the speech_emotion_recognition package is in PYTHONPATH.
from speech_emotion_recognition.config import config
from speech_emotion_recognition.data_loader.dataset import get_data_loader
from speech_emotion_recognition.preprocessing.audio_preprocessor import AudioPreprocessor
from speech_emotion_recognition.preprocessing.feature_extractor import PaperCombinedFeatureExtractor
from speech_emotion_recognition.models.combined_model import CombinedModel
from speech_emotion_recognition.utils.checkpoint_utils import save_checkpoint, load_checkpoint # Added

def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, # Added test_loader
                 optimizer, scheduler, criterion, device, num_epochs,
                 model_save_dir, wandb_run,
                 start_epoch=0, best_val_loss=float('inf'), 
                 model_name="combined_ser_model.pth"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader # Store test loader
        self.optimizer = optimizer
        self.scheduler = scheduler # Added scheduler
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.model_save_dir = model_save_dir
        # Store only base filenames, directory is handled by save_checkpoint
        self.checkpoint_base_filename = "checkpoint_" + model_name 
        self.best_model_base_filename = "best_" + model_name
        self.wandb_run = wandb_run # Added wandb run instance
        self.start_epoch = start_epoch # Added for resuming
        self.best_val_loss = best_val_loss # Added for resuming
        
        os.makedirs(model_save_dir, exist_ok=True)

        # Initialize metrics using torchmetrics
        num_target_classes = config.NUM_CLASSES # Get from config
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_target_classes),
            'f1': torchmetrics.F1Score(task="multiclass", num_classes=num_target_classes, average='weighted'),
            'precision': torchmetrics.Precision(task="multiclass", num_classes=num_target_classes, average='weighted'),
            'recall': torchmetrics.Recall(task="multiclass", num_classes=num_target_classes, average='weighted')
        }).to(self.device)

        self.val_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_target_classes),
            'f1': torchmetrics.F1Score(task="multiclass", num_classes=num_target_classes, average='weighted'),
            'precision': torchmetrics.Precision(task="multiclass", num_classes=num_target_classes, average='weighted'),
            'recall': torchmetrics.Recall(task="multiclass", num_classes=num_target_classes, average='weighted')
        }).to(self.device)

        self.test_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=num_target_classes),
            'f1': torchmetrics.F1Score(task="multiclass", num_classes=num_target_classes, average='weighted'),
            'precision': torchmetrics.Precision(task="multiclass", num_classes=num_target_classes, average='weighted'),
            'recall': torchmetrics.Recall(task="multiclass", num_classes=num_target_classes, average='weighted')
        }).to(self.device)
        
        self.conf_matrix_metric = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_target_classes).to(self.device)


    def train_epoch(self):
        self.model.train()
        total_loss = 0
        self.train_metrics.reset()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} Training", leave=False)
        for batch_idx, (features_1d, features_2d, labels, _) in enumerate(progress_bar):
            features_1d = features_1d.to(self.device)
            features_2d = features_2d.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features_1d, features_2d)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step() # Step the scheduler after optimizer step

            # Update metrics
            preds = torch.argmax(outputs, dim=1)
            self.train_metrics.update(preds, labels)
            total_loss += loss.item() # Log raw loss

            # Log to wandb (maybe less frequently for loss/lr)
            # if batch_idx % 50 == 0: # Log every 50 batches
            #     lr = self.scheduler.get_last_lr()[0]
            #     self.wandb_run.log({"train/batch_loss": loss.item(), "train/learning_rate": lr},
            #                        step=(self.current_epoch * len(self.train_loader) + batch_idx))
            
            progress_bar.set_postfix(loss=loss.item(), lr=f"{self.scheduler.get_last_lr()[0]:.1e}")

        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = self.train_metrics.compute()
        return avg_loss, epoch_metrics # Return loss and metrics dict

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        self.val_metrics.reset()
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} Validation", leave=False)
        with torch.no_grad():
            for batch_idx, (features_1d, features_2d, labels, _) in enumerate(progress_bar):
                features_1d = features_1d.to(self.device)
                features_2d = features_2d.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features_1d, features_2d)
                loss = self.criterion(outputs, labels)

                # Update metrics
                preds = torch.argmax(outputs, dim=1)
                self.val_metrics.update(preds, labels)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        epoch_metrics = self.val_metrics.compute()
        return avg_loss, epoch_metrics

    def train(self):
        print(f"Starting training from epoch {self.start_epoch+1} for {self.num_epochs} total epochs on device: {self.device}")
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch # For logging steps
            train_loss, train_metrics_computed = self.train_epoch()
            val_loss, val_metrics_computed = self.validate_epoch()

            # Prepare log dictionary for wandb
            log_dict = {
                "epoch": epoch + 1,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
            }
            # Add computed metrics
            for name, value in train_metrics_computed.items():
                log_dict[f"train/{name}"] = value.item() # .item() to get scalar value
            for name, value in val_metrics_computed.items():
                log_dict[f"val/{name}"] = value.item()

            self.wandb_run.log(log_dict, step=epoch + 1) # Log per epoch

            print(f"Epoch {epoch+1}/{self.num_epochs}:\n" \
                  f"  Train Loss: {train_loss:.4f}, Train Acc: {train_metrics_computed['accuracy']:.4f}, Train F1: {train_metrics_computed['f1']:.4f}\n" \
                  f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_metrics_computed['accuracy']:.4f}, Val F1:   {val_metrics_computed['f1']:.4f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  New best validation loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
            }, is_best, 
               directory=self.model_save_dir, 
               filename=self.checkpoint_base_filename, # Pass base filename 
               best_filename=self.best_model_base_filename) # Pass base filename

        print("Training finished.")

    def test(self):
        """Evaluate the model on the test set after loading the best checkpoint."""
        print("Starting testing phase...")
        # Load best model weights
        best_model_full_path = os.path.join(self.model_save_dir, self.best_model_base_filename)
        if not os.path.exists(best_model_full_path):
            print(f"Error: Best model checkpoint not found at {best_model_full_path}. Cannot run test.")
            return
        try:
            print(f"Loading best model from: {best_model_full_path}")
            checkpoint = torch.load(best_model_full_path, map_location=self.device)
            state_dict = checkpoint['state_dict']
            if list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            print("Best model loaded successfully.")
        except Exception as e:
            print(f"Error loading best model state_dict: {e}. Cannot run test.")
            return
        
        self.model.eval()
        total_loss = 0
        self.test_metrics.reset()
        self.conf_matrix_metric.reset()
        
        progress_bar = tqdm(self.test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for batch_idx, (features_1d, features_2d, labels, _) in enumerate(progress_bar):
                features_1d = features_1d.to(self.device)
                features_2d = features_2d.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features_1d, features_2d)
                loss = self.criterion(outputs, labels)

                # Update metrics
                preds = torch.argmax(outputs, dim=1)
                self.test_metrics.update(preds, labels)
                self.conf_matrix_metric.update(preds, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        final_metrics = self.test_metrics.compute()
        conf_matrix = self.conf_matrix_metric.compute().cpu().numpy() # Get confusion matrix as numpy array
        
        print("--- Test Results ---")
        print(f"Test Loss: {avg_loss:.4f}")
        for name, value in final_metrics.items():
            print(f"Test {name.capitalize()}: {value.item():.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        # Log final test metrics to wandb
        log_dict = {"test/final_loss": avg_loss}
        for name, value in final_metrics.items():
            log_dict[f"test/final_{name}"] = value.item()
        
        # Log confusion matrix plot to wandb
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            cm_display = self.conf_matrix_metric.plot(ax=ax)
            # Use class names if available
            class_names = list(config.EMOTION_LABELS.keys()) # Ensure order matches label indices
            if len(class_names) == config.NUM_CLASSES:
                 ax.set_xticks(np.arange(len(class_names)))
                 ax.set_yticks(np.arange(len(class_names)))
                 ax.set_xticklabels(class_names, rotation=45, ha='right')
                 ax.set_yticklabels(class_names)
            ax.set_title("Test Set Confusion Matrix")
            plt.tight_layout()
            
            # Save plot to a buffer and log as wandb.Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = PIL.Image.open(buf)
            log_dict["test/confusion_matrix"] = wandb.Image(image)
            plt.close(fig) # Close the plot to free memory
            print("Logged confusion matrix plot to wandb.")
        except Exception as plot_err:
             print(f"Error generating/logging confusion matrix plot: {plot_err}")
             log_dict["test/confusion_matrix_raw"] = wandb.Table(data=conf_matrix.tolist(), columns=class_names, index=class_names)

        self.wandb_run.log(log_dict)
        print("Testing finished.")

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

def get_scheduler(optimizer, warmup_epochs, max_epochs, steps_per_epoch):
    """Creates a SequentialLR scheduler: Linear Warmup -> Cosine Annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    main_steps = (max_epochs - warmup_epochs) * steps_per_epoch
    
    # Linear Warmup
    def warmup_lambda(current_step):
        return float(current_step) / float(max(1, warmup_steps))
    
    # Cosine Annealing requires T_max in steps
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=config.LEARNING_RATE * config.MIN_LR_FACTOR)
    
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    return scheduler

def main():
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to the dataset directory (default: {config.DATA_DIR} from config.py)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from (e.g., trained_models/checkpoint_...)')
    parser.add_argument('--wandb_project', type=str, default="SpeechEmotionRecognition",
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name (defaults to auto-generated)')
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Weights & Biases ---
    # Create a dictionary of uppercase config variables for wandb logging
    wandb_config_dict = {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.startswith('__')}

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name, # Optional: specify a run name
        config=wandb_config_dict # Log filtered configuration parameters
    )
    # Add command line args to wandb config as well (filter out None values if desired)
    wandb.config.update({k: v for k, v in vars(args).items() if v is not None})
    print(f"Wandb initialized. Run page: {wandb_run.get_url()}")

    # Use the data_dir from command line arguments or config
    current_data_dir = args.data_dir

    # --- Data Splitting --- 
    print(f"Scanning data directory: {current_data_dir}")
    all_wav_files = [os.path.join(current_data_dir, f) 
                     for f in os.listdir(current_data_dir) 
                     if f.endswith('.wav')]
    
    if not all_wav_files:
        print(f"Error: No .wav files found in {current_data_dir}. Please check the path.")
        wandb.finish() # Ensure wandb run is closed
        return

    print(f"Found {len(all_wav_files)} .wav files.")
    all_labels = [parse_label_from_filename(f, config.EMOTION_LABELS) for f in all_wav_files]
    
    # Train/Val/Test Split
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_wav_files, all_labels, test_size=0.30, random_state=config.RANDOM_SEED, stratify=all_labels
    )
    val_split_proportion = 0.05 
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_split_proportion, random_state=config.RANDOM_SEED, stratify=train_val_labels
    )
    print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # --- Preprocessors --- 
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

    # --- DataLoaders --- 
    print("Preparing DataLoaders...")
    train_loader = get_data_loader(
        file_paths_list=train_files, labels_list=train_labels,
        emotion_labels_map=config.EMOTION_LABELS,
        batch_size=config.BATCH_SIZE, audio_preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor, target_sample_rate=config.TARGET_SAMPLE_RATE,
        shuffle=config.SHUFFLE_TRAIN, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    val_loader = get_data_loader(
        file_paths_list=val_files, labels_list=val_labels,
        emotion_labels_map=config.EMOTION_LABELS,
        batch_size=config.BATCH_SIZE, audio_preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor, target_sample_rate=config.TARGET_SAMPLE_RATE,
        shuffle=config.SHUFFLE_VAL, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    test_loader = get_data_loader(
        file_paths_list=test_files, labels_list=test_labels,
        emotion_labels_map=config.EMOTION_LABELS,
        batch_size=config.BATCH_SIZE, audio_preprocessor=audio_preprocessor, 
        feature_extractor=feature_extractor, target_sample_rate=config.TARGET_SAMPLE_RATE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY # No shuffle for test
    )
    print("DataLoaders created.")

    # --- Model --- 
    model = CombinedModel(
        num_classes=config.NUM_CLASSES,
        model_type=config.MODEL_TYPE,
        cnn1d_input_channels=config.CNN1D_INPUT_CHANNELS,
        cnn2d_input_channels=config.CNN2D_INPUT_CHANNELS,
        dropout_rate_cnn=config.CNN_DROPOUT_RATE,
        dropout_rate_mlp=config.MLP_DROPOUT_RATE,
        activation_module_cnn=nn.SiLU,
        activation_module_mlp=nn.SiLU
    ).to(device)
    print("Model initialized.")
    wandb.watch(model, log='all', log_freq=100) # Log model gradients/parameters

    # --- Optimizer & Loss --- 
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
    criterion = nn.CrossEntropyLoss()
    print(f"Optimizer: {config.OPTIMIZER}, Loss: CrossEntropyLoss")
    
    # --- Scheduler --- 
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, 
                              warmup_epochs=config.WARMUP_EPOCHS, 
                              max_epochs=config.NUM_EPOCHS, 
                              steps_per_epoch=steps_per_epoch)
    print(f"Scheduler: Linear warmup ({config.WARMUP_EPOCHS} epochs) -> Cosine Annealing")

    # --- Load Checkpoint if resuming ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            start_epoch, best_val_loss = load_checkpoint(
                args.resume_checkpoint, model, optimizer, scheduler, device
            )
        else:
            print(f"Warning: Resume checkpoint specified but not found: {args.resume_checkpoint}")
    
    # --- Training --- 
    trainer = Trainer(model, train_loader, val_loader, test_loader, 
                      optimizer, scheduler, criterion, device, config.NUM_EPOCHS,
                      config.MODEL_SAVE_DIR, wandb_run, 
                      start_epoch=start_epoch, best_val_loss=best_val_loss)
    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        
    # --- Final Testing ---
    trainer.test() # Evaluate on the test set using the best model saved
    
    wandb.finish() # Finalize wandb run
    print("Script finished.")

if __name__ == '__main__':
    main() 