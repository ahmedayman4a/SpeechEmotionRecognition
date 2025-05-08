import torch
import os
import shutil

def save_checkpoint(state, is_best, directory="trained_models", filename="checkpoint.pth.tar", best_filename="model_best.pth.tar"):
    """Saves model and training parameters at checkpoint + 'best.pth.tar'"""
    filepath = os.path.join(directory, filename)
    best_filepath = os.path.join(directory, best_filename)
    os.makedirs(directory, exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_filepath)
        print(f"Saved new best model to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """Loads model parameters (state_dict) from file_path. Optionally loads optimizer and scheduler state."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Starting from scratch.")
        return 0, float('inf') # Return start epoch 0, best val loss infinity

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state, handling potential DataParallel prefix
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Optimizer state will be reset.")

    if scheduler is not None and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Loaded scheduler state.")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}. Scheduler state will be reset.")


    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch+1} with best_val_loss {best_val_loss:.4f}")
    return start_epoch, best_val_loss 