import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt

def visualize_emotions_audio(data_dir):
    """
    function to visualize waveforms of the first 6 audio samples
    from the dataset.
    
    Args:
        data_dir (str): Path to the CREMA dataset directory
    """    
    # Define emotion labels for reference
    emotion_labels = {'SAD': 'Sad', 'ANG': 'Angry', 'DIS': 'Disgust', 
                    'FEA': 'Fear', 'HAP': 'Happy', 'NEU': 'Neutral'}
      
    # Create images directory if it doesn't exist
    images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory: {images_dir}")
    
    # Get all audio files in the dataset
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    # Take only first 6 files containing the 6 classes
    files_to_visualize = all_files[:min(6, len(all_files))]
    
    
    for i, file in enumerate(files_to_visualize):
        
        file_path = os.path.join(data_dir, file)
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract emotion from filename (CREMA format: 1001_DFA_ANG_XX.wav)
        parts = file.split('_')
        emotion = parts[2] if len(parts) >= 3 else "Unknown"
        emotion_name = emotion_labels.get(emotion, emotion)
        
        
        plt.figure(figsize=(10, 4))

        librosa.display.waveshow(y, sr=sr)
        title = f"Waveform Visualization - {emotion_name}"
        plt.title(f"{title}\n{file}", fontsize=12)
        plt.tight_layout()
        
        # Save figure with clean filename
        clean_filename = f"{emotion_name.lower()}_waveform_{i+1}.png"
        save_path = os.path.join(images_dir, clean_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
        
        # Show each figure
        plt.show()


def main():
    data_path = r"E:\CSED\Term 6 CSE\Pattern recognition\lab\SpeechEmotionRecognition\speech_emotion_recognition\data"
    visualize_emotions_audio(data_path)


if __name__ == "__main__":
    main()