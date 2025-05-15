# Speech Emotion Recognition

![Alexandria University](https://img.shields.io/badge/Alexandria%20University-Faculty%20of%20Engineering-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning-based speech emotion recognition system using 1D and 2D CNN architectures on the CREMA-D dataset. This project explores different neural network architectures, activation functions, and learning rates to classify emotions from speech audio.

## Key Features

- **Multiple CNN Architectures**: Implements 1D CNN, 2D CNN, and combined models for speech emotion recognition
- **Advanced Feature Extraction**: Extract both 1D spectral features and 2D mel-spectrogram features from audio
- **Activation Function Analysis**: Comparative analysis of ReLU, SiLU (Swish), and ELU activations
- **Hyperparameter Investigation**: Exploration of different learning rates and model configurations
- **Variable-Length Processing**: Support for both fixed-size and variable-length audio inputs (for combined models)

## Performance Highlights

- **Best Model**: 2D CNN with SiLU activation and learning rate 0.001 achieved **64.1%** accuracy and **0.634** F1-score.
- **1D CNN Performance**: 1D CNN with SiLU activation and learning rate 0.001 achieved **61.2%** accuracy and **0.606** F1-score.
- **Variable Length Processing**: Improved combined model performance from 61.3% to 62.2% accuracy (based on prior experiments for combined models).
- **Emotion Recognition Insights**: Identified patterns in emotion recognition challenges (e.g., confusion between Fear/Sadness and Disgust being broadly confused).

## Repository Structure

```
SpeechEmotionRecognition/
├── speech_emotion_recognition/    # Main package directory
│   ├── config/                    # Configuration settings
│   ├── preprocessing/             # Audio processing and feature extraction
│   ├── models/                    # Neural network model implementations
│   ├── data_loader/               # Dataset loading utilities
│   └── utils/                     # Utility functions
├── data/                          # Data directory (CREMA-D dataset)
├── trained_models/                # Saved model checkpoints
├── report/                        # LaTeX report and figures
│   ├── sections/                  # Report sections
│   └── images/                    # Figures and diagrams
└── notebooks/                     # Jupyter notebooks for analysis and visualization
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SpeechEmotionRecognition.git
cd SpeechEmotionRecognition

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python speech_emotion_recognition/train.py --data_dir data/Crema --wandb_project SER
```
*Adjust configuration files in `speech_emotion_recognition/config/` for specific model types, learning rates, and activation functions.*

### Evaluating a Model

```bash
python speech_emotion_recognition/train.py --data_dir data/Crema --resume_checkpoint path/to/your/best_model.pth --evaluate_only
```

## CREMA-D Dataset

This project uses the [CREMA-D dataset](https://github.com/CheyneyComputerScience/CREMA-D), which contains:
- 7,442 audio clips from 91 actors (48 male, 43 female)
- 6 emotion categories: Anger, Disgust, Fear, Happiness, Neutral, and Sadness
- Diverse ethnic backgrounds and age distributions
- Audio files in WAV format with consistent recording quality

## Model Architectures

### 1D CNN Architecture
- Three convolutional blocks with increasing filter sizes (e.g., 128, 256, 512)
- Each block includes Conv1D, GroupNorm, Activation (ReLU/SiLU/ELU), MaxPool1D, and Dropout
- Global Average Pooling and fully connected classifier

### 2D CNN Architecture (Best Performer)
- Four convolutional blocks with increasing filter sizes (e.g., 32, 64, 512, 1024)
- Each block includes Conv2D, GroupNorm, Activation (ReLU/SiLU/ELU), MaxPool2D, and Dropout
- Global Average Pooling and fully connected classifier

### Combined Architecture
- Parallel 1D and 2D CNN paths
- Concatenation of features from both paths
- Fully connected head with LayerNorm and Dropout

## Results

| Model                   | Activation | Learning Rate | Accuracy  | F1-Score  | Precision* |
| ----------------------- | ---------- | ------------- | --------- | --------- | ---------- |
| 2D CNN                  | SiLU       | 0.001         | **64.1%** | **0.634** | **0.643**  |
| Combined (var. length)† | SiLU       | 0.001         | 62.2%     | 0.619     | 0.625      |
| 1D CNN                  | SiLU       | 0.001         | 61.2%     | 0.606     | 0.610      |
| Combined (fixed size)†  | SiLU       | 0.001         | 61.3%     | 0.608     | 0.614      |

## Key Findings

1.  **2D CNN models excel**: The 2D CNN with SiLU activation and a learning rate of 0.001 achieved the highest performance (64.1% accuracy, 0.634 F1-score).
2.  **Activation functions matter**: SiLU performed best for the top configurations of both 2D CNN and 1D CNN models (when paired with lr=0.001).
3.  **Variable-length processing improves results**: For combined models, preserving temporal dynamics improved performance (based on prior experiments).
4.  **Learning rate tuning is critical**: Lower rates (0.001) generally worked best for the optimal configurations of both 1D and 2D CNNs.
5.  **Some emotions are inherently harder to distinguish**: Disgust was particularly challenging, with Fear and Sadness also showing high confusion rates.

## Future Directions

- Attention mechanisms to focus on emotionally salient parts of speech
- Transformer-based architectures for improved temporal modeling
- Multi-modal approaches combining audio with text or visual data
- Data augmentation techniques specific to emotional speech
- Designing or optimizing custom activation functions specifically for SER tasks.

## Team

- **Youssif Khaled Ahmed** (21011655)
- **Esmail Mahmoud Hassan** (21010272)
- **Ahmed Ayman Ahmed** (21010048)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- Prof. Dr. Marwan Torki and Eng. Ismail El-Yamany for their guidance
- Alexandria University, Faculty of Engineering, Computer and Systems Engineering Department
- The creators of the CREMA-D dataset