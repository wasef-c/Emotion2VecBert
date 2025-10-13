# Multimodal Emotion Recognition with Audio and Text

A comprehensive multimodal emotion recognition system that extends the original audio-only Emotion2Vec classification system to support text and multimodal fusion approaches.

## Overview

This system supports three modality configurations:
- **Audio-only**: Uses emotion2vec audio features
- **Text-only**: Uses frozen BERT text encoder on transcripts
- **Multimodal**: Fuses audio and text features using cross-attention or other fusion mechanisms

All advanced features from the original system are preserved, including:
- Curriculum learning with multiple strategies
- Speaker disentanglement
- Difficulty scaling
- LOSO (Leave-One-Session-Out) evaluation
- Cross-corpus evaluation
- Comprehensive WandB logging and visualization

## Architecture

### Core Components

1. **Text Encoder** (`text_encoder.py`)
   - Frozen BERT model for text feature extraction
   - Supports caching for efficient processing
   - Output: 768-dimensional text features

2. **Fusion Mechanisms** (`fusion.py`)
   - **Cross-Attention**: Audio and text attend to each other
   - **Simple Concat**: Concatenation + MLP
   - **Gated Fusion**: Learned gating mechanism
   - **Adaptive Fusion**: Handles missing modalities

3. **Multimodal Model** (`model.py`)
   - Unified architecture supporting all modalities
   - Configurable fusion type
   - Dynamic dropout adjustment
   - Factory function for easy instantiation

4. **Training Pipeline** (`main.py`)
   - Multimodal dataset loading with text transcripts
   - Modality-aware training loops
   - Curriculum learning integration
   - LOSO and cross-corpus evaluation

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install wandb scikit-learn numpy pandas
pip install matplotlib seaborn scipy
```

### Datasets

The system works with HuggingFace datasets that include:
- `emotion2vec_features`: Pre-extracted audio features
- `transcript`: Text transcripts
- `label`: Emotion labels (0: neutral, 1: happy, 2: sad, 3: anger)
- Additional metadata: VAD values, speaker IDs, etc.

Supported datasets:
- IEMO (`cairocode/IEMO_Emotion2Vec`)
- MSPI (`cairocode/MSPI_Emotion2Vec`)
- MSPP (`cairocode/MSPP_Emotion2Vec_V3`)

## Usage

### Quick Start

#### 1. Audio-Only Mode (Baseline)

```bash
python main.py --config configs/audio_only.yaml --experiment 0
```

#### 2. Text-Only Mode

```bash
python main.py --config configs/text_only.yaml --experiment 0
```

#### 3. Multimodal Mode (Audio + Text)

```bash
python main.py --config configs/multimodal_baseline.yaml --experiment 0
```

### Running Multiple Experiments

Run all experiments in a configuration file:

```bash
python main.py --config configs/multimodal_baseline.yaml --all
```

### Configuration Files

#### `configs/multimodal_baseline.yaml`
Complete multimodal experiments comparing audio-only, text-only, and fusion approaches.

**Experiments included:**
- Multimodal Full System (cross-attention)
- Audio Only Baseline
- Text Only Baseline
- Multimodal Concat Fusion
- Multimodal Gated Fusion

#### `configs/audio_only.yaml`
Audio-only experiments replicating the optimal configuration from the original system.

#### `configs/text_only.yaml`
Text-only experiments with BERT encoder.

#### `configs/fusion_ablation.yaml`
Comprehensive ablation study of fusion mechanisms and hyperparameters.

## Configuration Parameters

### Modality Settings

```yaml
modality: "both"  # "audio", "text", or "both"
audio_dim: 768
text_model_name: "bert-base-uncased"
freeze_text_encoder: true
text_max_length: 128
```

### Fusion Settings (for modality="both")

```yaml
fusion_type: "cross_attention"  # "cross_attention", "concat", "gated", "adaptive"
fusion_hidden_dim: 512
num_attention_heads: 8
```

### Training Settings (Optimal Configuration)

```yaml
learning_rate: 9e-3
weight_decay: 5e-6
num_epochs: 30
batch_size: 64
dropout: 0.1

# Curriculum learning
use_curriculum_learning: true
curriculum_epochs: 15
curriculum_pacing: "sqrt"
curriculum_type: "difficulty"

# Advanced features
use_difficulty_scaling: true
use_speaker_disentanglement: true
```

### Evaluation Settings

```yaml
evaluation_mode: "cross_corpus"  # "loso", "cross_corpus", or "both"
train_dataset: "IEMO"  # "IEMO", "MSPI", or "MSPP"
```

## Key Features

### 1. Cross-Attention Fusion

The default fusion mechanism allows audio and text modalities to attend to each other:

```
Audio Features ──→ Project ──→ Attend to Text ──→ FFN ──→ ┐
                                                            ├──→ Concat ──→ Fuse ──→ Classify
Text Features  ──→ Project ──→ Attend to Audio ──→ FFN ──→ ┘
```

### 2. Curriculum Learning

Multiple curriculum strategies preserved from original system:
- **Difficulty-based**: Start with easy samples (based on VAD distance)
- **Class balance**: Gradually introduce emotional classes
- **Model confidence**: Use model's own confidence scores
- **Preset order**: Use pre-computed curriculum from dataset

### 3. Speaker Disentanglement

Groups training batches by speaker to learn speaker-invariant representations.

### 4. Difficulty Scaling

Weights loss based on sample difficulty (VAD distance from prototypes).

### 5. Comprehensive Evaluation

- **LOSO**: Leave-One-Session-Out cross-validation
- **Cross-Corpus**: Train on one dataset, test on others
- **Metrics**: Accuracy, UAR (Unweighted Average Recall), F1
- **Visualizations**: Confusion matrices, difficulty-accuracy plots

## Project Structure

```
Emotion2Vec_Text/
├── main.py                        # Main training script
├── model.py                       # Multimodal models
├── text_encoder.py                # BERT text feature extractor
├── fusion.py                      # Fusion mechanisms
├── functions.py                   # Utility functions
├── config.py                      # Configuration class
├── run_config.py                  # Config runner
├── configs/
│   ├── multimodal_baseline.yaml   # Main multimodal experiments
│   ├── audio_only.yaml            # Audio baseline
│   ├── text_only.yaml             # Text baseline
│   └── fusion_ablation.yaml       # Fusion ablation study
└── README.md
```

## Experimental Results

The system is designed to evaluate:

1. **Modality Comparison**
   - Audio-only performance (baseline)
   - Text-only performance
   - Multimodal fusion performance

2. **Fusion Mechanisms**
   - Cross-attention vs simple concatenation
   - Gated fusion vs adaptive fusion
   - Impact of fusion hidden dimensions
   - Impact of attention head count

3. **Training Strategies**
   - Curriculum learning effectiveness
   - Speaker disentanglement impact
   - Difficulty scaling benefits

## WandB Logging

All experiments automatically log to Weights & Biases:

- Training/validation curves
- Per-epoch metrics
- Confusion matrices
- Difficulty-accuracy plots
- Hyperparameter tracking
- Model artifacts

## Advanced Usage

### Custom Fusion Mechanism

Add new fusion modules in `fusion.py`:

```python
class MyCustomFusion(nn.Module):
    def __init__(self, audio_dim, text_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Your implementation

    def forward(self, audio_features, text_features):
        # Fusion logic
        return fused_features
```

Register in `get_fusion_module()` factory function.

### Custom Text Encoder

Replace BERT with alternative models:

```yaml
text_model_name: "roberta-base"
# or
text_model_name: "distilbert-base-uncased"
# or any HuggingFace model
```

### Testing Individual Modules

Each module includes test code:

```bash
# Test text encoder
python text_encoder.py

# Test fusion modules
python fusion.py

# Test models
python model.py
```

## Citation

If you use this code, please cite:

```bibtex
@software{multimodal_emotion2vec,
  title={Multimodal Emotion Recognition with Audio and Text},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Emotion2Vec_Text}
}
```

## Original System

This project extends the audio-only emotion recognition system with:
- Optimal configuration from `ablation_with_speaker2.yaml`
- All curriculum learning strategies
- Speaker disentanglement
- Difficulty scaling
- Comprehensive evaluation framework

## License

[Specify your license]

## Contact

[Your contact information]

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```yaml
batch_size: 32  # or 16
```

### Missing Transcripts

The system automatically handles missing transcripts with `[EMPTY]` placeholder.

### Frozen Text Encoder Not Training

This is expected behavior. Text encoder is frozen by design. To fine-tune:
```yaml
freeze_text_encoder: false
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Future Work

Potential extensions:
- Video modality integration
- Attention visualization
- Early fusion strategies
- Late fusion approaches
- Multi-task learning with VAD prediction
- Real-time inference pipeline
