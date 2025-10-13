# Quick Start Guide

## Setup (5 minutes)

```bash
cd /home/rml/Documents/pythontest/Emotion2VecTraining/Emotion2Vec_Text

# Install dependencies
pip install -r requirements.txt

# Login to WandB
wandb login

# Test system
python test_system.py
```

## Run Your First Experiment (2 commands)

### Option 1: Audio-Only Baseline (Your Optimal Config)
```bash
python main.py --config configs/audio_only.yaml --experiment 0
```

### Option 2: Multimodal (Audio + Text)
```bash
python main.py --config configs/multimodal_baseline.yaml --experiment 0
```

### Option 3: Text-Only
```bash
python main.py --config configs/text_only.yaml --experiment 0
```

## Run All Experiments in Config File
```bash
python main.py --config configs/multimodal_baseline.yaml --all
```

## Key Configuration Parameters

Edit any `.yaml` file in `configs/` to customize:

```yaml
# Change modality
modality: "both"  # "audio", "text", or "both"

# Change fusion type (for modality="both")
fusion_type: "cross_attention"  # "cross_attention", "concat", "gated"

# Change dataset
train_dataset: "IEMO"  # "IEMO", "MSPI", or "MSPP"

# Evaluation mode
evaluation_mode: "cross_corpus"  # "loso", "cross_corpus", or "both"
```

## Understanding Results

Results are logged to WandB with:
- Training/validation curves
- Confusion matrices
- Difficulty-accuracy plots
- Final metrics: Accuracy, UAR, F1

Check your WandB dashboard at: https://wandb.ai/[your-username]/

## Experiment Configs Explained

### `multimodal_baseline.yaml` - Main Experiments
5 experiments comparing audio, text, and fusion approaches:
1. Multimodal Full System (cross-attention) ← **Recommended first run**
2. Audio Only Baseline
3. Text Only Baseline
4. Multimodal Concat Fusion
5. Multimodal Gated Fusion

### `audio_only.yaml` - Audio Baseline
Your optimal audio-only configuration:
- Curriculum learning
- Speaker disentanglement
- Difficulty scaling

### `text_only.yaml` - Text Baseline
Text-only with BERT encoder

### `fusion_ablation.yaml` - Fusion Study
8 experiments testing different fusion mechanisms and hyperparameters

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config:
```yaml
batch_size: 32  # or 16
```

### Import Errors
```bash
pip install torch transformers datasets wandb scikit-learn
```

### WandB Not Logging
```bash
wandb login
```

## File Structure

```
Emotion2Vec_Text/
├── main.py              ← Main training script
├── configs/             ← Configuration files
│   ├── multimodal_baseline.yaml  ← Start here
│   ├── audio_only.yaml
│   ├── text_only.yaml
│   └── fusion_ablation.yaml
├── test_system.py       ← Run this to test setup
├── README.md            ← Full documentation
└── IMPLEMENTATION_SUMMARY.md  ← Technical details
```

## Running Experiments

### Single experiment by index
```bash
python main.py --config configs/multimodal_baseline.yaml --experiment 0
```

### Single experiment by name
```bash
python main.py --config configs/multimodal_baseline.yaml --experiment "Multimodal Full System IEMO"
```

### All experiments in sequence
```bash
python main.py --config configs/multimodal_baseline.yaml --all
```

## Quick Reference

### Modalities
- `audio`: Uses emotion2vec features only
- `text`: Uses BERT text encoder only
- `both`: Fuses audio and text

### Fusion Types (for modality="both")
- `cross_attention`: Audio and text attend to each other (recommended)
- `concat`: Simple concatenation + MLP
- `gated`: Learned gating mechanism
- `adaptive`: Handles missing modalities

### Evaluation Modes
- `loso`: Leave-One-Session-Out cross-validation
- `cross_corpus`: Train on one dataset, test on others
- `both`: Run both evaluations

### Datasets
- `IEMO`: Interactive Emotional Dyadic Motion Capture
- `MSPI`: MSP-Improv
- `MSPP`: MSP-Podcast

## Next Steps

1. ✅ Run test: `python test_system.py`
2. ✅ Run baseline: `python main.py --config configs/multimodal_baseline.yaml --experiment 0`
3. ✅ Check WandB dashboard for results
4. 📊 Run full suite: `python main.py --config configs/multimodal_baseline.yaml --all`
5. 🔬 Analyze results and iterate

## Getting Help

- See `README.md` for full documentation
- See `IMPLEMENTATION_SUMMARY.md` for technical details
- Run `python test_system.py` to diagnose issues
- Check each module has `if __name__ == "__main__"` test code

## Example: Custom Experiment

Create `configs/my_experiment.yaml`:

```yaml
wandb_project: "My_Experiments"

template_config: &template
  learning_rate: 9e-3
  weight_decay: 5e-6
  num_epochs: 30
  batch_size: 64

  modality: "both"
  fusion_type: "cross_attention"

  train_dataset: "IEMO"
  evaluation_mode: "cross_corpus"

  use_curriculum_learning: true
  curriculum_epochs: 15
  use_difficulty_scaling: true
  use_speaker_disentanglement: true

experiments:
  - <<: *template
    name: "My Custom Experiment"
```

Run it:
```bash
python main.py --config configs/my_experiment.yaml --experiment 0
```

---

**That's it! You're ready to go! 🚀**
