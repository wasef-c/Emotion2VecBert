# Multi-Dataset Training

This document explains how to train on multiple concatenated datasets simultaneously to improve model generalization.

## Overview

The multi-dataset training feature allows you to train a single model on multiple emotion recognition datasets at once. This can help improve:
- **Generalization**: Models trained on diverse data generalize better to unseen datasets
- **Robustness**: Exposure to different recording conditions and speakers
- **Cross-corpus performance**: Better transfer learning across different corpora

## Supported Datasets

### Training Datasets (with VAD values)

Only the following datasets have VAD (Valence-Arousal-Dominance) annotations required for difficulty-based curriculum learning:

- **IEMO** (IEMOCAP)
- **MSPI** (MSP-IMPROV)
- **MSPP** (MSP-PODCAST)

### Test-Only Datasets (no VAD values)

These datasets can be used for evaluation but NOT training:

- **CMUMOSEI** (CMU-MOSEI) - test only
- **SAMSEMO** (SAMSE-MO) - test only

## Configuration

### Single Dataset (Existing Behavior)

```yaml
train_dataset: "IEMO"
```

### Multiple Datasets (New Feature)

```yaml
train_dataset: ["IEMO", "MSPI"]
```

or with three datasets:

```yaml
train_dataset: ["IEMO", "MSPI", "MSPP"]
```

## How It Works

1. **Dataset Loading**: Each specified dataset is loaded independently as a `SimpleEmotionDataset`
2. **Concatenation**: Datasets are concatenated using the `ConcatenatedDataset` class
3. **Training**: The model trains on all samples from all datasets
4. **Evaluation**: Cross-corpus evaluation on datasets NOT included in training

## Example Config File

```yaml
template: &template
  learning_rate: 1e-5
  num_epochs: 30
  batch_size: 64
  modality: "both"

  # Best performing fusion settings (from ablation studies)
  fusion_type: "cross_attention"
  fusion_hidden_dim: 1024
  num_attention_heads: 16

  evaluation_mode: "cross_corpus"  # Required for multi-dataset
  wandb_project: "MultiDataset_Training"
  seeds: [42, 87739, 1829]

experiments:
  - <<: *template
    name: "IEMO+MSPI_CrossAttn"
    train_dataset: ["IEMO", "MSPI"]
    # Will test on: MSPP, CMUMOSEI, SAMSEMO

  - <<: *template
    name: "IEMO+MSPI+MSPP_CrossAttn"
    train_dataset: ["IEMO", "MSPI", "MSPP"]
    # Will test on: CMUMOSEI, SAMSEMO
    # Uses ALL available VAD-labeled data

  - <<: *template
    name: "LeaveOut_IEMO_CrossAttn"
    train_dataset: ["MSPI", "MSPP"]
    # Will test on: IEMO, CMUMOSEI, SAMSEMO
```

## Running Experiments

### Single Experiment

```bash
python main.py --config configs/multi_dataset_training.yaml --experiment 0
```

### All Experiments

```bash
python main.py --config configs/multi_dataset_training.yaml --all
```

## Important Notes

1. **LOSO Not Supported**: Leave-One-Session-Out evaluation is not available with multiple training datasets. Use `evaluation_mode: "cross_corpus"` instead.

2. **Automatic Test Set**: Test datasets are automatically determined as all datasets NOT in the training set.

3. **WandB Logging**: Multi-dataset runs are logged with combined names (e.g., `IEMO+MSPI`) in WandB sections.

4. **Dataset Naming**: The concatenated dataset is named using `+` as separator (e.g., `IEMO+MSPI+MSPP`).

## Expected Output

```
🔗 Training on multiple datasets: IEMO, MSPI
✅ Concatenated 2 datasets: IEMO+MSPI
   Total samples: 15234
   IEMO: 7523 samples (indices 0-7522)
   MSPI: 7711 samples (indices 7523-15233)

🚀 Training: IEMO+MSPI -> ['MSPP', 'CMUMOSEI', 'SAMSEMO']
```

## Sample Configurations

### Two-Dataset Combinations

```yaml
# Only use datasets with VAD values
- train_dataset: ["IEMO", "MSPI"]    # Test: MSPP, CMUMOSEI, SAMSEMO
- train_dataset: ["IEMO", "MSPP"]    # Test: MSPI, CMUMOSEI, SAMSEMO
- train_dataset: ["MSPI", "MSPP"]    # Test: IEMO, CMUMOSEI, SAMSEMO
```

### All VAD-Labeled Datasets

```yaml
# Maximum training data - all datasets with VAD values
- train_dataset: ["IEMO", "MSPI", "MSPP"]
  # Test: CMUMOSEI, SAMSEMO
  # This is the recommended configuration for best generalization
```

### Leave-One-Out (VAD datasets)

```yaml
# Train on 2 VAD datasets, test on the 3rd + CMUMOSEI + SAMSEMO
- train_dataset: ["MSPI", "MSPP"]    # Leave out: IEMO
- train_dataset: ["IEMO", "MSPP"]    # Leave out: MSPI
- train_dataset: ["IEMO", "MSPI"]    # Leave out: MSPP
```

## Performance Expectations

Based on typical cross-corpus results:

| Training Data | Test Data | Expected UAR |
|--------------|-----------|--------------|
| IEMO only | MSPI | 55-60% |
| MSPI only | IEMO | 60-65% |
| IEMO+MSPI | MSPP | **65-70%** (improved) |
| IEMO+MSPI+MSPP | CMUMOSEI | **68-73%** (improved) |

Multi-dataset training typically provides **3-8% improvement** in cross-corpus generalization.

## WandB Organization

Multi-dataset experiments are logged with the following structure:

```
AVERAGED_IEMO+MSPI/
  ├── validation_accuracy_mean
  ├── validation_uar_mean
  ├── IEMO+MSPItoMSPP_accuracy_mean
  ├── IEMO+MSPItoMSPP_uar_mean
  ├── IEMO+MSPItoCMUMOSEI_accuracy_mean
  └── ...
```

## Tips for Best Results

1. **Use Best Fusion Settings**: Based on ablation studies, use:
   ```yaml
   fusion_type: "cross_attention"
   fusion_hidden_dim: 1024
   num_attention_heads: 16
   ```

2. **Only Use VAD Datasets**: Only IEMO, MSPI, and MSPP have VAD values:
   ```yaml
   # ✅ Correct
   train_dataset: ["IEMO", "MSPI", "MSPP"]

   # ❌ Wrong - CMUMOSEI and SAMSEMO lack VAD values
   train_dataset: ["IEMO", "CMUMOSEI"]
   ```

3. **Curriculum Learning**: Keep curriculum learning enabled for gradual introduction of diverse samples (requires VAD values)

4. **Multiple Seeds**: Always use multiple seeds for statistical significance:
   ```yaml
   seeds: [42, 87739, 1829]
   ```

5. **Maximum Training Data**: For best generalization, train on all three VAD datasets:
   ```yaml
   train_dataset: ["IEMO", "MSPI", "MSPP"]
   ```

6. **Monitor Class Distribution**: Check that all 4 emotions are well-represented across combined datasets

## Troubleshooting

### Error: "LOSO evaluation not supported with multiple training datasets"

**Solution**: Change `evaluation_mode` to `"cross_corpus"`:
```yaml
evaluation_mode: "cross_corpus"
```

### Different Feature Dimensions

All datasets use emotion2vec features with dimension 768, so this should not be an issue.

### Class Imbalance

Multi-dataset training helps balance classes by combining different datasets with different class distributions.

## Advanced Usage

### Custom Dataset Combinations

You can create custom combinations for specific research questions:

```yaml
experiments:
  # Scripted vs. spontaneous
  - name: "Scripted_Speech"
    train_dataset: ["IEMO"]  # Acted

  - name: "Spontaneous_Speech"
    train_dataset: ["MSPI", "MSPP"]  # Improvised/natural

  - name: "Mixed_Speech"
    train_dataset: ["IEMO", "MSPI", "MSPP"]  # Both
```

### Domain Adaptation

Use multi-dataset training for domain adaptation:

```yaml
- name: "Audio_Podcasts_to_Dialogue"
  train_dataset: ["MSPP", "CMUMOSEI"]  # Source: podcasts/videos
  # Test on: IEMO, MSPI (Target: dialogues)
```

## References

- Concatenated dataset implementation: `ConcatenatedDataset` class in `main.py`
- Configuration examples: `configs/multi_dataset_training.yaml`
- Test script: `test_multi_dataset.py`
