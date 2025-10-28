# Fusion Model Configuration Changes

## Summary

Made the FocalLoss temperature and post-curriculum dropout configurable to allow different values for fusion models vs audio-only models, without changing hardcoded values.

## Issues Fixed

### 1. Seeds Functionality Not Working ✅
**Problem:** `run_all_experiments_from_yaml()` was calling `run_experiment()` instead of `run_experiment_with_seeds()`
**Fix:** Changed line 165 in main.py to call `run_experiment_with_seeds(config)`

### 2. Model Only Predicting Label 0 for Fusion Models 🔧
**Root Cause:** Temperature of 2000 in FocalLoss works well for audio-only but causes issues for fusion models
**Solution:** Made temperature and dropout configurable per experiment

## New Configuration Parameters

Added to `config.py`:

```python
# Loss function settings
self.loss_temperature = 2000  # Temperature for FocalLoss (lower = sharper, higher = softer)
self.focal_gamma = 2.0  # Focal loss gamma parameter
self.post_curriculum_dropout = 0.6  # Dropout rate after curriculum learning completes
```

## Configuration Values

### Audio-Only (Works Well - Keep Existing)
```yaml
loss_temperature: 2000  # High temperature
post_curriculum_dropout: 0.6  # High dropout
```

### Fusion Models (New - Better Values)
```yaml
loss_temperature: 1.0  # Lower temperature for sharper gradients
post_curriculum_dropout: 0.3  # Lower dropout to preserve multimodal signal
```

### Text-Only (New - Better Values)
```yaml
loss_temperature: 1.0  # Lower temperature
post_curriculum_dropout: 0.3  # Lower dropout
```

## Updated Files

### 1. `config.py`
- Added `loss_temperature`, `focal_gamma`, and `post_curriculum_dropout` parameters

### 2. `functions.py`
- Updated `FocalLossAutoWeights.__init__()` to accept `temperature` parameter
- Temperature now configurable instead of hardcoded to 2000

### 3. `main.py`
- Updated `load_config_from_yaml()` to parse new float parameters
- Updated dropout increase code (2 places) to use `config.post_curriculum_dropout`
- Updated FocalLoss initialization in LOSO to pass config parameters
- Fixed `run_all_experiments_from_yaml()` to call `run_experiment_with_seeds()`

### 4. `configs/multimodal_baseline.yaml`
- Added default values in template (audio-only values: 2000, 0.6)
- Override for each fusion experiment:
  - Cross-attention: `loss_temperature: 1.0`, `post_curriculum_dropout: 0.3`
  - Concat fusion: `loss_temperature: 1.0`, `post_curriculum_dropout: 0.3`
  - Gated fusion: `loss_temperature: 1.0`, `post_curriculum_dropout: 0.3`
  - Text-only: `loss_temperature: 1.0`, `post_curriculum_dropout: 0.3`
- Audio-only keeps template defaults (no override needed)

## How It Works

### Temperature Effect
- **High (2000)**: Softens logits → nearly uniform probabilities → works for audio-only
- **Low (1.0)**: Preserves logit magnitudes → sharp gradients → works for fusion/text

### Dropout Effect
- **High (0.6)**: Strong regularization for audio-only after curriculum
- **Low (0.3)**: Preserves multimodal interactions for fusion models

## Example Usage

```yaml
experiments:
  - <<: *template
    name: "Audio Only"
    modality: "audio"
    # Uses template defaults: temperature=2000, dropout=0.6

  - <<: *template
    name: "Multimodal Fusion"
    modality: "both"
    fusion_type: "cross_attention"
    loss_temperature: 1.0  # Override for fusion
    post_curriculum_dropout: 0.3  # Override for fusion
```

## Benefits

1. **No hardcoded changes**: Audio-only model continues to work with proven settings
2. **Per-experiment control**: Each experiment can specify its own values
3. **Backward compatible**: Default values match current audio-only behavior
4. **Flexible**: Easy to tune for new modalities or fusion mechanisms

## Testing

Run experiments with:
```bash
python main.py --config configs/multimodal_baseline.yaml --all
```

- Audio-only will use temperature=2000, dropout=0.6 (existing good behavior)
- Fusion models will use temperature=1.0, dropout=0.3 (fix for label 0 prediction)
- Multi-seed functionality will work correctly

## What Changed in Behavior

### Before
- All experiments used temperature=2000 and dropout=0.6 (hardcoded)
- Fusion models predicted only label 0
- Seeds didn't work with `--all` flag

### After
- Audio-only: temperature=2000, dropout=0.6 (same as before - no change)
- Fusion models: temperature=1.0, dropout=0.3 (should fix label 0 issue)
- Text-only: temperature=1.0, dropout=0.3 (better training)
- Seeds work correctly with `--all` flag
- All configurable per experiment in YAML

## Recommended Next Steps

1. Run experiments to verify fusion models no longer predict only label 0
2. If needed, tune temperature/dropout values per experiment
3. Document final best values for each modality type
