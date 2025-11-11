# Merged Datasets Migration - Summary

## Overview
The codebase has been updated to use **merged datasets** that contain both pre-extracted features AND raw audio data in a single dataset. This simplifies the code and improves memory efficiency.

## What Changed

### 1. Dataset Structure
**Before (separate datasets):**
- Feature dataset: `cairocode/MSPI_Emotion2Vec_Text` (only features)
- WAV dataset: `cairocode/MSPI_WAV_Diff` (only audio)
- Needed DatasetMapper to join them

**After (merged datasets):**
- Merged dataset: `cairocode/MSPI_Audio_Text_Merged`
- Contains BOTH features and audio in same dataset
- Direct column access - no mapping needed

### 2. Merged Dataset Columns
Each sample now has:
```python
{
    "emotion2vec_features": [...],     # Pre-extracted Emotion2Vec features
    "audio": {                          # Raw audio (for wav2vec2/hubert)
        "array": [0.00036621, ...],     # Waveform samples as float array
        "sampling_rate": 16000,         # Sampling rate (Hz)
        "path": "..."                   # Original file path
    },
    "transcript": "...",                # Text transcription
    "text": "...",                      # Alternative text field
    "label": 0,                         # Emotion label (0=angry, 1=happy, 2=neutral, 3=sad)
    "speaker_id": 123,                  # Speaker identifier
    "valence": 2.5,                     # Valence score
    "arousal": 3.2,                     # Arousal score
    "domination": 3.0,                  # Dominance score
    # ... other metadata
}
```

## Files Updated

### 1. `main.py` (lines 626-690)
**Changes:**
- Updated `feature_dataset_map` → `merged_dataset_map`
- Changed dataset paths to merged versions
- Removed DatasetMapper loading logic
- Simplified audio access: `item["audio"]` directly
- Removed `wav_map`, `id_column`, `mapper` attributes

**Key Code:**
```python
# Load merged dataset
merged_dataset_map = {
    "IEMO": "cairocode/IEMO_Audio_Text_Merged",
    "MSPI": "cairocode/MSPI_Audio_Text_Merged",
    "MSPP": "cairocode/MSPP_Audio_Text_Merged",
    "CMUMOSEI": "cairocode/CMUMOSEI_Audio_Text_Merged",
    "SAMSEMO": "cairocode/SAMSEMO_Audio_Text_Merged",
}

# Use audio directly from merged dataset
if self.audio_encoder_type == "preextracted":
    features = torch.tensor(
        item["emotion2vec_features"][0]["feats"],
        dtype=torch.float32
    )
else:
    # Raw audio for wav2vec2/hubert
    features = item["audio"]  # {"array": [...], "sampling_rate": 16000}
```

### 2. `dataset_mapper.py`
**Changes:**
- Added `MERGED_DATASET_MAP` constant at top
- Added `load_merged_dataset()` utility function
- Updated docstrings to reflect merged dataset approach
- Marked `DatasetMapper` class as deprecated (kept for backward compatibility)
- Updated test code to demonstrate merged dataset usage

**New Utility Function:**
```python
from dataset_mapper import load_merged_dataset

# Load merged dataset
dataset = load_merged_dataset("MSPI", split="train")
sample = dataset[0]

# Access features
features = sample["emotion2vec_features"][0]["feats"]

# Access raw audio
audio_array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]
```

### 3. `AUDIO_ENCODER_FIX.md`
**Changes:**
- Updated to reflect merged dataset approach
- Explained new dataset structure
- Updated code examples
- Added migration guide from old approach

## Available Datasets

All datasets are now available as merged versions on HuggingFace:

| Dataset | HuggingFace Path | Samples |
|---------|------------------|---------|
| IEMO | `cairocode/IEMO_Audio_Text_Merged` | ~10k |
| MSPI | `cairocode/MSPI_Audio_Text_Merged` | ~10k |
| MSPP | `cairocode/MSPP_Audio_Text_Merged` | ~108k |
| CMUMOSEI | `cairocode/CMUMOSEI_Audio_Text_Merged` | ~23k |
| SAMSEMO | `cairocode/SAMSEMO_Audio_Text_Merged` | ~7k |

## How to Use

### Training with Pre-extracted Features (Default)
```yaml
# config.yaml
modality: "audio"
audio_encoder_type: "preextracted"  # Uses emotion2vec_features column
```

### Training with Raw Audio (Wav2Vec2/HuBERT)
```yaml
# config.yaml
modality: "audio"
audio_encoder_type: "wav2vec2"      # Uses audio column
audio_model_name: "facebook/wav2vec2-base"
```

### Multimodal Training (Audio + Text)
```yaml
# config.yaml
modality: "both"
audio_encoder_type: "preextracted"
text_model_name: "bert-base-uncased"
fusion_type: "attention"
```

## Benefits

✅ **Simpler Code**
   - No DatasetMapper complexity
   - Direct column access
   - Fewer lines of code

✅ **Memory Efficient**
   - Single dataset load instead of two
   - No separate mapping dictionary
   - Better for large datasets like MSPP

✅ **Faster Loading**
   - One HuggingFace load operation
   - No joining/mapping step
   - Quicker iteration

✅ **Easier Debugging**
   - All data in one place
   - Can inspect samples easily
   - Clear data structure

✅ **Backward Compatible**
   - Config files work the same
   - Same `audio_encoder_type` options
   - Same training pipeline

## Testing

Test the merged dataset loading:
```bash
cd /home/rml/Documents/pythontest/Emotion2VecTraining/Emotion2Vec_Text
python dataset_mapper.py
```

Expected output:
```
Testing merged dataset loading...

============================================================
Available Merged Datasets
============================================================
  IEMO: cairocode/IEMO_Audio_Text_Merged
  MSPI: cairocode/MSPI_Audio_Text_Merged
  MSPP: cairocode/MSPP_Audio_Text_Merged
  CMUMOSEI: cairocode/CMUMOSEI_Audio_Text_Merged
  SAMSEMO: cairocode/SAMSEMO_Audio_Text_Merged

============================================================
Testing MSPI merged dataset
============================================================
📥 Loading MSPI from cairocode/MSPI_Audio_Text_Merged
✅ Loaded 10000 samples
   Columns: ['emotion2vec_features', 'audio', 'transcript', 'label', ...]

📊 Sample structure:
   Keys: ['emotion2vec_features', 'audio', 'transcript', 'label', ...]

🎵 Audio data:
   Type: dict
   Keys: ['array', 'sampling_rate', 'path']
   Array shape: 160000 samples
   Sampling rate: 16000 Hz

🔊 Pre-extracted features available

📝 Text transcript available

✅ Merged dataset test completed!
```

## Migration Checklist

If you have custom code using the old approach:

- [ ] Update dataset names to merged versions
- [ ] Remove DatasetMapper imports and usage
- [ ] Change `wav_map` lookups to `item["audio"]`
- [ ] Update any hardcoded dataset paths
- [ ] Test with your configs to ensure everything works
- [ ] Re-run experiments to verify results match

## Questions?

- **Q: Can I still use pre-extracted features?**
  - A: Yes! Set `audio_encoder_type: "preextracted"` in config

- **Q: Do I need to re-download datasets?**
  - A: Yes, but HuggingFace will handle caching automatically

- **Q: Will my old configs work?**
  - A: Yes! The config format hasn't changed, just the underlying data loading

- **Q: What if audio column is missing?**
  - A: Code automatically falls back to pre-extracted features with a warning

- **Q: Is DatasetMapper still needed?**
  - A: No, but it's kept for backward compatibility. Use `load_merged_dataset()` instead

## Summary

✅ Updated to merged datasets with audio column
✅ Simplified codebase (removed complex mapping logic)
✅ Improved memory efficiency
✅ Backward compatible with existing configs
✅ Ready for training with both pre-extracted features and raw audio
