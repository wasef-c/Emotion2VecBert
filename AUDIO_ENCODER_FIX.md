# Audio Encoder Loading - Merged Dataset Approach

## Current Approach (2024)
The codebase now uses **MERGED DATASETS** that contain both:
- ✅ Pre-extracted Emotion2Vec features (for fast training)
- ✅ Raw audio data (for training with wav2vec2/hubert/other encoders)
- ✅ Text transcripts
- ✅ All metadata and labels

Merged datasets on HuggingFace:
- `cairocode/IEMO_Audio_Text_Merged`
- `cairocode/MSPI_Audio_Text_Merged`
- `cairocode/MSPP_Audio_Text_Merged`
- `cairocode/CMUMOSEI_Audio_Text_Merged`
- `cairocode/SAMSEMO_Audio_Text_Merged`

## How It Works

### Dataset Structure
Each sample in the merged dataset contains:
```python
{
    "emotion2vec_features": [...],  # Pre-extracted features
    "audio": {                       # Raw audio data
        "array": [0.00036621, ...],  # Waveform samples
        "sampling_rate": 16000,      # Sample rate
        "path": "..."                # Original path
    },
    "transcript": "...",             # Text transcription
    "label": 0,                      # Emotion label (0-3)
    # ... other metadata
}
```

### Previous Problem (Fixed)
The old `SimpleEmotionDataset` was **always loading pre-extracted Emotion2Vec features**, regardless of the `audio_encoder_type` setting:

```python
# OLD CODE (WRONG):
if self.modality in ["audio", "both"]:
    features = torch.tensor(
        item["emotion2vec_features"][0]["feats"], dtype=torch.float32
    )
```

The code never checked `config.audio_encoder_type` and never loaded raw WAV audio from the WAV datasets.

## Current Solution (Merged Datasets)
Modified `SimpleEmotionDataset.__init__()` in `main.py` to:

1. **Load merged datasets** that already contain both features and audio
2. **Check `audio_encoder_type`** from config
3. **Select appropriate data source**:
   - `audio_encoder_type="preextracted"` → Use `emotion2vec_features` column
   - `audio_encoder_type="wav2vec2/hubert/etc"` → Use `audio` column

### Code (main.py lines 665-686)

```python
if self.modality in ["audio", "both"]:
    if self.audio_encoder_type == "preextracted":
        # Use pre-extracted Emotion2Vec features
        features = torch.tensor(
            item["emotion2vec_features"][0]["feats"],
            dtype=torch.float32
        )
        sequence_length = features.shape[0]  # [seq_len, feature_dim]
    else:
        # Use raw audio from merged dataset
        if "audio" in item and item["audio"] is not None:
            # Audio format: {"array": [...], "sampling_rate": 16000}
            features = item["audio"]
            sequence_length = 1  # Raw audio
        else:
            print(f"⚠️ Warning: No audio data found, skipping")
            continue
```

### Benefits
- ✅ **No separate WAV dataset loading** needed
- ✅ **Memory efficient** - datasets not loaded separately
- ✅ **Simpler code** - no DatasetMapper complexity
- ✅ **Faster loading** - single dataset load
- ✅ **Always available** - both features and audio in same dataset

## Usage in Config

Set the audio encoder type in your YAML config:

```yaml
# Use pre-extracted Emotion2Vec features (fast, recommended)
audio_encoder_type: "preextracted"

# Use raw audio with Wav2Vec2 encoder
audio_encoder_type: "wav2vec2"
audio_model_name: "facebook/wav2vec2-base"

# Use raw audio with HuBERT encoder
audio_encoder_type: "hubert"
audio_model_name: "facebook/hubert-base-ls960"
```

## Testing

Test that merged datasets work:

```bash
python3 dataset_mapper.py
```

Expected output:
- ✅ Loads merged dataset successfully
- ✅ Shows audio data structure
- ✅ Shows pre-extracted features available
- ✅ Shows text transcript available

## Impact

Now when you run experiments with merged datasets:

```yaml
audio_encoder_type: "wav2vec2"     # Loads RAW audio from merged dataset
audio_encoder_type: "hubert"       # Loads RAW audio from merged dataset
audio_encoder_type: "preextracted" # Uses pre-extracted features from merged dataset
```

Each encoder type will process **different audio representations**, leading to different results!

## Migration from Old Approach

If you have old code using separate feature and WAV datasets:

1. **Update dataset names** in `main.py`:
   ```python
   # OLD (separate datasets)
   feature_dataset_map = {
       "MSPI": "cairocode/MSPI_Emotion2Vec_Text"
   }

   # NEW (merged datasets)
   merged_dataset_map = {
       "MSPI": "cairocode/MSPI_Audio_Text_Merged"
   }
   ```

2. **Remove DatasetMapper usage** - audio is now in the dataset directly

3. **Update audio access** - use `item["audio"]` instead of looking up separately

## Summary

✅ **Merged datasets** simplify audio loading
✅ **Both features and audio** available in one dataset
✅ **Memory efficient** - no separate dataset loading
✅ **Simpler code** - direct column access
✅ **Backward compatible** - configs work the same way
