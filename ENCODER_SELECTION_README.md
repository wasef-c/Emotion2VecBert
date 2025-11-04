# Audio and Text Encoder Selection Guide

Your system now supports flexible selection of different audio and text encoders! This allows you to experiment with models like RoBERTa, wav2vec2, HuBERT, and more.

## Quick Start

### Using Different Text Models

In your YAML config:

```yaml
modality: "text"  # or "both" for multimodal
text_model_name: "roberta-base"  # Instead of default BERT
freeze_text_encoder: true
```

**Supported Text Models:**
- `"bert-base-uncased"` (default)
- `"roberta-base"` or `"roberta-large"`
- `"distilbert-base-uncased"` (faster, smaller)
- `"microsoft/deberta-v3-base"`
- `"albert-base-v2"` or `"albert-large-v2"`
- Any HuggingFace transformer model

### Using Different Audio Models

**Option 1: Pre-extracted Features (Default, Fastest)**
```yaml
modality: "audio"  # or "both"
audio_encoder_type: "preextracted"
audio_dim: 768  # emotion2vec default
```

**Option 2: Wav2Vec2 (Requires WAV Dataset)**
```yaml
audio_encoder_type: "wav2vec2"
audio_model_name: "facebook/wav2vec2-base-960h"
freeze_audio_encoder: true
audio_pooling: "mean"
```

**Option 3: HuBERT (Requires WAV Dataset)**
```yaml
audio_encoder_type: "hubert"
audio_model_name: "facebook/hubert-base-ls960"
freeze_audio_encoder: true
audio_pooling: "mean"
```

## Configuration Parameters

### Audio Encoder Settings

| Parameter | Options | Description |
|-----------|---------|-------------|
| `audio_encoder_type` | `"preextracted"`, `"wav2vec2"`, `"hubert"`, `"emotion2vec"` | Type of audio encoder |
| `audio_model_name` | HuggingFace model name or `null` | Specific model to use (auto-selects if null) |
| `audio_dim` | Integer (e.g., 768) | Feature dimension (auto-detected for transformer models) |
| `freeze_audio_encoder` | `true`/`false` | Whether to freeze encoder weights |
| `audio_pooling` | `"mean"`, `"first"`, `"last"`, `"max"` | How to pool sequence features |

### Text Encoder Settings

| Parameter | Options | Description |
|-----------|---------|-------------|
| `text_model_name` | HuggingFace model name | Text encoder model |
| `freeze_text_encoder` | `true`/`false` | Whether to freeze encoder weights |
| `text_max_length` | Integer (e.g., 128) | Maximum text sequence length |

### Fusion Settings (for multimodal)

| Parameter | Options | Description |
|-----------|---------|-------------|
| `fusion_type` | `"cross_attention"`, `"concat"`, `"gated"`, `"adaptive"` | How to fuse audio and text |
| `fusion_hidden_dim` | Integer (e.g., 512) | Hidden dimension for fusion |
| `num_attention_heads` | Integer (e.g., 8) | Attention heads (for cross_attention) |

## Example Configurations

See `configs/model_encoder_examples.yaml` for comprehensive examples including:
- Text model variations (BERT, RoBERTa, DistilBERT, DeBERTa)
- Audio encoder variations (Wav2Vec2, HuBERT, Emotion2Vec)
- Audio-only and text-only experiments
- Different fusion strategies
- Fine-tuning experiments

## Running Experiments

```bash
# Run specific experiment by index
python main.py -c configs/model_encoder_examples.yaml -e 0

# Run by name
python main.py -c configs/model_encoder_examples.yaml -e "Wav2Vec2_BERT"

# Run all experiments
python main.py -c configs/model_encoder_examples.yaml -a
```

## WAV Dataset Mapping

For using raw audio encoders (wav2vec2, hubert), you need WAV datasets that can be mapped to your feature datasets.

**Available WAV Datasets:**
- MSPI: `cairocode/MSPI_WAV_Diff`
- MSPP: `cairocode/MSPP_WAV_Filtered_Ordered_v2`
- CMUMOSEI: `cairocode/cmu_mosei_wav_2`
- IEMO: (Add your WAV dataset name in `dataset_mapper.py`)
- SAMSEMO: (Add your WAV dataset name in `dataset_mapper.py`)

**To add WAV dataset mappings:**

Edit `dataset_mapper.py` and update the `WAV_DATASET_MAP`:

```python
WAV_DATASET_MAP = {
    "IEMO": ["your/iemo/wav/dataset"],
    "SAMSEMO": ["your/samsemo/wav/dataset"],
    # ... existing mappings
}
```

The mapper automatically finds common ID columns (`utterance_id`, `video`, `filename`, etc.) to link datasets.

## New Files Created

1. **`audio_encoder.py`**: Audio encoder module supporting multiple models
2. **`dataset_mapper.py`**: Utilities for mapping WAV datasets to feature datasets
3. **`configs/model_encoder_examples.yaml`**: Comprehensive configuration examples
4. **Updated `config.py`**: Added audio encoder parameters
5. **Updated `main.py`**: Support for loading audio encoder config
6. **Updated `model.py`**: Integrated audio encoder into models

## Performance Notes

- **Pre-extracted features**: Fastest, lowest memory (recommended for most experiments)
- **Frozen encoders**: Fast, moderate memory (good for exploring different models)
- **Fine-tuning**: Slow, high memory, but potentially better performance
  - Use lower learning rate (e.g., `5e-5`)
  - Requires more epochs
  - May need larger batch sizes for stability

## Backwards Compatibility

All existing configs and code continue to work! The default behavior is:
- Audio: Pre-extracted emotion2vec features
- Text: BERT-base-uncased (if using text)
- Everything frozen

## Need Help?

- See `configs/model_encoder_examples.yaml` for examples
- Check `audio_encoder.py` for audio encoder options
- Check `text_encoder.py` for text encoder implementation
- Run `python dataset_mapper.py` to test dataset mapping
