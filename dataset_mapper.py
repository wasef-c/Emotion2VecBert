#!/usr/bin/env python3
"""
Dataset utilities for emotion recognition with merged datasets

All datasets now contain both pre-extracted features AND raw audio in a single dataset.
No need for complex mapping - just load and use!
"""

from datasets import load_dataset


# Merged dataset mappings - each contains features + audio + text + labels
MERGED_DATASET_MAP = {
    "IEMO": "cairocode/IEMO_Audio_Text_Merged",
    "MSPI": "cairocode/MSPI_Audio_Text_Merged",
    "MSPP": "cairocode/MSPP_WAV_Filtered_Ordered_v2",
    "CMUMOSEI": "cairocode/cmu_mosei_wav",
    "SAMSEMO": "cairocode/samsemo_audio",
}


def load_merged_dataset(dataset_name: str, split: str = "train"):
    """
    Load a merged dataset containing features, audio, text, and labels

    Args:
        dataset_name: Name like "IEMO", "MSPI", "MSPP", "CMUMOSEI", or "SAMSEMO"
        split: Dataset split (default: "train")

    Returns:
        HuggingFace Dataset with columns:
          - emotion2vec_features: Pre-extracted audio features (for fast training)
          - audio: Raw audio data {"array": [...], "sampling_rate": 16000}
          - transcript/text: Text transcription
          - label: Emotion label (0=angry, 1=happy, 2=neutral, 3=sad)
          - Other metadata (speaker_id, valence, arousal, etc.)

    Example:
        >>> # Load dataset
        >>> dataset = load_merged_dataset("MSPI", split="train")
        >>> sample = dataset[0]
        >>>
        >>> # Use pre-extracted features (fast)
        >>> features = sample["emotion2vec_features"][0]["feats"]
        >>>
        >>> # Or use raw audio (for wav2vec2/hubert)
        >>> audio_array = sample["audio"]["array"]
        >>> sampling_rate = sample["audio"]["sampling_rate"]
        >>>
        >>> # Get text and label
        >>> transcript = sample["transcript"]
        >>> label = sample["label"]
    """
    if dataset_name not in MERGED_DATASET_MAP:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(MERGED_DATASET_MAP.keys())}"
        )

    dataset_path = MERGED_DATASET_MAP[dataset_name]
    print(f"📥 Loading {dataset_name} from {dataset_path}")

    dataset = load_dataset(dataset_path, split=split, trust_remote_code=True)

    print(f"✅ Loaded {len(dataset)} samples")
    print(f"   Columns: {dataset.column_names}")

    return dataset


def get_dataset_info(dataset_name: str, split: str = "train"):
    """
    Get information about a merged dataset without loading all data

    Args:
        dataset_name: Dataset name
        split: Dataset split

    Returns:
        Dictionary with dataset information
    """
    dataset = load_merged_dataset(dataset_name, split)
    sample = dataset[0]

    info = {
        "dataset_name": dataset_name,
        "dataset_path": MERGED_DATASET_MAP[dataset_name],
        "split": split,
        "num_samples": len(dataset),
        "columns": dataset.column_names,
        "has_audio": "audio" in sample,
        "has_features": "emotion2vec_features" in sample,
        "has_transcript": "transcript" in sample or "text" in sample,
        "sample_keys": list(sample.keys()),
    }

    # Check audio structure
    if "audio" in sample and sample["audio"] is not None:
        audio = sample["audio"]
        if isinstance(audio, dict):
            info["audio_keys"] = list(audio.keys())
            if "array" in audio:
                info["audio_length"] = len(audio["array"])
            if "sampling_rate" in audio:
                info["sampling_rate"] = audio["sampling_rate"]

    return info


if __name__ == "__main__":
    print("=" * 70)
    print("MERGED DATASETS - Testing")
    print("=" * 70)
    print()

    # List available datasets
    print("📋 Available Merged Datasets:")
    for name, path in MERGED_DATASET_MAP.items():
        print(f"   {name:12} → {path}")
    print()

    # Test loading a dataset
    print("=" * 70)
    print("Testing MSPI merged dataset")
    print("=" * 70)
    print()

    try:
        dataset = load_merged_dataset("MSPI", split="train")
        sample = dataset[0]

        print(f"📊 Sample structure:")
        print(f"   Keys: {list(sample.keys())}")
        print()

        # Check for audio
        if "audio" in sample and sample["audio"] is not None:
            audio = sample["audio"]
            if isinstance(audio, dict):
                print(f"🎵 Audio data:")
                print(f"   Format: dict")
                print(f"   Keys: {list(audio.keys())}")
                if "array" in audio:
                    print(f"   Array length: {len(audio['array'])} samples")
                    print(f"   Duration: {len(audio['array']) / audio.get('sampling_rate', 16000):.2f} seconds")
                if "sampling_rate" in audio:
                    print(f"   Sampling rate: {audio['sampling_rate']} Hz")
                print()

        # Check for features
        if "emotion2vec_features" in sample:
            print(f"🔊 Pre-extracted features:")
            print(f"   Available: Yes")
            features = sample["emotion2vec_features"]
            if isinstance(features, list) and len(features) > 0:
                feats = features[0].get("feats", [])
                print(f"   Shape: {len(feats)} features")
            print()

        # Check for text
        text = sample.get("transcript") or sample.get("text")
        if text:
            print(f"📝 Text transcript:")
            print(f"   Text: \"{text[:100]}...\"" if len(text) > 100 else f"   Text: \"{text}\"")
            print()

        # Check for label
        if "label" in sample:
            label_names = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
            label = sample["label"]
            print(f"🎯 Label:")
            print(f"   Value: {label} ({label_names.get(label, 'unknown')})")
            print()

        print("✅ Merged dataset test completed!")
        print()

        # Show usage examples
        print("=" * 70)
        print("Usage Examples")
        print("=" * 70)
        print()
        print("# Load dataset")
        print('dataset = load_merged_dataset("MSPI", split="train")')
        print("sample = dataset[0]")
        print()
        print("# Use pre-extracted features (fast training)")
        print('features = sample["emotion2vec_features"][0]["feats"]')
        print()
        print("# Use raw audio (for wav2vec2/hubert)")
        print('audio_array = sample["audio"]["array"]')
        print('sampling_rate = sample["audio"]["sampling_rate"]')
        print()
        print("# Get text and label")
        print('transcript = sample["transcript"]')
        print('label = sample["label"]  # 0=angry, 1=happy, 2=neutral, 3=sad')

    except Exception as e:
        print(f"❌ Error testing merged dataset: {e}")
        import traceback
        traceback.print_exc()
