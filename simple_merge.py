#!/usr/bin/env python3
"""
Simple dataset merger - Load audio dataset and merge with feature dataset
"""

import argparse
import gc
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm


# Dataset mappings
DATASET_MAP = {
    "IEMO": {
        "feature": "cairocode/IEMO_Emotion2Vec_Text",
        "audio": "cairocode/IEMO_WAV_002",
        "id_column": "utterance_id"
    },
    "MSPI": {
        "feature": "cairocode/MSPI_Emotion2Vec_Text",
        "audio": "cairocode/MSPI_WAV_Diff",
        "id_column": "utterance_id"
    },
    "MSPP": {
        "feature": "cairocode/MSPP_Emotion2Vec_Text",
        "audio": "cairocode/MSPP_WAV_Filtered_Ordered_v2",
        "id_column": "utterance_id"
    },
    "CMUMOSEI": {
        "feature": "cairocode/CMU_MOSEI_EMOTION2VEC_4class_2",
        "audio": "cairocode/cmu_mosei_wav_2",
        "id_column": "video"
    },
    "SAMSEMO": {
        "feature": "cairocode/samsemo_emotion2vec_4_V2",
        "audio": "cairocode/samsemo-audio",
        "id_column": "utterance_id"
    }
}


def merge_dataset(dataset_name: str, output_name: str, split: str = "train"):
    """
    Simple merge: Load audio dataset, create mapping, iterate and merge

    Args:
        dataset_name: Name like "MSPI", "MSPP", etc.
        output_name: Output HuggingFace dataset name
        split: Dataset split (default: "train")
    """
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_MAP.keys())}")

    config = DATASET_MAP[dataset_name]

    print(f"\n{'='*70}")
    print(f"Merging {dataset_name}")
    print(f"{'='*70}")
    print(f"Feature dataset: {config['feature']}")
    print(f"Audio dataset: {config['audio']}")
    print(f"ID column: {config['id_column']}")
    print(f"Split: {split}")

    # Load datasets
    print("\n📥 Loading feature dataset...")
    feature_ds = load_dataset(config['feature'], split=split, trust_remote_code=True)
    print(f"   Loaded {len(feature_ds)} samples")

    print("\n📥 Loading audio dataset...")
    audio_ds = load_dataset(config['audio'], split=split, trust_remote_code=True)
    print(f"   Loaded {len(audio_ds)} samples")

    # Create audio mapping: id -> audio data
    print(f"\n🗺️  Creating audio mapping...")
    id_col = config['id_column']
    audio_map = {}

    for sample in tqdm(audio_ds, desc="Mapping audio"):
        sample_id = sample[id_col]
        audio_map[sample_id] = sample['audio']

    print(f"   Mapped {len(audio_map)} audio files")

    # Clear audio dataset from memory
    del audio_ds
    gc.collect()

    # Merge: Iterate through feature dataset and add audio
    print(f"\n🔗 Merging datasets...")
    merged_samples = []
    skipped = 0

    for sample in tqdm(feature_ds, desc="Merging"):
        sample_id = sample[id_col]

        # Get matching audio
        audio_data = audio_map.get(sample_id)

        if audio_data is None:
            skipped += 1
            continue

        # Create merged sample
        merged_sample = {
            **sample,  # All feature columns
            "audio": audio_data  # Add audio
        }
        merged_samples.append(merged_sample)

    print(f"\n✅ Merged {len(merged_samples)} samples")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} samples (no matching audio)")

    # Create HuggingFace Dataset
    print("\n📦 Creating HuggingFace Dataset...")
    merged_ds = Dataset.from_list(merged_samples)

    print(f"   Dataset created with {len(merged_ds)} samples")
    print(f"   Columns: {merged_ds.column_names}")

    # Upload to Hub
    print(f"\n📤 Pushing to HuggingFace Hub: {output_name}")
    try:
        merged_ds.push_to_hub(output_name, private=False)
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/datasets/{output_name}")
    except Exception as e:
        print(f"\n❌ Error uploading: {e}")
        print("💡 Make sure you're logged in: huggingface-cli login")
        raise

    return merged_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple dataset merger")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_MAP.keys()),
        help="Dataset to merge"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset name (e.g., cairocode/MSPI_Merged)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to process (default: train)"
    )

    args = parser.parse_args()

    try:
        merge_dataset(args.dataset, args.output, args.split)
        print("\n" + "="*70)
        print("✅ DONE!")
        print("="*70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
