#!/usr/bin/env python3
"""
Merge feature datasets with WAV datasets and upload to HuggingFace Hub
This version uses Pandas for merging and copies audio arrays directly
from HF 'audio' columns. Memory-safe and avoids TorchCodec.

Usage:
    python merge_and_upload_dataset.py --feature-dataset <FEATURE_DS> \
                                       --wav-dataset <WAV_DS> \
                                       --output-name <OUTPUT_DS>
"""

import argparse
import sys
import gc
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


# ---------------------------
# Main merging function
# ---------------------------


def merge_feature_and_wav(
    feature_dataset_name: str,
    wav_dataset_name: Optional[str] = None,
    split: str = "train",
):
    """Merge a feature dataset and optional WAV dataset using Pandas."""
    print(f"\n{'='*60}")
    print(f"Merging split: {split}")
    print(f"{'='*60}\n")

    # Load feature dataset
    print("📥 Loading feature dataset...")
    feature_ds = load_dataset(feature_dataset_name, split=split)
    feature_df = pd.DataFrame(feature_ds)

    # Load WAV dataset if provided
    wav_df = None
    if wav_dataset_name:
        print("📥 Loading WAV dataset...")
        wav_ds = load_dataset(wav_dataset_name, split=split)
        wav_df = pd.DataFrame(wav_ds)

        if "audio" not in wav_df.columns:
            raise ValueError("WAV dataset must have an 'audio' column")

        # Extract the raw array from HF Audio column
        print("🔹 Extracting audio arrays from 'audio' column...")

        def extract_array(row):

            return row["audio"].get("array")

        wav_df["audio_array"] = wav_df.apply(extract_array, axis=1)

    # Determine ID column
    id_column = None
    ID_COLUMNS = [
        "utterance_id",
        "utterance",
        "video",
        "filename",
        "file_name",
        "video_id",
        "FileName",
    ]
    for col in ID_COLUMNS:
        if col in feature_df.columns:
            id_column = col
            break
    if id_column is None:
        raise ValueError("No ID column found in feature dataset")
    print(f"✅ Using ID column: {id_column}")

    # Merge datasets
    if wav_df is not None:
        print(f"🔹 Merging feature and WAV datasets on '{id_column}'...")
        merged_df = feature_df.merge(
            wav_df[[id_column, "audio_array"]], on=id_column, how="inner"
        )
        skipped = len(feature_df) - len(merged_df)
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} samples (no matching audio)")
    else:
        merged_df = feature_df

    # Convert audio array to list (HF Dataset-friendly)
    if "audio_array" in merged_df.columns:
        merged_df["audio"] = merged_df["audio_array"].apply(
            lambda x: x.tolist() if x is not None else None
        )
        # merged_df.drop(columns=["audio_array"], inplace=True)

    # Convert back to HF Dataset
    print("📦 Converting merged DataFrame to HuggingFace Dataset...")
    merged_ds = Dataset.from_pandas(merged_df)

    print(f"✅ Merged {len(merged_ds)} samples")
    return merged_ds


# ---------------------------
# Merge all splits and optionally push to Hub
# ---------------------------


def merge_and_upload(
    feature_dataset_name: str,
    output_dataset_name: str,
    wav_dataset_name: Optional[str] = None,
    splits: list = ["train"],
    push_to_hub: bool = True,
):
    merged_datasets = {}

    for split in splits:
        try:
            merged_ds = merge_feature_and_wav(
                feature_dataset_name=feature_dataset_name,
                wav_dataset_name=wav_dataset_name,
                split=split,
            )
            merged_datasets[split] = merged_ds
        except Exception as e:
            print(f"\n❌ Error processing {split}: {e}")
            continue

    if not merged_datasets:
        print("\n❌ No splits were successfully merged!")
        return

    dataset_dict = DatasetDict(merged_datasets)

    # Show dataset info
    print("\n📊 MERGED DATASET INFO")
    for split, ds in dataset_dict.items():
        print(f"{split}: {len(ds)} samples, columns: {ds.column_names}")

    # Upload to Hub
    if push_to_hub:
        print(f"\n📤 Pushing to HuggingFace Hub: {output_dataset_name}")
        try:
            dataset_dict.push_to_hub(output_dataset_name, private=False)
            print(f"✅ Uploaded: https://huggingface.co/datasets/{output_dataset_name}")
        except Exception as e:
            print(f"❌ Error uploading: {e}")
            print("💡 Make sure you're logged in: huggingface-cli login")
    else:
        print("\n⚠️ Skipping upload (push_to_hub=False)")

    return dataset_dict


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge feature + audio datasets")
    parser.add_argument(
        "--feature-dataset", type=str, required=True, help="Feature dataset name (HF)"
    )
    parser.add_argument(
        "--wav-dataset", type=str, default=None, help="Optional WAV dataset"
    )
    parser.add_argument(
        "--output-name", type=str, required=True, help="Output dataset name on Hub"
    )
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--no-upload", action="store_true", help="Skip upload")

    args = parser.parse_args()

    try:
        merge_and_upload(
            feature_dataset_name=args.feature_dataset,
            wav_dataset_name=args.wav_dataset,
            output_dataset_name=args.output_name,
            splits=args.splits,
            push_to_hub=not args.no_upload,
        )
        print("\n✅ ALL DONE!")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback

        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
