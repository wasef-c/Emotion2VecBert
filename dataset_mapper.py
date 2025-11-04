#!/usr/bin/env python3
"""
Dataset mapping utilities for linking WAV datasets to feature datasets
Maps using common ID columns: video, filename, utterance, utterance_id
"""

from datasets import load_dataset
from typing import Dict, List, Optional, Tuple


class DatasetMapper:
    """
    Maps WAV datasets to feature datasets using common ID columns
    """

    # Mapping of dataset names to their WAV dataset counterparts
    WAV_DATASET_MAP = {
        "IEMO": [],  # Add IEMOCAP WAV dataset names here
        "MSPI": ["cairocode/MSPI_WAV_Diff"],
        "MSPP": ["cairocode/MSPP_WAV_Filtered_Ordered_v2"],
        "CMUMOSEI": ["cairocode/cmu_mosei_wav_2"],
        "SAMSEMO": [],  # Add SAMSEMO WAV dataset names here
    }

    # Common ID column names to try (in order of preference)
    ID_COLUMNS = ["utterance_id", "utterance", "video", "filename", "file_name", "video_id"]

    def __init__(self, dataset_name: str, feature_dataset_name: str, wav_dataset_name: Optional[str] = None):
        """
        Initialize dataset mapper

        Args:
            dataset_name: Name of the dataset (e.g., "MSPI", "MSPP")
            feature_dataset_name: HuggingFace dataset name with features (e.g., "cairocode/MSPI_Emotion2Vec_Text")
            wav_dataset_name: Optional specific WAV dataset name. If None, will auto-select
        """
        self.dataset_name = dataset_name
        self.feature_dataset_name = feature_dataset_name

        # Auto-select WAV dataset if not provided
        if wav_dataset_name is None:
            if dataset_name in self.WAV_DATASET_MAP:
                wav_dataset_name = self.WAV_DATASET_MAP[dataset_name][0]  # Use first option
                print(f"📁 Auto-selected WAV dataset: {wav_dataset_name}")
            else:
                raise ValueError(f"No WAV dataset mapping found for {dataset_name}")

        self.wav_dataset_name = wav_dataset_name
        self.id_column = None  # Will be detected

    def find_common_id_column(self, feature_ds, wav_ds) -> str:
        """
        Find common ID column between feature and WAV datasets

        Args:
            feature_ds: Feature dataset
            wav_ds: WAV dataset

        Returns:
            Common column name

        Raises:
            ValueError if no common column found
        """
        feature_cols = set(feature_ds.column_names)
        wav_cols = set(wav_ds.column_names)

        # Try each ID column in order of preference
        for id_col in self.ID_COLUMNS:
            if id_col in feature_cols and id_col in wav_cols:
                print(f"✅ Found common ID column: '{id_col}'")
                return id_col

        # If no exact match, try to find similar columns
        common_cols = feature_cols & wav_cols
        if common_cols:
            print(f"⚠️ No standard ID column found. Common columns: {common_cols}")
            # Use first common column
            id_col = list(common_cols)[0]
            print(f"   Using '{id_col}' as ID column")
            return id_col

        raise ValueError(
            f"No common ID column found between datasets.\n"
            f"Feature columns: {feature_cols}\n"
            f"WAV columns: {wav_cols}"
        )

    def load_datasets(self, split="train") -> Tuple[any, any, str]:
        """
        Load both feature and WAV datasets and find common ID column

        Args:
            split: Dataset split to load

        Returns:
            Tuple of (feature_dataset, wav_dataset, id_column)
        """
        print(f"📚 Loading datasets for split: {split}")
        print(f"   Feature dataset: {self.feature_dataset_name}")
        print(f"   WAV dataset: {self.wav_dataset_name}")

        # Load datasets
        feature_ds = load_dataset(self.feature_dataset_name, split=split, trust_remote_code=True)
        wav_ds = load_dataset(self.wav_dataset_name, split=split, trust_remote_code=True)

        print(f"   Feature dataset size: {len(feature_ds)}")
        print(f"   WAV dataset size: {len(wav_ds)}")

        # Find common ID column
        id_column = self.find_common_id_column(feature_ds, wav_ds)
        self.id_column = id_column

        return feature_ds, wav_ds, id_column

    def create_mapping_dict(self, feature_ds, wav_ds, id_column: str) -> Dict:
        """
        Create mapping dictionary from ID to WAV audio

        Args:
            feature_ds: Feature dataset
            wav_ds: WAV dataset
            id_column: Common ID column name

        Returns:
            Dictionary mapping ID to WAV audio data
        """
        print(f"🗺️  Creating mapping dictionary using column '{id_column}'...")

        wav_map = {}
        for item in wav_ds:
            item_id = item[id_column]
            wav_map[item_id] = item

        print(f"   Created mapping for {len(wav_map)} WAV files")

        # Check coverage
        feature_ids = set(item[id_column] for item in feature_ds)
        wav_ids = set(wav_map.keys())

        matched = len(feature_ids & wav_ids)
        total = len(feature_ids)

        print(f"   Matched {matched}/{total} samples ({matched/total*100:.1f}%)")

        if matched < total:
            missing = total - matched
            print(f"   ⚠️ {missing} samples from feature dataset not found in WAV dataset")

        return wav_map

    def get_audio_for_sample(self, sample: Dict, wav_map: Dict, id_column: str) -> Optional[Dict]:
        """
        Get WAV audio data for a given sample

        Args:
            sample: Sample from feature dataset
            wav_map: Mapping dictionary
            id_column: ID column name

        Returns:
            WAV audio data or None if not found
        """
        sample_id = sample[id_column]
        return wav_map.get(sample_id)

    @staticmethod
    def get_wav_dataset_for_dataset(dataset_name: str) -> List[str]:
        """
        Get available WAV datasets for a given dataset name

        Args:
            dataset_name: Dataset name (e.g., "MSPI")

        Returns:
            List of available WAV dataset names
        """
        return DatasetMapper.WAV_DATASET_MAP.get(dataset_name, [])


def create_merged_dataset_info(dataset_name: str, feature_dataset_name: str, wav_dataset_name: Optional[str] = None, split: str = "train"):
    """
    Utility function to inspect and merge dataset information

    Args:
        dataset_name: Dataset name (e.g., "MSPI")
        feature_dataset_name: Feature dataset HuggingFace name
        wav_dataset_name: Optional WAV dataset name
        split: Dataset split

    Returns:
        Dictionary with merged dataset info
    """
    mapper = DatasetMapper(dataset_name, feature_dataset_name, wav_dataset_name)
    feature_ds, wav_ds, id_column = mapper.load_datasets(split)
    wav_map = mapper.create_mapping_dict(feature_ds, wav_ds, id_column)

    # Get sample statistics
    sample = feature_ds[0]
    wav_sample = mapper.get_audio_for_sample(sample, wav_map, id_column)

    info = {
        "dataset_name": dataset_name,
        "feature_dataset": feature_dataset_name,
        "wav_dataset": wav_dataset_name,
        "split": split,
        "id_column": id_column,
        "feature_columns": feature_ds.column_names,
        "wav_columns": wav_ds.column_names,
        "num_samples": len(feature_ds),
        "num_matched": len(set(item[id_column] for item in feature_ds) & set(wav_map.keys())),
        "sample_feature_keys": list(sample.keys()),
        "sample_wav_keys": list(wav_sample.keys()) if wav_sample else None,
    }

    return info


if __name__ == "__main__":
    # Test the dataset mapper
    print("Testing DatasetMapper...\n")

    # Test MSPI dataset mapping
    print("="*60)
    print("Testing MSPI dataset mapping")
    print("="*60)

    try:
        info = create_merged_dataset_info(
            dataset_name="MSPI",
            feature_dataset_name="cairocode/MSPI_Emotion2Vec_Text",
            wav_dataset_name="cairocode/MSPI_WAV_Diff",
            split="train"
        )

        print(f"\n📊 Dataset Info:")
        print(f"   Dataset: {info['dataset_name']}")
        print(f"   ID Column: {info['id_column']}")
        print(f"   Samples: {info['num_samples']}")
        print(f"   Matched: {info['num_matched']}")
        print(f"\n   Feature columns: {info['feature_columns'][:5]}...")
        print(f"   WAV columns: {info['wav_columns'][:5]}...")

    except Exception as e:
        print(f"❌ Error testing MSPI: {e}")

    # Test MSPP dataset mapping
    print(f"\n{'='*60}")
    print("Testing MSPP dataset mapping")
    print("="*60)

    try:
        info = create_merged_dataset_info(
            dataset_name="MSPP",
            feature_dataset_name="cairocode/MSPP_Emotion2Vec_Text",
            wav_dataset_name="cairocode/MSPP_WAV_Filtered_Ordered_v2",
            split="train"
        )

        print(f"\n📊 Dataset Info:")
        print(f"   Dataset: {info['dataset_name']}")
        print(f"   ID Column: {info['id_column']}")
        print(f"   Samples: {info['num_samples']}")
        print(f"   Matched: {info['num_matched']}")

    except Exception as e:
        print(f"❌ Error testing MSPP: {e}")

    # List available WAV datasets
    print(f"\n{'='*60}")
    print("Available WAV datasets:")
    print("="*60)
    for dataset, wav_datasets in DatasetMapper.WAV_DATASET_MAP.items():
        print(f"\n{dataset}:")
        for wav_ds in wav_datasets:
            print(f"  - {wav_ds}")

    print("\n✅ DatasetMapper tests completed!")
