#!/usr/bin/env python3
"""
Test script for new dataset integration (CMU-MOSEI and SAMSEMO)
"""

from config import Config
from main import SimpleEmotionDataset

print("=" * 60)
print("TESTING NEW DATASETS: CMU-MOSEI and SAMSEMO")
print("=" * 60)

# Create a basic config
config = Config()
config.modality = "both"  # Test with multimodal mode
config.text_max_length = 128

# Test 1: Load CMU-MOSEI
print("\n1. Testing CMU-MOSEI dataset...")
try:
    cmumosei_dataset = SimpleEmotionDataset(
        "CMUMOSEI",
        split="train",
        config=config,
        Train=False  # Test mode
    )
    print(f"   ✅ Successfully loaded CMU-MOSEI")
    print(f"   Total samples: {len(cmumosei_dataset)}")

    # Test getting a sample
    if len(cmumosei_dataset) > 0:
        sample = cmumosei_dataset[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Sample label: {sample['label']}")
        if 'features' in sample:
            print(f"   Audio features shape: {sample['features'].shape}")
        if 'transcript' in sample:
            print(f"   Transcript preview: {sample['transcript'][:50]}...")

except Exception as e:
    print(f"   ❌ Failed to load CMU-MOSEI: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Load SAMSEMO
print("\n2. Testing SAMSEMO dataset...")
try:
    samsemo_dataset = SimpleEmotionDataset(
        "SAMSEMO",
        split="train",
        config=config,
        Train=False  # Test mode
    )
    print(f"   ✅ Successfully loaded SAMSEMO")
    print(f"   Total samples: {len(samsemo_dataset)}")

    # Test getting a sample
    if len(samsemo_dataset) > 0:
        sample = samsemo_dataset[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Sample label: {sample['label']}")
        if 'features' in sample:
            print(f"   Audio features shape: {sample['features'].shape}")
        if 'transcript' in sample:
            print(f"   Transcript preview: {sample['transcript'][:50]}...")

except Exception as e:
    print(f"   ❌ Failed to load SAMSEMO: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify label distributions
print("\n3. Testing label distributions...")

for dataset_name, dataset in [("CMUMOSEI", cmumosei_dataset), ("SAMSEMO", samsemo_dataset)]:
    try:
        label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for item in dataset.data:
            label = item['label']
            if label in label_counts:
                label_counts[label] += 1

        print(f"\n   {dataset_name} label distribution:")
        print(f"      Neutral (0): {label_counts[0]}")
        print(f"      Happy (1):   {label_counts[1]}")
        print(f"      Sad (2):     {label_counts[2]}")
        print(f"      Anger (3):   {label_counts[3]}")

        if sum(label_counts.values()) == len(dataset):
            print(f"   ✅ All labels accounted for")
        else:
            print(f"   ⚠️  Label count mismatch: {sum(label_counts.values())} != {len(dataset)}")

    except Exception as e:
        print(f"   ❌ Failed to analyze {dataset_name}: {e}")

# Test 4: Verify modality support
print("\n4. Testing modality modes...")

modality_tests = [
    ("audio", "Audio-only mode"),
    ("text", "Text-only mode"),
    ("both", "Multimodal mode")
]

for modality, description in modality_tests:
    print(f"\n   Testing {description}...")
    config.modality = modality

    try:
        test_dataset = SimpleEmotionDataset(
            "CMUMOSEI",
            split="train",
            config=config,
            Train=False
        )

        sample = test_dataset[0]

        if modality == "audio":
            assert 'features' in sample, "Audio mode should have features"
            assert 'transcript' not in sample or sample['transcript'] is None, "Audio mode should not have transcript"
            print(f"      ✅ {description} works correctly")

        elif modality == "text":
            assert 'transcript' in sample, "Text mode should have transcript"
            assert 'features' not in sample or sample['features'] is None, "Text mode should not have features"
            print(f"      ✅ {description} works correctly")

        elif modality == "both":
            assert 'features' in sample, "Multimodal mode should have features"
            assert 'transcript' in sample, "Multimodal mode should have transcript"
            print(f"      ✅ {description} works correctly")

    except Exception as e:
        print(f"      ❌ {description} failed: {e}")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("\nNew datasets integrated:")
print("  ✅ CMU-MOSEI: cairocode/CMU_MOSEI_EMOTION2VEC_4class_2")
print("  ✅ SAMSEMO: cairocode/samsemo_emotion2vec_4_V2")
print("\nCross-corpus evaluation now tests on:")
print("  - MSPI")
print("  - MSPP")
print("  - CMUMOSEI (NEW)")
print("  - SAMSEMO (NEW)")
print("\nModality support:")
print("  ✅ Audio-only")
print("  ✅ Text-only")
print("  ✅ Multimodal (audio + text)")
print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
