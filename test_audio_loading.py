#!/usr/bin/env python3
"""
Test script to verify audio loading works correctly for different encoder types
"""

import sys
from config import Config
from main import SimpleEmotionDataset

def test_audio_loading():
    """Test that different audio encoder types load different data"""

    print("="*60)
    print("Testing Audio Loading for Different Encoder Types")
    print("="*60)

    # Test 1: Pre-extracted features
    print("\n1. Testing PREEXTRACTED features...")
    config_preextracted = Config()
    config_preextracted.audio_encoder_type = "preextracted"
    config_preextracted.modality = "both"

    try:
        dataset_preextracted = SimpleEmotionDataset("MSPP", split="train", config=config_preextracted, Train=True)
        sample = dataset_preextracted[0]
        print(f"   ✅ Loaded {len(dataset_preextracted)} samples")
        print(f"   Features type: {type(sample['features'])}")
        print(f"   Features shape: {sample['features'].shape if hasattr(sample['features'], 'shape') else 'N/A'}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Wav2Vec2 with raw audio
    print("\n2. Testing WAV2VEC2 with raw audio...")
    config_wav2vec = Config()
    config_wav2vec.audio_encoder_type = "wav2vec2"
    config_wav2vec.modality = "both"

    try:
        dataset_wav2vec = SimpleEmotionDataset("MSPP", split="train", config=config_wav2vec, Train=True)
        sample = dataset_wav2vec[0]
        print(f"   ✅ Loaded {len(dataset_wav2vec)} samples")
        print(f"   Features type: {type(sample['features'])}")
        if isinstance(sample['features'], dict):
            print(f"   Features keys: {sample['features'].keys()}")
            if 'array' in sample['features']:
                print(f"   Audio array shape: {sample['features']['array'].shape if hasattr(sample['features']['array'], 'shape') else len(sample['features']['array'])}")
                print(f"   Sampling rate: {sample['features'].get('sampling_rate', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: HuBERT with raw audio
    print("\n3. Testing HUBERT with raw audio...")
    config_hubert = Config()
    config_hubert.audio_encoder_type = "hubert"
    config_hubert.modality = "both"

    try:
        dataset_hubert = SimpleEmotionDataset("MSPP", split="train", config=config_hubert, Train=True)
        sample = dataset_hubert[0]
        print(f"   ✅ Loaded {len(dataset_hubert)} samples")
        print(f"   Features type: {type(sample['features'])}")
        if isinstance(sample['features'], dict):
            print(f"   Features keys: {sample['features'].keys()}")
            if 'array' in sample['features']:
                print(f"   Audio array shape: {sample['features']['array'].shape if hasattr(sample['features']['array'], 'shape') else len(sample['features']['array'])}")
                print(f"   Sampling rate: {sample['features'].get('sampling_rate', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    print("\n✅ If you see different feature types above, the fix is working!")
    print("   - Preextracted should show torch.Tensor with shape like [seq_len, 768]")
    print("   - Wav2Vec2/HuBERT should show dict with 'array' and 'sampling_rate'")

if __name__ == "__main__":
    test_audio_loading()
