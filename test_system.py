#!/usr/bin/env python3
"""
Quick system test to verify all components are working
"""

import torch
import sys
from pathlib import Path

print("="*60)
print("MULTIMODAL EMOTION RECOGNITION SYSTEM TEST")
print("="*60)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from config import Config
    from text_encoder import FrozenBERTEncoder
    from fusion import get_fusion_module, CrossAttentionFusion
    from model import MultimodalEmotionClassifier, SimpleEmotionClassifier, create_model
    from functions import calculate_metrics, calculate_difficulty
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create config
print("\n2. Testing configuration...")
try:
    config = Config()
    config.modality = "both"
    config.fusion_type = "cross_attention"
    print(f"   ✅ Config created (modality={config.modality}, fusion={config.fusion_type})")
except Exception as e:
    print(f"   ❌ Config error: {e}")
    sys.exit(1)

# Test 3: Test text encoder
print("\n3. Testing text encoder...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    encoder = FrozenBERTEncoder().to(device)
    test_texts = ["I am very happy!", "This is sad."]
    features = encoder.encode_texts(test_texts, device=device)
    print(f"   ✅ Text encoder working (output shape: {features.shape})")
except Exception as e:
    print(f"   ❌ Text encoder error: {e}")
    sys.exit(1)

# Test 4: Test fusion modules
print("\n4. Testing fusion modules...")
try:
    batch_size = 4
    audio_features = torch.randn(batch_size, 768).to(device)
    text_features = torch.randn(batch_size, 768).to(device)

    for fusion_type in ["cross_attention", "concat", "gated"]:
        fusion_module = get_fusion_module(
            fusion_type=fusion_type,
            audio_dim=768,
            text_dim=768,
            hidden_dim=512
        ).to(device)
        fused = fusion_module(audio_features, text_features)
        print(f"   ✅ {fusion_type} fusion working (output: {fused.shape})")
except Exception as e:
    print(f"   ❌ Fusion module error: {e}")
    sys.exit(1)

# Test 5: Test models
print("\n5. Testing models...")
try:
    # Test audio-only model
    config_audio = Config()
    config_audio.modality = "audio"
    model_audio = create_model(config_audio).to(device)
    audio_input = torch.randn(batch_size, 768).to(device)
    logits_audio = model_audio(audio_input)
    print(f"   ✅ Audio-only model working (output: {logits_audio.shape})")

    # Test text-only model
    config_text = Config()
    config_text.modality = "text"
    model_text = MultimodalEmotionClassifier(
        modality="text",
        audio_dim=768
    ).to(device)
    text_input_ids = torch.randint(0, 30000, (batch_size, 50)).to(device)
    text_attention_mask = torch.ones(batch_size, 50).to(device)
    logits_text = model_text(
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask
    )
    print(f"   ✅ Text-only model working (output: {logits_text.shape})")

    # Test multimodal model
    config_both = Config()
    config_both.modality = "both"
    config_both.fusion_type = "cross_attention"
    model_both = MultimodalEmotionClassifier(
        modality="both",
        fusion_type="cross_attention",
        audio_dim=768
    ).to(device)
    logits_both = model_both(
        audio_features=audio_input,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask
    )
    print(f"   ✅ Multimodal model working (output: {logits_both.shape})")

except Exception as e:
    print(f"   ❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test config loading
print("\n6. Testing YAML config loading...")
try:
    import yaml
    config_path = Path("configs/multimodal_baseline.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        num_experiments = len(yaml_config.get('experiments', []))
        print(f"   ✅ YAML config loaded ({num_experiments} experiments found)")
    else:
        print(f"   ⚠️  Config file not found at {config_path}")
except Exception as e:
    print(f"   ❌ Config loading error: {e}")

# Test 7: Test utility functions
print("\n7. Testing utility functions...")
try:
    # Test calculate_metrics
    predictions = [0, 1, 2, 3, 0, 1]
    labels = [0, 1, 2, 2, 0, 1]
    metrics = calculate_metrics(predictions, labels)
    print(f"   ✅ Metrics calculation working (acc={metrics['accuracy']:.3f}, uar={metrics['uar']:.3f})")

    # Test calculate_difficulty
    expected_vad = {
        0: [3.0, 2.5, 3.0],
        1: [4.0, 3.8, 3.8],
        2: [1.8, 2.2, 2.0],
        3: [1.8, 4.2, 4.0]
    }
    item = {
        'label': 0,
        'valence': 3.2,
        'arousal': 2.3,
        'domination': 3.1
    }
    difficulty = calculate_difficulty(item, expected_vad, method="euclidean_distance")
    print(f"   ✅ Difficulty calculation working (difficulty={difficulty:.3f})")

except Exception as e:
    print(f"   ❌ Utility function error: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Memory check
print("\n8. Memory usage check...")
try:
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"   GPU Memory: {memory_allocated:.1f} MB allocated, {memory_reserved:.1f} MB reserved")
    else:
        print(f"   CPU mode - no GPU memory tracking")
except Exception as e:
    print(f"   ⚠️  Memory check error: {e}")

# Summary
print("\n" + "="*60)
print("SYSTEM TEST COMPLETE")
print("="*60)
print("\n✅ All core components are working correctly!")
print("\nNext steps:")
print("  1. Install requirements: pip install -r requirements.txt")
print("  2. Login to WandB: wandb login")
print("  3. Run audio baseline: python main.py --config configs/audio_only.yaml --experiment 0")
print("  4. Run text baseline: python main.py --config configs/text_only.yaml --experiment 0")
print("  5. Run multimodal: python main.py --config configs/multimodal_baseline.yaml --experiment 0")
print("\nFor full experiment suite:")
print("  python main.py --config configs/multimodal_baseline.yaml --all")
print("="*60)
