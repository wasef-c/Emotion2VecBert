#!/usr/bin/env python3
"""
Audio encoder module supporting multiple audio feature extraction models
Supports: wav2vec2, hubert, emotion2vec, and pre-extracted features
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
import warnings


class AudioEncoder(nn.Module):
    """
    Flexible audio encoder that can use different pretrained models or pre-extracted features
    """

    def __init__(
        self,
        encoder_type="emotion2vec",  # "wav2vec2", "hubert", "emotion2vec", "preextracted"
        model_name=None,
        output_dim=768,
        freeze=True,
        pooling="mean"  # "mean", "first", "last", "attention"
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.freeze = freeze
        self.pooling = pooling

        # Set default model names if not provided
        if model_name is None:
            model_name = self._get_default_model_name(encoder_type)

        self.model_name = model_name

        if encoder_type == "preextracted":
            # No model needed - features are already extracted
            print(f"✅ Using pre-extracted audio features (output_dim={output_dim})")
            self.model = None
            self.feature_extractor = None

        elif encoder_type in ["wav2vec2", "hubert", "emotion2vec"]:
            # Load pretrained transformer-based audio model
            print(f"📚 Loading audio model: {model_name} (type: {encoder_type})")

            try:
                self.model = AutoModel.from_pretrained(model_name)
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            except Exception as e:
                print(f"⚠️ Could not load with AutoFeatureExtractor, trying Wav2Vec2FeatureExtractor...")
                self.model = AutoModel.from_pretrained(model_name)
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

            # Get actual output dimension from model
            if hasattr(self.model, 'config'):
                self.output_dim = self.model.config.hidden_size
                print(f"   Model hidden size: {self.output_dim}")

            # Freeze model parameters if requested
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
                print(f"   Model frozen (freeze={freeze})")
            else:
                print(f"   ⚠️ Model will be fine-tuned (freeze={freeze})")

            print(f"✅ Audio encoder loaded (output_dim={self.output_dim}, pooling={pooling})")
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def _get_default_model_name(self, encoder_type):
        """Get default model name for each encoder type"""
        defaults = {
            "wav2vec2": "facebook/wav2vec2-base-960h",
            "hubert": "facebook/hubert-base-ls960",
            "emotion2vec": "emotion2vec/emotion2vec_base",
            "preextracted": None
        }
        return defaults.get(encoder_type)

    def forward(self, audio_input):
        """
        Extract features from audio

        Args:
            audio_input:
                - For preextracted: [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
                - For wav2vec2/hubert/emotion2vec: raw waveform [batch_size, samples] or
                  already processed features [batch_size, seq_len, hidden_dim]

        Returns:
            features: [batch_size, output_dim]
        """
        if self.encoder_type == "preextracted":
            # Features are already extracted
            if len(audio_input.shape) == 3:
                # [batch_size, seq_len, feature_dim] -> pool to [batch_size, feature_dim]
                return self._pool_features(audio_input)
            else:
                # Already [batch_size, feature_dim]
                return audio_input

        else:
            # Use transformer model to extract features
            if self.freeze:
                self.model.eval()
                with torch.no_grad():
                    return self._extract_with_model(audio_input)
            else:
                return self._extract_with_model(audio_input)

    def _extract_with_model(self, audio_input):
        """Extract features using the transformer model with memory optimization"""
        # Check if input is raw waveform or already processed features
        if len(audio_input.shape) == 2 and audio_input.shape[-1] > 1000:
            # Likely raw waveform [batch_size, samples]
            # Enable gradient checkpointing for memory efficiency during training
            if self.training and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # Use mixed precision if available
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(audio_input, return_dict=True)
        else:
            # Already processed features [batch_size, seq_len, hidden_dim]
            # This handles the case where features are pre-extracted but we still want pooling
            return self._pool_features(audio_input)

        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

        # Apply pooling (this reduces memory usage by collapsing sequence dimension)
        pooled_features = self._pool_features(hidden_states)

        # Clear intermediate outputs to free memory
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pooled_features

    def _pool_features(self, features):
        """
        Pool sequence features to fixed-size representation

        Args:
            features: [batch_size, seq_len, feature_dim]

        Returns:
            pooled: [batch_size, feature_dim]
        """
        if len(features.shape) == 2:
            # Already pooled
            return features

        if self.pooling == "mean":
            return features.mean(dim=1)
        elif self.pooling == "first":
            return features[:, 0, :]
        elif self.pooling == "last":
            return features[:, -1, :]
        elif self.pooling == "max":
            return features.max(dim=1)[0]
        else:
            # Default to mean
            return features.mean(dim=1)

    def process_waveform(self, waveform, sampling_rate=16000):
        """
        Process raw waveform through feature extractor

        Args:
            waveform: Raw audio waveform (can be list, numpy array, or tensor)
            sampling_rate: Sampling rate of audio

        Returns:
            processed: Processed features ready for model
        """
        if self.feature_extractor is None:
            raise ValueError("No feature extractor available for preextracted mode")

        # Process through feature extractor
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )

        return inputs

    def encode_waveform(self, waveform, sampling_rate=16000, device='cpu'):
        """
        Complete pipeline: process waveform + extract features

        Args:
            waveform: Raw audio waveform
            sampling_rate: Sampling rate of audio
            device: Device to place tensors on

        Returns:
            features: [batch_size, output_dim]
        """
        if self.encoder_type == "preextracted":
            raise ValueError("Cannot encode waveform in preextracted mode")

        # Process waveform
        inputs = self.process_waveform(waveform, sampling_rate)

        # Move to device
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            audio_values = inputs['input_values']
        else:
            audio_values = inputs.to(device)

        # Extract features
        features = self.forward(audio_values)

        return features

    def get_output_dim(self):
        """Get output feature dimension"""
        return self.output_dim


class AudioFeatureCache:
    """
    Utility class for extracting and caching audio features
    Useful for pre-extracting features to avoid repeated computation
    """

    def __init__(self, encoder_type="wav2vec2", model_name=None, device='cuda', freeze=True):
        self.encoder = AudioEncoder(
            encoder_type=encoder_type,
            model_name=model_name,
            freeze=freeze
        ).to(device)
        self.device = device
        self.feature_cache = {}

    def extract_features(self, audio_data, sampling_rate=16000, use_cache=True, cache_key=None):
        """
        Extract features with optional caching

        Args:
            audio_data: Audio waveform or list of waveforms
            sampling_rate: Sampling rate
            use_cache: Whether to use caching
            cache_key: Optional key for caching (if None, no caching)

        Returns:
            features: [batch_size, output_dim]
        """
        if use_cache and cache_key is not None:
            # Check cache
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key].to(self.device)

        # Extract features
        features = self.encoder.encode_waveform(audio_data, sampling_rate, self.device)

        # Update cache
        if use_cache and cache_key is not None:
            self.feature_cache[cache_key] = features.cpu()

        return features

    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        print("🧹 Audio feature cache cleared")

    def cache_size(self):
        """Get number of cached features"""
        return len(self.feature_cache)


if __name__ == "__main__":
    # Test the audio encoder
    print("Testing AudioEncoder...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test with pre-extracted features (most common use case)
    print(f"\n{'='*60}")
    print("Testing pre-extracted features mode...")
    encoder_preextracted = AudioEncoder(encoder_type="preextracted", output_dim=768).to(device)

    # Test with 2D features (already pooled)
    features_2d = torch.randn(8, 768).to(device)
    output_2d = encoder_preextracted(features_2d)
    print(f"Input shape (2D): {features_2d.shape}")
    print(f"Output shape: {output_2d.shape}")

    # Test with 3D features (sequence)
    features_3d = torch.randn(8, 100, 768).to(device)
    output_3d = encoder_preextracted(features_3d)
    print(f"Input shape (3D): {features_3d.shape}")
    print(f"Output shape: {output_3d.shape}")

    # Test wav2vec2 (if you want to extract from raw audio)
    print(f"\n{'='*60}")
    print("Testing wav2vec2 encoder...")
    try:
        encoder_wav2vec = AudioEncoder(
            encoder_type="wav2vec2",
            model_name="facebook/wav2vec2-base-960h",
            freeze=True
        ).to(device)

        # Test with dummy raw audio (16kHz, 3 seconds)
        raw_audio = torch.randn(2, 48000).to(device)
        output_wav2vec = encoder_wav2vec(raw_audio)
        print(f"Raw audio shape: {raw_audio.shape}")
        print(f"Output shape: {output_wav2vec.shape}")
        print(f"Output dim: {encoder_wav2vec.get_output_dim()}")
    except Exception as e:
        print(f"⚠️ Wav2Vec2 test skipped: {e}")

    print("\n✅ Audio encoder tests completed!")
