#!/usr/bin/env python3
"""
Configuration class for multimodal emotion recognition
Extended to support text and fusion settings
"""

class Config:
    """Configuration class with multimodal parameters"""

    def __init__(self):
        # Data settings
        self.batch_size = 32
        self.num_epochs = 20

        # Model settings
        self.hidden_dim = 1024
        self.dropout = 0.1
        self.num_classes = 4  # neutral, happy, sad, anger

        # Multimodal settings
        self.modality = "both"  # "audio", "text", or "both"
        self.audio_dim = 768  # emotion2vec feature dimension
        self.text_model_name = "bert-base-uncased"  # BERT model for text
        self.freeze_text_encoder = True  # Keep BERT frozen
        self.text_max_length = 128  # Maximum text sequence length

        # Fusion settings (only used when modality="both")
        self.fusion_type = "cross_attention"  # "cross_attention", "concat", "gated", "adaptive"
        self.fusion_hidden_dim = 512  # Hidden dimension for fusion module
        self.num_attention_heads = 8  # Number of attention heads for cross-attention

        # Training settings
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5

        # Dataset settings
        self.train_dataset = "MSPI"  # or "IEMO" or "MSPP"

        # Evaluation settings
        self.evaluation_mode = "both"  # "loso", "cross_corpus", "both"
        self.val_split = 0.2  # validation split for cross-corpus only mode

        # Curriculum Learning settings
        self.use_curriculum_learning = True
        self.curriculum_epochs = 10  # number of epochs to gradually introduce samples
        self.curriculum_pacing = "linear"  # or "sqrt", "log"
        self.difficulty_method = "euclidean_distance"  # method to calculate difficulty
        self.curriculum_type = "difficulty"  # "difficulty", "class_balance", "inverse_difficulty", "model_confidence", "random", "none", "preset_order"

        # Expected VAD values for difficulty calculation
        self.expected_vad = {
            0: [3.0, 3.0, 3.0],  # neutral
            1: [4.2, 3.8, 3.5],  # happy
            2: [1.8, 2.2, 2.5],  # sad
            3: [2.5, 4.0, 3.2]   # anger
        }

        # WandB settings
        self.wandb_project = "Multimodal_Emotion_Recognition"
        self.experiment_name = "baseline"

        # Class labels
        self.class_names = ["neutral", "happy", "sad", "anger"]

        # Advanced training settings
        self.use_difficulty_scaling = False
        self.use_speaker_disentanglement = False

        # Class weights (optional)
        self.class_weights = {
            "neutral": 1.0,
            "happy": 1.0,
            "sad": 1.0,
            "anger": 1.0
        }

        # Random seed(s) - can be single seed or list of seeds
        self.seeds = [42]  # Default single seed, can be [42, 123, 456] for multi-seed experiments
        self.seed = 42  # Backward compatibility

    def __repr__(self):
        """String representation of config"""
        config_str = "Configuration:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str

    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
