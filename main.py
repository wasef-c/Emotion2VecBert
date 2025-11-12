#!/usr/bin/env python3
"""
Multimodal emotion recognition training script
Supports audio-only, text-only, and multimodal (audio + text) emotion recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import yaml
import random
import math
import torch.optim.lr_scheduler as lr_scheduler

# Add parent directory to path to access original dataset classes
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from collections import defaultdict
import wandb

# Import local modules
from config import Config
from model import create_model
from text_encoder import FrozenBERTEncoder
from functions import *  # All utility functions


def load_config_from_yaml(yaml_path, experiment_id=None):
    """Load configuration from YAML file and create Config object"""
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Check if this is a multi-experiment config
    if "experiments" in yaml_config:
        if experiment_id is None:
            # List available experiments
            print("📋 Available experiments:")
            for i, exp in enumerate(yaml_config["experiments"]):
                print(f"   {i}: {exp.get('name', exp.get('id', f'experiment_{i}'))}")
            raise ValueError(
                "Please specify --experiment <id> when using multi-experiment config"
            )

        # Find the specified experiment
        if isinstance(experiment_id, int):
            if 0 <= experiment_id < len(yaml_config["experiments"]):
                experiment_config = yaml_config["experiments"][experiment_id]
            else:
                raise ValueError(f"Experiment index {experiment_id} out of range")
        else:
            # Find by name or id
            experiment_config = None
            for exp in yaml_config["experiments"]:
                if exp.get("id") == experiment_id or exp.get("name") == experiment_id:
                    experiment_config = exp
                    break
            if experiment_config is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")

        # Use the specific experiment config
        yaml_config = experiment_config
        print(
            f"🧪 Running experiment: {yaml_config.get('name', yaml_config.get('id', experiment_id))}"
        )

    config = Config()

    # Define type conversions for config parameters
    float_params = [
        "learning_rate",
        "weight_decay",
        "dropout",
        "val_split",
        "loss_temperature",
        "focal_gamma",
        "post_curriculum_dropout",
        "lr_scheduler_gamma",
        "lr_scheduler_factor",
        "early_stopping_min_delta",
    ]
    int_params = [
        "batch_size",
        "num_epochs",
        "hidden_dim",
        "num_classes",
        "curriculum_epochs",
        "audio_dim",
        "fusion_hidden_dim",
        "num_attention_heads",
        "text_max_length",
        "lr_scheduler_step_size",
        "lr_scheduler_patience",
        "early_stopping_patience",
    ]
    bool_params = [
        "use_curriculum_learning",
        "use_difficulty_scaling",
        "use_speaker_disentanglement",
        "freeze_text_encoder",
        "freeze_audio_encoder",
        "use_early_stopping",
    ]
    string_params = [
        "wandb_project",
        "experiment_name",
        "curriculum_type",
        "difficulty_method",
        "curriculum_pacing",
        "modality",
        "audio_encoder_type",
        "audio_model_name",
        "audio_pooling",
        "text_model_name",
        "fusion_type",
        "lr_scheduler",
    ]

    # Update config with YAML values with proper type conversion
    for key, value in yaml_config.items():
        if hasattr(config, key):
            # Special handling for seeds - can be single value or list
            if key == "seeds":
                if isinstance(value, list):
                    config.seeds = value
                else:
                    config.seeds = [value]
                config.seed = config.seeds[0]  # Set first seed as default
            # Apply type conversion
            elif key in float_params and value is not None:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    print(f"⚠️  Could not convert {key}={value} to float, using as-is")
            elif key in int_params and value is not None:
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    print(f"⚠️  Could not convert {key}={value} to int, using as-is")
            elif key in bool_params and value is not None:
                if isinstance(value, str):
                    value = value.lower() in ["true", "1", "yes", "on"]
                else:
                    value = bool(value)
            elif key in string_params and value is not None:
                value = str(value)

            if key != "seeds":  # Seeds already handled above
                setattr(config, key, value)
        elif key not in ["id", "name", "description", "category"]:  # Skip metadata
            print(f"⚠️  Unknown config parameter: {key}")

    # Set experiment name from YAML if available
    if "name" in yaml_config:
        config.experiment_name = yaml_config["name"]

    return config


def run_all_experiments_from_yaml(yaml_path):
    """Run all experiments from a multi-experiment YAML file"""
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    if "experiments" not in yaml_config:
        raise ValueError("YAML file doesn't contain multiple experiments")

    print(f"🚀 Running {len(yaml_config['experiments'])} experiments from {yaml_path}")
    results = []

    for i, experiment in enumerate(yaml_config["experiments"]):
        exp_name = experiment.get("name", experiment.get("id", f"experiment_{i}"))
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT {i+1}/{len(yaml_config['experiments'])}: {exp_name}")
        print(f"{'='*60}")

        try:
            config = load_config_from_yaml(yaml_path, i)
            result = run_experiment_with_seeds(config)
            results.append(
                {
                    "experiment_id": i,
                    "name": exp_name,
                    "result": result,
                    "status": "completed",
                }
            )
        except Exception as e:
            import traceback

            full_traceback = traceback.format_exc()
            print(f"❌ Experiment {exp_name} failed: {e}")
            print(f"🔍 Full traceback:")
            print(full_traceback)
            results.append(
                {
                    "experiment_id": i,
                    "name": exp_name,
                    "result": None,
                    "status": "failed",
                    "error": str(e),
                    "traceback": full_traceback,
                }
            )

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = len(results) - completed
    print(f"✅ Completed: {completed}")
    print(f"❌ Failed: {failed}")

    return results


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_memory_usage(config):
    """Apply memory optimizations for Wav2Vec2 training"""
    if hasattr(config, 'audio_encoder_type') and config.audio_encoder_type == "wav2vec2":
        # Enable memory-efficient attention if available
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Enable memory mapping for datasets
        os.environ['HF_DATASETS_OFFLINE'] = '1' if not hasattr(config, 'hf_datasets_offline') else str(config.hf_datasets_offline)
        
        # Suggest gradient accumulation if batch size is too small
        if hasattr(config, 'batch_size') and config.batch_size < 16:
            effective_batch_size = getattr(config, 'effective_batch_size', 64)
            gradient_accumulation_steps = max(1, effective_batch_size // config.batch_size)
            if not hasattr(config, 'gradient_accumulation_steps'):
                config.gradient_accumulation_steps = gradient_accumulation_steps
                print(f"🔧 Using gradient accumulation: {gradient_accumulation_steps} steps for effective batch size {effective_batch_size}")
        
        print(f"🔧 Memory optimizations applied for Wav2Vec2 (batch_size: {config.batch_size})")


def create_lr_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on config

    Args:
        optimizer: PyTorch optimizer
        config: Config object with scheduler settings

    Returns:
        scheduler: Learning rate scheduler or None
    """
    scheduler_type = getattr(config, "lr_scheduler", "cosine").lower()

    if scheduler_type == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max= config.num_epochs
        )
    elif scheduler_type == "step":
        step_size = getattr(config, "lr_scheduler_step_size", 10)
        gamma = getattr(config, "lr_scheduler_gamma", 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "exponential":
        gamma = getattr(config, "lr_scheduler_gamma", 0.95)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == "plateau":
        patience = getattr(config, "lr_scheduler_patience", 5)
        factor = getattr(config, "lr_scheduler_factor", 0.5)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Maximize UAR
            factor=factor,
            patience=patience,
            verbose=True,
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        print(f"⚠️  Unknown scheduler type '{scheduler_type}', using CosineAnnealingLR")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    return scheduler


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""

    def __init__(self, patience=10, min_delta=0.001, mode="max", start_epoch=0):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy/UAR, 'min' for loss
            start_epoch: Epoch to start checking (e.g., after curriculum learning)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.start_epoch = start_epoch
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, epoch):
        """
        Check if training should stop

        Args:
            score: Current validation metric
            epoch: Current epoch number

        Returns:
            bool: True if training should stop
        """
        # Don't check early stopping until after start_epoch
        if epoch < self.start_epoch:
            return False

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"\n⏹️  Early stopping triggered after {self.counter} epochs without improvement"
                )
                print(
                    f"   Best score: {self.best_score:.4f}, Current score: {score:.4f}"
                )
                return True
            else:
                print(
                    f"   Early stopping: {self.counter}/{self.patience} epochs without improvement"
                )
            return False


def run_experiment(config):
    """Run a single experiment with given config"""
    # Set seed for reproducibility
    seed = getattr(config, "seed", 42)
    set_seed(seed)
    print(f"🔢 Random seed set to: {seed}")

    # Apply memory optimizations
    optimize_memory_usage(config)

    # Initialize wandb
    wandb.init(
        project=config.wandb_project, name=config.experiment_name, config=vars(config)
    )

    print(f"🚀 Starting experiment: {config.experiment_name}")
    print(f"📊 Evaluation Mode: {config.evaluation_mode.upper()}")
    print(f"🎭 Modality: {getattr(config, 'modality', 'audio').upper()}")

    # Handle multiple training datasets
    train_datasets_list = (
        config.train_dataset
        if isinstance(config.train_dataset, list)
        else [config.train_dataset]
    )

    # Load datasets
    if len(train_datasets_list) > 1:
        # Multiple training datasets - concatenate them
        print(f"🔗 Training on multiple datasets: {', '.join(train_datasets_list)}")
        individual_datasets = []
        for dataset_name in train_datasets_list:
            individual_datasets.append(
                SimpleEmotionDataset(dataset_name, config=config, Train=True)
            )
        train_dataset = ConcatenatedDataset(individual_datasets)

        # For cross-corpus evaluation, test on all datasets NOT in training
        all_datasets = ["IEMO", "MSPI", "MSPP", "CMUMOSEI", "SAMSEMO"]
        test_dataset_names = [d for d in all_datasets if d not in train_datasets_list]

        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset(name, config=config) for name in test_dataset_names
            ]
            print(
                f"🚀 Training: {'+'.join(train_datasets_list)} -> {test_dataset_names}"
            )
        elif config.evaluation_mode == "loso":
            # LOSO not supported for multi-dataset training
            raise ValueError(
                "LOSO evaluation not supported with multiple training datasets. Use 'cross_corpus' or 'both' mode."
            )
        else:
            # For "both" mode, we'll do cross-corpus only
            test_datasets = [
                SimpleEmotionDataset(name, config=config) for name in test_dataset_names
            ]
            print(
                f"🚀 Training: {'+'.join(train_datasets_list)} -> {test_dataset_names}"
            )
            print(
                f"⚠️  Note: LOSO not available with multiple training datasets, using cross-corpus only"
            )
            config.evaluation_mode = "cross_corpus"  # Override to cross_corpus

    elif config.train_dataset == "MSPI":
        train_dataset = SimpleEmotionDataset("MSPI", config=config, Train=True)
        # For cross-corpus, we want to test on both other datasets
        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset("IEMO", config=config),
                SimpleEmotionDataset("MSPP", config=config),
                SimpleEmotionDataset("CMUMOSEI", config=config),
                SimpleEmotionDataset("SAMSEMO", config=config),
            ]
            print(f"🚀 Training: MSPI -> [IEMO, MSPP, CMUMOSEI, SAMSEMO]")
        else:
            test_dataset = SimpleEmotionDataset("IEMO", config=config)
            print(f"🚀 Training: MSPI -> IEMO")
    elif config.train_dataset == "IEMO":
        train_dataset = SimpleEmotionDataset("IEMO", config=config, Train=True)
        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset("MSPI", config=config),
                SimpleEmotionDataset("MSPP", config=config),
                SimpleEmotionDataset("CMUMOSEI", config=config),
                SimpleEmotionDataset("SAMSEMO", config=config),
            ]
            print(f"🚀 Training: IEMO -> [MSPI, MSPP, CMUMOSEI, SAMSEMO]")
        else:
            test_dataset = SimpleEmotionDataset("MSPI", config=config)
            print(f"🚀 Training: IEMO -> MSPI")
    elif config.train_dataset == "MSPP":
        train_dataset = SimpleEmotionDataset("MSPP", config=config, Train=True)
        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset("IEMO", config=config),
                SimpleEmotionDataset("MSPI", config=config),
                SimpleEmotionDataset("CMUMOSEI", config=config),
                SimpleEmotionDataset("SAMSEMO", config=config),
            ]
            print(f"🚀 Training: MSPP -> [IEMO, MSPI, CMUMOSEI, SAMSEMO]")
        else:
            test_dataset = SimpleEmotionDataset("IEMO", config=config)
            print(f"🚀 Training: MSPP -> IEMO")
    else:
        raise ValueError(f"Unknown train dataset: {config.train_dataset}")

    # Run evaluation based on mode
    if config.evaluation_mode == "loso":
        results = run_loso_evaluation(config, train_dataset, test_dataset)
    elif config.evaluation_mode == "cross_corpus":
        results = run_cross_corpus_evaluation(config, train_dataset, test_datasets)
    elif config.evaluation_mode == "both":
        print("\n" + "=" * 60)
        print("RUNNING LOSO EVALUATION")
        print("=" * 60)
        loso_results = run_loso_evaluation(config, train_dataset, test_dataset)

        print("\n" + "=" * 60)
        print("RUNNING CROSS-CORPUS EVALUATION")
        print("=" * 60)
        cross_corpus_results = run_cross_corpus_evaluation(
            config, train_dataset, [test_dataset]
        )

        results = {"loso": loso_results, "cross_corpus": cross_corpus_results}
    else:
        raise ValueError(f"Unknown evaluation mode: {config.evaluation_mode}")

    # Print final results based on evaluation mode
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {config.evaluation_mode.upper()}")
    print(f"{'='*60}")

    if config.evaluation_mode == "loso":
        print(
            f"LOSO Accuracy: {results['loso_accuracy_mean']:.4f} ± {results['loso_accuracy_std']:.4f}"
        )
        print(
            f"LOSO UAR: {results['loso_uar_mean']:.4f} ± {results['loso_uar_std']:.4f}"
        )

        # Log final metrics
        wandb.log(
            {
                "final/loso_acc_mean": results["loso_accuracy_mean"],
                "final/loso_acc_std": results["loso_accuracy_std"],
                "final/loso_uar_mean": results["loso_uar_mean"],
                "final/loso_uar_std": results["loso_uar_std"],
            }
        )

    elif config.evaluation_mode == "cross_corpus":
        print(f"Validation Accuracy: {results['validation']['accuracy']:.4f}")
        print(f"Validation UAR: {results['validation']['uar']:.4f}")

        for test_result in results["test_results"]:
            dataset_name = test_result["dataset"]
            acc = test_result["results"]["accuracy"]
            uar = test_result["results"]["uar"]
            print(f"{dataset_name} Test Accuracy: {acc:.4f}")
            print(f"{dataset_name} Test UAR: {uar:.4f}")

        # Log final metrics
        final_log = {
            "final/validation_acc": results["validation"]["accuracy"],
            "final/validation_uar": results["validation"]["uar"],
        }
        for test_result in results["test_results"]:
            dataset_name = test_result["dataset"].lower()
            final_log[f"final/{dataset_name}_acc"] = test_result["results"]["accuracy"]
            final_log[f"final/{dataset_name}_uar"] = test_result["results"]["uar"]
        wandb.log(final_log)

    elif config.evaluation_mode == "both":
        print("LOSO Results:")
        loso = results["loso"]
        print(
            f"  LOSO Accuracy: {loso['loso_accuracy_mean']:.4f} ± {loso['loso_accuracy_std']:.4f}"
        )
        print(f"  LOSO UAR: {loso['loso_uar_mean']:.4f} ± {loso['loso_uar_std']:.4f}")

        print("\nCross-Corpus Only Results:")
        cross = results["cross_corpus"]
        print(f"  Validation Accuracy: {cross['validation']['accuracy']:.4f}")
        print(f"  Validation UAR: {cross['validation']['uar']:.4f}")
        for test_result in cross["test_results"]:
            dataset_name = test_result["dataset"]
            acc = test_result["results"]["accuracy"]
            uar = test_result["results"]["uar"]
            print(f"  {dataset_name} Test Accuracy: {acc:.4f}")
            print(f"  {dataset_name} Test UAR: {uar:.4f}")

        # Log both sets of metrics
        final_log = {
            "final/loso_acc_mean": loso["loso_accuracy_mean"],
            "final/loso_uar_mean": loso["loso_uar_mean"],
            "final/validation_acc": cross["validation"]["accuracy"],
            "final/validation_uar": cross["validation"]["uar"],
        }
        for test_result in cross["test_results"]:
            dataset_name = test_result["dataset"].lower()
            final_log[f"final/{dataset_name}_cross_only_acc"] = test_result["results"][
                "accuracy"
            ]
            final_log[f"final/{dataset_name}_cross_only_uar"] = test_result["results"][
                "uar"
            ]
        wandb.log(final_log)

    wandb.finish()
    return results


class ConcatenatedDataset(Dataset):
    """
    Concatenates multiple SimpleEmotionDataset instances
    Useful for training on multiple datasets simultaneously
    """

    def __init__(self, datasets):
        """
        Args:
            datasets: List of SimpleEmotionDataset instances
        """
        self.datasets = datasets
        self.dataset_name = "+".join([d.dataset_name for d in datasets])

        # Concatenate all data
        self.data = []
        for dataset in datasets:
            self.data.extend(dataset.data)

        print(f"✅ Concatenated {len(datasets)} datasets: {self.dataset_name}")
        print(f"   Total samples: {len(self.data)}")

        # Print per-dataset statistics
        start_idx = 0
        for dataset in datasets:
            end_idx = start_idx + len(dataset.data)
            print(
                f"   {dataset.dataset_name}: {len(dataset.data)} samples (indices {start_idx}-{end_idx-1})"
            )
            start_idx = end_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        result = {
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "session": item["session"],
            "dataset": item["dataset"],
            "difficulty": item["difficulty"],
            "curriculum_order": item["curriculum_order"],
            "sequence_length": item["sequence_length"],
        }

        # Add modality-specific data
        modality = self.datasets[0].modality  # All datasets share same modality
        if modality in ["audio", "both"]:
            result["features"] = item["features"]

        if modality in ["text", "both"]:
            result["transcript"] = item["transcript"]

        return result


class SimpleEmotionDataset(Dataset):
    """
    Dataset class for multimodal emotion recognition
    Supports audio-only, text-only, and multimodal (audio + text) modes
    """

    def __init__(self, dataset_name, split="train", config=None, Train=False):
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.modality = getattr(config, "modality", "audio")
        self.audio_encoder_type = getattr(config, "audio_encoder_type", "preextracted")

        # Map dataset names to their merged dataset names (with audio + features)
        # These datasets have both pre-extracted Emotion2Vec features AND raw audio
        merged_dataset_map = {
            "IEMO": "cairocode/IEMO_Audio_Text_Merged",
            "MSPI": "cairocode/MSPI_Audio_Text_Merged",
            "MSPP": "cairocode/MSPP_WAV_Filtered_Ordered_v2",
            "CMUMOSEI": "cairocode/cmu_mosei_wav",
            "SAMSEMO": "cairocode/samsemo-audio",
        }
        # Load HuggingFace merged dataset (has features, audio, and metadata)
        if dataset_name not in merged_dataset_map:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        dataset_path = merged_dataset_map[dataset_name]
        self.hf_dataset = load_dataset(
            dataset_path, split=split, trust_remote_code=True
        )

        print(f"📥 Loaded merged dataset: {dataset_path}")
        print(f"   Columns: {self.hf_dataset.column_names}")

        # Note: Merged datasets contain both pre-extracted features AND raw audio
        # - Use audio_encoder_type="preextracted" to use Emotion2Vec features
        # - Use audio_encoder_type="wav2vec2"/"hubert"/etc to use raw audio
        if self.modality in ["audio", "both"] and self.audio_encoder_type != "preextracted":
            if "audio" not in self.hf_dataset.column_names:
                print(f"⚠️ Warning: 'audio' column not found in {dataset_name}")
                print(f"   Available columns: {self.hf_dataset.column_names}")
                print(f"   Falling back to pre-extracted features")
                self.audio_encoder_type = "preextracted"
            else:
                print(f"🎵 Using raw audio from 'audio' column (encoder: {self.audio_encoder_type})")

        # Ultra-lazy loading: don't even iterate through dataset
        # Just store length and let __getitem__ handle everything
        print(f"🔍 DEBUG: Dataset length: {len(self.hf_dataset)}")
        
        # Create dummy metadata for compatibility
        self.metadata = [{"hf_index": i} for i in range(len(self.hf_dataset))]
        
        # Skip the full iteration that was causing memory issues
        if False:  # Disable the problematic loop
            for idx, item in enumerate(self.hf_dataset):
            # Calculate basic metadata without loading heavy data
            
            # Get speaker and session information
            if Train == True:
                # Get speaker ID and calculate session directly
                if self.dataset_name == "IEMO":
                    speaker_id = item["speaker_id"]
                    session = (speaker_id - 1) // 2 + 1
                elif self.dataset_name == "MSPI":
                    speaker_id = item["speakerID"]
                    session = (speaker_id - 947) // 2 + 1
                elif self.dataset_name == "MSPP":
                    speaker_id = item["SpkrID"]
                    session = (speaker_id - 1) // 500 + 1
                elif self.dataset_name == "CMUMOSEI":
                    # CMU-MOSEI has video_id field
                    speaker_id = hash(item.get("video_id", "unknown")) % 10000
                    session = (speaker_id - 1) // 100 + 1
                elif self.dataset_name == "SAMSEMO":
                    # SAMSEMO may have speaker_id or file_name
                    speaker_id = item.get(
                        "speaker_id", hash(item.get("file_name", "unknown")) % 10000
                    )
                    session = (speaker_id - 1) // 100 + 1
                else:
                    # Fallback for other datasets
                    try:
                        speaker_id = item["speaker_id"]
                    except:
                        speaker_id = item.get("speakerID", item.get("SpkrID", 1))
                    session = (speaker_id - 1) // 2 + 1
            else:
                speaker_id = -1  # Use -1 instead of None for test datasets
                session = -1  # Use -1 instead of None for test datasets

            label = item["label"]

            # Get curriculum order from dataset
            curriculum_order = item.get(
                "curriculum_order", 0.5
            )  # Default to middle if missing

            # Get VAD values for difficulty calculation
            valence = item.get("valence", item.get("EmoVal", None))
            arousal = item.get("arousal", item.get("EmoAct", None))
            domination = item.get(
                "domination", item.get("consensus_dominance", item.get("EmoDom", None))
            )

            # Replace NaN or None with 3
            def fix_vad(value):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return 3
                return value

            valence = fix_vad(valence)
            arousal = fix_vad(arousal)
            domination = fix_vad(domination)
            item_with_vad = {
                "label": label,
                "valence": valence,
                "arousal": arousal,
                "domination": domination,
            }
            difficulty = calculate_difficulty(
                item_with_vad,
                config.expected_vad,
                config.difficulty_method,
                dataset=dataset_name,
            )

            # Store only metadata - audio/text will be loaded on-demand
            self.metadata.append(
                {
                    "hf_index": idx,  # Index into original HF dataset
                    "label": label,
                    "speaker_id": speaker_id,
                    "session": session,
                    "dataset": dataset_name,
                    "difficulty": difficulty,
                    "curriculum_order": curriculum_order,
                    "sequence_length": 1,  # Will be calculated on-demand
                    "valence": valence,
                    "arousal": arousal,
                    "domination": domination,
                }
            )

        # Keep reference to original data for compatibility with existing code
        self.data = self.metadata

        print(f"✅ Loaded {len(self.data)} samples from {dataset_name}")
        print(f"   Modality: {self.modality}")
        print(f"🔍 DEBUG: Dataset initialization complete")

        # Print session distribution for debugging
        session_counts = defaultdict(int)
        for item in self.data:
            session_counts[item["session"]] += 1

        print(f"📊 {dataset_name} Sessions:")
        for session_id in sorted(session_counts.keys()):
            print(f"   Session {session_id}: {session_counts[session_id]} samples")
        print(f"🔍 DEBUG: Session counting complete")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hf_item = self.hf_dataset[idx]
        
        # Calculate metadata on-demand
        label = hf_item["label"]
        
        # Quick speaker/session calculation
        if self.dataset_name == "MSPP":
            speaker_id = hf_item.get("SpkrID", 1)
            session = (speaker_id - 1) // 500 + 1
        else:
            speaker_id = 1  # Simplified for now
            session = 1
            
        # Calculate difficulty properly
        valence = hf_item.get("valence", hf_item.get("EmoVal", 3.0))
        arousal = hf_item.get("arousal", hf_item.get("EmoAct", 3.0))  
        domination = hf_item.get("domination", hf_item.get("EmoDom", 3.0))
        
        # Fix NaN values
        def fix_vad(value):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return 3.0
            return value
            
        valence = fix_vad(valence)
        arousal = fix_vad(arousal)
        domination = fix_vad(domination)
        
        item_with_vad = {
            "label": label,
            "valence": valence,
            "arousal": arousal,
            "domination": domination,
        }
        difficulty = calculate_difficulty(
            item_with_vad,
            self.config.expected_vad,
            self.config.difficulty_method,
            dataset=self.dataset_name,
        )
        curriculum_order = hf_item.get("curriculum_order", 0.5)
        
        result = {
            "label": torch.tensor(label, dtype=torch.long),
            "speaker_id": speaker_id,
            "session": session,
            "dataset": self.dataset_name,
            "difficulty": difficulty,
            "curriculum_order": curriculum_order,
            "sequence_length": 1,
        }

        # Load audio features on-demand
        if self.modality in ["audio", "both"]:
            if self.audio_encoder_type == "preextracted":
                # Use pre-extracted Emotion2Vec features
                features = torch.tensor(
                    hf_item["emotion2vec_features"][0]["feats"], dtype=torch.float32
                )
            else:
                # Use raw audio for wav2vec2/hubert/emotion2vec encoders
                if "audio" in hf_item and hf_item["audio"] is not None:
                    features = hf_item["audio"]
                else:
                    # Fallback to dummy features if audio missing
                    features = torch.zeros(768)
            
            result["features"] = features

        # Load text on-demand
        if self.modality in ["text", "both"]:
            # Try to get the 'transcript'
            transcript = hf_item.get("transcript")
            
            # If 'transcript' is missing or None, try 'text' as fallback
            if transcript is None or transcript == "":
                transcript = hf_item.get("text")
            
            # Handle cases where both are missing/empty
            if transcript is None or transcript == "":
                transcript = "[EMPTY]"
                
            result["transcript"] = transcript

        return result


def train_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
    text_encoder=None,
    use_difficulty_scaling=False,
):
    """
    Train model for one epoch
    Supports audio-only, text-only, and multimodal training
    """
    model.train()

    # Keep text encoder frozen if using multimodal/text mode
    if text_encoder is not None:
        text_encoder.eval()

    # Verify dropout is active
    if hasattr(model, "dropout"):
        print(
            f"      Training with dropout rate: {model.dropout.p:.3f}, training mode: {model.training}"
        )

    total_loss = 0
    predictions = []
    labels = []

    # Track batch statistics for debugging
    batch_num = 0
    print_every_n_batches = max(1, len(data_loader) // 3)  # Print 3 times per epoch

    modality = getattr(config, "modality", "audio")
    text_max_length = getattr(config, "text_max_length", 128)

    for batch in data_loader:
        batch_labels = batch["label"].to(device)
        difficulties = batch["difficulty"].to(device)

        optimizer.zero_grad()

        # Forward pass based on modality
        if modality == "audio":
            # Audio-only mode
            features = batch["features"]
            # Handle raw audio data for wav2vec2/hubert vs pre-extracted features
            if isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
                # Raw audio data - needs special handling
                # For now, convert to tensor format that AudioEncoder expects
                audio_arrays = [f["array"] for f in features if "array" in f]
                if audio_arrays:
                    # Convert to tensor and move to device
                    # Note: This assumes all audio has same length (may need padding)
                    features = torch.tensor(audio_arrays, dtype=torch.float32).to(device)
                else:
                    # Fallback if no valid audio
                    features = torch.zeros(len(features), 768).to(device)
            else:
                # Pre-extracted features - normal tensor
                features = features.to(device)
            logits = model(features)

        elif modality == "text":
            # Text-only mode
            transcripts = batch["transcript"]

            # Tokenize text
            if hasattr(model, "text_encoder") and model.text_encoder is not None:
                # Model has integrated text encoder
                input_ids, attention_mask = model.text_encoder.tokenize_batch(
                    transcripts, max_length=text_max_length, device=device
                )
                logits = model(
                    text_input_ids=input_ids, text_attention_mask=attention_mask
                )
            elif text_encoder is not None:
                # External text encoder
                input_ids, attention_mask = text_encoder.tokenize_batch(
                    transcripts, max_length=text_max_length, device=device
                )
                logits = model(
                    text_input_ids=input_ids, text_attention_mask=attention_mask
                )
            else:
                raise ValueError("Text encoder not available for text-only mode")

        elif modality == "both":
            # Multimodal mode
            features = batch["features"]
            # Handle raw audio data for wav2vec2/hubert vs pre-extracted features
            if isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
                # Raw audio data - needs special handling
                audio_arrays = [f["array"] for f in features if "array" in f]
                if audio_arrays:
                    features = torch.tensor(audio_arrays, dtype=torch.float32).to(device)
                else:
                    features = torch.zeros(len(features), 768).to(device)
            else:
                # Pre-extracted features - normal tensor
                features = features.to(device)
            transcripts = batch["transcript"]

            # Tokenize text
            if hasattr(model, "text_encoder") and model.text_encoder is not None:
                # Model has integrated text encoder
                input_ids, attention_mask = model.text_encoder.tokenize_batch(
                    transcripts, max_length=text_max_length, device=device
                )
                logits = model(
                    audio_features=features,
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                )
            elif text_encoder is not None:
                # External text encoder
                input_ids, attention_mask = text_encoder.tokenize_batch(
                    transcripts, max_length=text_max_length, device=device
                )
                logits = model(
                    audio_features=features,
                    text_input_ids=input_ids,
                    text_attention_mask=attention_mask,
                )
            else:
                raise ValueError("Text encoder not available for multimodal mode")
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Calculate loss
        loss_per_sample = criterion(logits, batch_labels)  # reduction='none'
        loss = loss_per_sample.mean()

        # Track predictions for metrics
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(preds)
        labels.extend(batch_labels.cpu().numpy())

        # # Debug logging for first few batches
        # if batch_num % print_every_n_batches == 0:
        #     with torch.no_grad():
        #         probs = torch.softmax(logits, dim=-1)
        #         print(f"\n      Batch {batch_num}:")
        #         print(f"        Logits stats: min={logits.min().item():.3f}, max={logits.max().item():.3f}, mean={logits.mean().item():.3f}")
        #         print(f"        Probs stats: min={probs.min().item():.3f}, max={probs.max().item():.3f}")
        #         print(f"        Predicted labels: {preds[:8]}")  # First 8 predictions
        #         print(f"        True labels:      {batch_labels[:8].cpu().numpy()}")
        #         print(f"        Label distribution in batch: {np.bincount(batch_labels.cpu().numpy(), minlength=4)}")
        #         print(f"        Prediction distribution: {np.bincount(preds, minlength=4)}")
        #         print(f"        Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]

        # Clear GPU cache periodically for memory efficiency
        if batch_num % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_dict = {
            "loss": loss,
            "batch difficulty": difficulties.mean(),
            "lr": current_lr,
        }

        wandb.log(log_dict)

        batch_num += 1

    # Safety check for empty dataloader
    if len(data_loader) == 0:
        print("   Warning: Empty dataloader, returning zero loss and metrics")
        return 0.0, {"accuracy": 0.0, "uar": 0.0, "f1_weighted": 0.0}

    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(predictions, labels)

    # # Print epoch summary
    # print(f"\n      Epoch Summary:")
    # print(f"        Overall prediction distribution: {np.bincount(predictions, minlength=4)}")
    # print(f"        Overall label distribution:      {np.bincount(labels, minlength=4)}")
    # print(f"        Accuracy: {metrics['accuracy']:.4f}, UAR: {metrics['uar']:.4f}")

    # Step the scheduler if it's not ReduceLROnPlateau (which needs validation metrics)
    if scheduler is not None and not isinstance(
        scheduler, lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()

    return avg_loss, metrics


def evaluate_model_multimodal(
    model,
    data_loader,
    criterion,
    device,
    config,
    text_encoder=None,
    return_difficulties=True,
    create_plots=True,
    plot_title="",
):
    """
    Evaluate model on a dataset with support for multimodal inputs
    """
    model.eval()

    # Keep text encoder frozen
    if text_encoder is not None:
        text_encoder.eval()

    total_loss = 0
    predictions = []
    labels = []
    difficulties = []

    modality = getattr(config, "modality", "audio")
    text_max_length = getattr(config, "text_max_length", 128)

    with torch.no_grad():
        for batch in data_loader:
            batch_labels = batch["label"].to(device)

            # Forward pass based on modality
            if modality == "audio":
                # Audio-only mode
                features = batch["features"]
                # Handle raw audio data for wav2vec2/hubert vs pre-extracted features
                if isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
                    # Raw audio data - needs special handling
                    audio_arrays = [f["array"] for f in features if "array" in f]
                    if audio_arrays:
                        features = torch.tensor(audio_arrays, dtype=torch.float32).to(device)
                    else:
                        features = torch.zeros(len(features), 768).to(device)
                else:
                    # Pre-extracted features - normal tensor
                    features = features.to(device)
                logits = model(features)

            elif modality == "text":
                # Text-only mode
                transcripts = batch["transcript"]

                # Tokenize text
                if hasattr(model, "text_encoder") and model.text_encoder is not None:
                    input_ids, attention_mask = model.text_encoder.tokenize_batch(
                        transcripts, max_length=text_max_length, device=device
                    )
                    logits = model(
                        text_input_ids=input_ids, text_attention_mask=attention_mask
                    )
                elif text_encoder is not None:
                    input_ids, attention_mask = text_encoder.tokenize_batch(
                        transcripts, max_length=text_max_length, device=device
                    )
                    logits = model(
                        text_input_ids=input_ids, text_attention_mask=attention_mask
                    )
                else:
                    raise ValueError("Text encoder not available for text-only mode")

            elif modality == "both":
                # Multimodal mode
                features = batch["features"]
                # Handle raw audio data for wav2vec2/hubert vs pre-extracted features
                if isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict):
                    # Raw audio data - needs special handling
                    audio_arrays = [f["array"] for f in features if "array" in f]
                    if audio_arrays:
                        features = torch.tensor(audio_arrays, dtype=torch.float32).to(device)
                    else:
                        features = torch.zeros(len(features), 768).to(device)
                else:
                    # Pre-extracted features - normal tensor
                    features = features.to(device)
                transcripts = batch["transcript"]

                # Tokenize text
                if hasattr(model, "text_encoder") and model.text_encoder is not None:
                    input_ids, attention_mask = model.text_encoder.tokenize_batch(
                        transcripts, max_length=text_max_length, device=device
                    )
                    logits = model(
                        audio_features=features,
                        text_input_ids=input_ids,
                        text_attention_mask=attention_mask,
                    )
                elif text_encoder is not None:
                    input_ids, attention_mask = text_encoder.tokenize_batch(
                        transcripts, max_length=text_max_length, device=device
                    )
                    logits = model(
                        audio_features=features,
                        text_input_ids=input_ids,
                        text_attention_mask=attention_mask,
                    )
                else:
                    raise ValueError("Text encoder not available for multimodal mode")

            loss = criterion(logits, batch_labels)
            total_loss += loss.mean().item()

            # Get predictions
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            labels.extend(batch_labels.cpu().numpy())

            # Collect difficulties if requested
            if return_difficulties:
                batch_difficulties = batch.get("difficulty", [0.5] * len(batch_labels))
                if torch.is_tensor(batch_difficulties):
                    batch_difficulties = batch_difficulties.cpu().numpy()
                difficulties.extend(batch_difficulties)

    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(predictions, labels)

    results = {
        "loss": avg_loss,
        "predictions": predictions,
        "labels": labels,
        "difficulties": difficulties if return_difficulties else None,
        **metrics,
    }

    # Create plots if requested
    if create_plots and plot_title:
        # Confusion matrix
        confusion_matrix_plot = create_confusion_matrix(predictions, labels, plot_title)
        results["confusion_matrix"] = confusion_matrix_plot

        # Difficulty vs accuracy plot (only if we have difficulties)
        if return_difficulties and len(difficulties) > 0:
            difficulty_plot, difficulty_analysis = create_difficulty_accuracy_plot(
                predictions, labels, difficulties, plot_title
            )
            results["difficulty_plot"] = difficulty_plot
            results["difficulty_analysis"] = difficulty_analysis

    return results


def calculate_model_confidences_multimodal(
    model, dataset, indices, device, config, text_encoder=None
):
    """Calculate model confidence scores for samples in multimodal setting"""
    model.eval()
    if text_encoder is not None:
        text_encoder.eval()

    confidences = []
    modality = getattr(config, "modality", "audio")
    text_max_length = getattr(config, "text_max_length", 128)

    with torch.no_grad():
        for i in indices:
            sample = dataset[i]

            # Forward pass based on modality
            if modality == "audio":
                features = sample["features"]
                # Handle raw audio data for wav2vec2/hubert vs pre-extracted features
                if isinstance(features, dict) and "array" in features:
                    # Raw audio data - convert to tensor
                    features = torch.tensor([features["array"]], dtype=torch.float32).to(device)
                else:
                    # Pre-extracted features - normal tensor
                    features = features.unsqueeze(0).to(device)
                logits = model(features)

            elif modality == "text":
                transcript = sample["transcript"]
                if hasattr(model, "text_encoder") and model.text_encoder is not None:
                    input_ids, attention_mask = model.text_encoder.tokenize_batch(
                        [transcript], max_length=text_max_length, device=device
                    )
                    logits = model(
                        text_input_ids=input_ids, text_attention_mask=attention_mask
                    )
                elif text_encoder is not None:
                    input_ids, attention_mask = text_encoder.tokenize_batch(
                        [transcript], max_length=text_max_length, device=device
                    )
                    logits = model(
                        text_input_ids=input_ids, text_attention_mask=attention_mask
                    )

            elif modality == "both":
                features = sample["features"].unsqueeze(0).to(device)
                transcript = sample["transcript"]
                if hasattr(model, "text_encoder") and model.text_encoder is not None:
                    input_ids, attention_mask = model.text_encoder.tokenize_batch(
                        [transcript], max_length=text_max_length, device=device
                    )
                    logits = model(
                        audio_features=features,
                        text_input_ids=input_ids,
                        text_attention_mask=attention_mask,
                    )
                elif text_encoder is not None:
                    input_ids, attention_mask = text_encoder.tokenize_batch(
                        [transcript], max_length=text_max_length, device=device
                    )
                    logits = model(
                        audio_features=features,
                        text_input_ids=input_ids,
                        text_attention_mask=attention_mask,
                    )

            probs = torch.softmax(logits, dim=-1)
            confidence = torch.max(probs).item()
            confidences.append(confidence)

    return confidences


def run_loso_evaluation(config, train_dataset, test_dataset):
    """Run Leave-One-Session-Out evaluation with curriculum learning and multimodal support"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    print(
        f"📚 Curriculum Learning: {'Enabled' if config.use_curriculum_learning else 'Disabled'}"
    )

    # Get session splits
    train_sessions = get_session_splits(train_dataset, train_dataset.dataset_name)

    session_results = []

    for test_session in sorted(train_sessions.keys()):
        print(f"\n📊 LOSO Session {test_session} ({train_dataset.dataset_name})")

        # Create train/test splits
        train_indices = []
        for session_id, indices in train_sessions.items():
            if session_id != test_session:
                train_indices.extend(indices)

        test_indices = train_sessions[test_session]

        # Get difficulties for curriculum learning
        train_difficulties = [
            train_dataset.data[i]["difficulty"] for i in train_indices
        ]

        # Create base datasets
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(train_dataset, test_indices)
        cross_corpus_loader = create_data_loader(
            test_dataset,
            config.batch_size,
            shuffle=False,
            use_speaker_disentanglement=False,
        )
        loso_loader = create_data_loader(
            test_subset,
            config.batch_size,
            shuffle=False,
            use_speaker_disentanglement=False,
        )

        # Get actual feature dimension from dataset
        sample = train_dataset[0]
        modality = getattr(config, "modality", "audio")

        if modality in ["audio", "both"]:
            sample_features = sample["features"]
            if len(sample_features.shape) == 2:
                actual_input_dim = sample_features.shape[-1]
            else:
                actual_input_dim = sample_features.shape[0]
        else:
            # Text-only mode doesn't need audio dim
            actual_input_dim = 768  # Default BERT dimension

        print(
            f"🔍 Using input_dim: {actual_input_dim} (detected from {train_dataset.dataset_name})"
        )

        # Update config with actual audio dimension
        config.audio_dim = actual_input_dim

        # Initialize model using factory function
        print(f"🔍 DEBUG: About to create model...")
        model = create_model(config)
        print(f"🔍 DEBUG: Model created, moving to device...")
        model = model.to(device)
        print(f"🔍 DEBUG: Model moved to device successfully")

        # Initialize text encoder if needed (external, for backward compatibility)
        text_encoder = None
        if modality in ["text", "both"] and not hasattr(model, "text_encoder"):
            text_encoder = FrozenBERTEncoder(
                model_name=getattr(config, "text_model_name", "bert-base-uncased")
            ).to(device)

        focal_gamma = getattr(config, "focal_gamma", 2.0)
        loss_temperature = getattr(config, "loss_temperature", 2000)
        criterion = FocalLossAutoWeights(
            num_classes=4,
            gamma=focal_gamma,
            reduction="none",
            device=device,
            temperature=loss_temperature,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = create_lr_scheduler(optimizer, config)

        # Get curriculum pacing function
        pacing_function = get_curriculum_pacing_function(config.curriculum_pacing)

        # Training loop with curriculum learning
        best_loso_acc = 0
        model_confidences = None

        for epoch in range(config.num_epochs):

            # Increase dropout after curriculum learning ends
            if config.use_curriculum_learning and epoch == config.curriculum_epochs:
                post_dropout = getattr(config, "post_curriculum_dropout", 0.6)
                print(
                    f"   Epoch {epoch+1}: Increasing dropout from {config.dropout} to {post_dropout}"
                )
                model.set_dropout_rate(post_dropout)

            # Create curriculum subset if enabled
            if config.use_curriculum_learning and epoch < config.curriculum_epochs:
                curriculum_indices = create_curriculum_subset(
                    [train_dataset[i] for i in train_indices],
                    train_difficulties,
                    epoch,
                    config.curriculum_epochs,
                    pacing_function,
                    use_preset=False,
                    curriculum_type=config.curriculum_type,
                    model_confidences=model_confidences,
                )
                curriculum_train_indices = [
                    train_indices[i] for i in curriculum_indices
                ]
                curriculum_subset = Subset(train_dataset, curriculum_train_indices)
                train_loader = create_data_loader(
                    curriculum_subset,
                    config.batch_size,
                    shuffle=True,
                    use_speaker_disentanglement=config.use_speaker_disentanglement,
                )

                fraction = pacing_function(epoch, config.curriculum_epochs)
                print(
                    f"   Epoch {epoch+1}: Using {len(curriculum_indices)}/{len(train_indices)} samples ({fraction:.2f})"
                )
            else:
                # Use all training data
                train_loader = create_data_loader(
                    train_subset,
                    config.batch_size,
                    shuffle=True,
                    use_speaker_disentanglement=config.use_speaker_disentanglement,
                )
                if epoch == config.curriculum_epochs:
                    print(
                        f"   Epoch {epoch+1}: Curriculum complete, using all training data"
                    )

            train_loss, train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                config,
                text_encoder=text_encoder,
                use_difficulty_scaling=config.use_difficulty_scaling,
            )

            loso_results = evaluate_model_multimodal(
                model,
                loso_loader,
                criterion,
                device,
                config,
                text_encoder=text_encoder,
                create_plots=False,
            )

            # Calculate model confidences for next epoch
            if (
                config.curriculum_type == "model_confidence"
                and config.use_curriculum_learning
                and epoch < config.num_epochs - 1
            ):
                print(f"   Calculating model confidences for epoch {epoch + 2}...")
                model_confidences = calculate_model_confidences_multimodal(
                    model, train_dataset, train_indices, device, config, text_encoder
                )

            if loso_results["uar"] > best_loso_acc:
                best_loso_acc = loso_results["accuracy"]
                best_model_state = model.state_dict().copy()

            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loso_results["uar"])  # Needs validation metric
                else:
                    scheduler.step()  # Standard schedulers

        # Load best model and evaluate with plots
        model.load_state_dict(best_model_state)
        loso_results = evaluate_model_multimodal(
            model,
            loso_loader,
            criterion,
            device,
            config,
            text_encoder=text_encoder,
            create_plots=True,
            plot_title=f"LOSO-Session-{test_session}-{train_dataset.dataset_name}",
        )

        print(
            f"   LOSO: Acc={loso_results['accuracy']:.4f}, UAR={loso_results['uar']:.4f}"
        )

        session_results.append(
            {
                "session": test_session,
                "loso": loso_results,
            }
        )

        # Log to wandb
        if wandb.run:
            log_dict = {
                f"session_{test_session}/loso_acc": loso_results["accuracy"],
                f"session_{test_session}/loso_uar": loso_results["uar"],
            }

            # Add plots if available
            if "confusion_matrix" in loso_results:
                log_dict[f"session_{test_session}/confusion_matrix"] = loso_results[
                    "confusion_matrix"
                ]

            if "difficulty_plot" in loso_results:
                log_dict[f"session_{test_session}/difficulty_plot"] = loso_results[
                    "difficulty_plot"
                ]

                # Add difficulty analysis metrics
                if "difficulty_analysis" in loso_results:
                    analysis = loso_results["difficulty_analysis"]
                    log_dict[f"session_{test_session}/difficulty_correlation"] = (
                        analysis["difficulty_accuracy_correlation"]
                    )
                    log_dict[f"session_{test_session}/correlation_p_value"] = analysis[
                        "correlation_p_value"
                    ]

            wandb.log(log_dict)

    # Calculate averages
    loso_accs = [r["loso"]["accuracy"] for r in session_results]
    loso_uars = [r["loso"]["uar"] for r in session_results]

    results = {
        "loso_accuracy_mean": np.mean(loso_accs),
        "loso_accuracy_std": np.std(loso_accs),
        "loso_uar_mean": np.mean(loso_uars),
        "loso_uar_std": np.std(loso_uars),
        "session_results": session_results,
    }

    return results


def run_cross_corpus_evaluation(config, train_dataset, test_datasets):
    """Run cross-corpus evaluation with train/val split and multimodal support"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    print(
        f"📚 Curriculum Learning: {'Enabled' if config.use_curriculum_learning else 'Disabled'}"
    )
    print(
        f"📊 Cross-Corpus Only Mode: Train={1-config.val_split:.0%}, Val={config.val_split:.0%}"
    )

    # Create train/validation split
    total_samples = len(train_dataset)
    indices = list(range(total_samples))

    np.random.shuffle(indices)

    val_size = int(total_samples * config.val_split)
    train_indices = indices[val_size:]
    if config.curriculum_type == "preset_order":
        # Sort indices by curriculum_order
        train_indices.sort(key=lambda i: train_dataset[i]["curriculum_order"])
    val_indices = indices[:val_size]

    print(f"📈 Training samples: {len(train_indices)}")
    print(f"📋 Validation samples: {len(val_indices)}")

    # Get difficulties for curriculum learning
    train_difficulties = [train_dataset.data[i]["difficulty"] for i in train_indices]

    # Create datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    # Create test loaders for cross-corpus datasets
    test_loaders = []
    test_names = []
    if isinstance(test_datasets, list):
        for test_dataset in test_datasets:
            test_loader = DataLoader(
                test_dataset, batch_size=config.batch_size, shuffle=False
            )
            test_loaders.append(test_loader)
            test_names.append(test_dataset.dataset_name)
    else:
        test_loader = DataLoader(
            test_datasets, batch_size=config.batch_size, shuffle=False
        )
        test_loaders = [test_loader]
        test_names = [test_datasets.dataset_name]

    print(f"🎯 Test datasets: {', '.join(test_names)}")

    # Get actual feature dimension from dataset
    sample = train_dataset[0]
    modality = getattr(config, "modality", "audio")

    if modality in ["audio", "both"]:
        sample_features = sample["features"]
        if len(sample_features.shape) == 2:
            actual_input_dim = sample_features.shape[-1]
        else:
            actual_input_dim = sample_features.shape[0]
    else:
        # Text-only mode doesn't need audio dim
        actual_input_dim = 768  # Default BERT dimension

    print(
        f"🔍 Using input_dim: {actual_input_dim} (detected from {train_dataset.dataset_name})"
    )

    # Update config with actual audio dimension
    config.audio_dim = actual_input_dim

    # Initialize model using factory function
    model = create_model(config).to(device)

    # Initialize text encoder if needed (external, for backward compatibility)
    text_encoder = None
    if modality in ["text", "both"] and not hasattr(model, "text_encoder"):
        text_encoder = FrozenBERTEncoder(
            model_name=getattr(config, "text_model_name", "bert-base-uncased")
        ).to(device)

    # Calculate class weights
    class_counts = [0, 0, 0, 0]
    class_difficulties = [[], [], [], []]

    for item in train_dataset.data:
        label = item["label"]
        difficulty = item["difficulty"]
        class_counts[label] += 1
        class_difficulties[label].append(difficulty)

    class_weights = []
    freq_weights = []
    for i in range(4):
        freq_ratio = class_counts[i] / total_samples
        freq_weight = (1.0 / freq_ratio) / 4
        avg_difficulty = (
            sum(class_difficulties[i]) / len(class_difficulties[i])
            if class_difficulties[i]
            else 1.0
        )
        class_weights.append(freq_weight + 0.75 * avg_difficulty)
        freq_weights.append(freq_weight + 1)
        print(f"########### LABEL {i} ###############")
        print(
            f"  freq_weight: {freq_weight:.4f} ----   avg_difficulty: {avg_difficulty:.4f}"
        )
        print(f"  class_count: {class_counts[i]} ({class_counts[i]/total_samples:.2%})")

    # Normalize weights
    total_weight = sum(class_weights)
    class_weights = [w / total_weight * 4 for w in class_weights]

    class_weights = torch.tensor(class_weights).to(device)

    if config.use_difficulty_scaling:
        print(f"📊 Class weights (freq × difficulty): {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    else:
        print(f"📊 Class weights (freq): {freq_weights}")
        freq_weights = torch.tensor(freq_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=freq_weights, reduction="none")

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = create_lr_scheduler(optimizer, config)

    # Get curriculum pacing function
    pacing_function = get_curriculum_pacing_function(config.curriculum_pacing)

    # Early stopping setup
    use_early_stopping = getattr(config, "use_early_stopping", False)
    if use_early_stopping:
        # Start early stopping only after curriculum learning completes
        curriculum_epochs = (
            config.curriculum_epochs if config.use_curriculum_learning else 0
        )
        early_stopping = EarlyStopping(
            patience=getattr(config, "early_stopping_patience", 10),
            min_delta=getattr(config, "early_stopping_min_delta", 0.001),
            mode="max",  # Maximize UAR
            start_epoch=curriculum_epochs,
        )
        print(
            f"📉 Early stopping enabled (patience={early_stopping.patience}, starting after epoch {curriculum_epochs})"
        )

    # Training loop with curriculum learning
    best_val_acc = 0
    best_val_uar = 0
    model_confidences = None

    for epoch in range(config.num_epochs):

        # Increase dropout after curriculum learning ends
        if epoch == config.curriculum_epochs:
            post_dropout = getattr(config, "post_curriculum_dropout", 0.6)
            print(
                f"   Epoch {epoch+1}: Increasing dropout from {config.dropout} to {post_dropout}"
            )
            model.set_dropout_rate(post_dropout)

        # Create curriculum subset if enabled
        if config.use_curriculum_learning and epoch < config.curriculum_epochs:
            if config.curriculum_type == "preset_order":
                use_preset = True
            else:
                use_preset = False
            curriculum_indices = create_curriculum_subset(
                [train_dataset[i] for i in train_indices],
                train_difficulties,
                epoch,
                config.curriculum_epochs,
                pacing_function,
                use_preset=use_preset,
                curriculum_type=config.curriculum_type,
                model_confidences=model_confidences,
            )
            curriculum_train_indices = [train_indices[i] for i in curriculum_indices]
            curriculum_subset = Subset(train_dataset, curriculum_train_indices)
            train_loader = DataLoader(
                curriculum_subset, batch_size=config.batch_size, shuffle=False
            )

            fraction = pacing_function(epoch, config.curriculum_epochs)
            print(
                f"   Epoch {epoch+1}: Using {len(curriculum_indices)}/{len(train_indices)} samples ({fraction:.2f})"
            )
        else:
            # Use all training data
            train_loader = create_data_loader(
                train_subset,
                config.batch_size,
                shuffle=True,
                use_speaker_disentanglement=config.use_speaker_disentanglement,
            )
            if epoch == config.curriculum_epochs:
                print(
                    f"   Epoch {epoch+1}: Curriculum complete, using all training data"
                )

        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            config,
            text_encoder=text_encoder,
            use_difficulty_scaling=config.use_difficulty_scaling,
        )

        val_results = evaluate_model_multimodal(
            model,
            val_loader,
            criterion,
            device,
            config,
            text_encoder=text_encoder,
            create_plots=False,
        )

        # Calculate model confidences for next epoch
        if (
            config.curriculum_type == "model_confidence"
            and config.use_curriculum_learning
            and epoch < config.num_epochs - 1
        ):
            print(f"   Calculating model confidences for epoch {epoch + 2}...")
            model_confidences = calculate_model_confidences_multimodal(
                model, train_dataset, train_indices, device, config, text_encoder
            )

        val_dict = {
            "val/accuracy": val_results["accuracy"],
            "val/loss": val_results["loss"],
            "val/uar": val_results["uar"],
            "val/f1": val_results["f1_weighted"],
        }

        wandb.log(val_dict)

        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            best_model_state = model.state_dict().copy()

        if val_results["uar"] > best_val_uar:
            best_val_uar = val_results["uar"]

        print(
            f"   Epoch {epoch+1}: Train Acc={train_metrics['accuracy']:.4f}, Val Acc={val_results['accuracy']:.4f}, Val UAR={val_results['uar']:.4f}"
        )

        # Step metric-based scheduler (ReduceLROnPlateau) after validation
        if scheduler is not None and isinstance(
            scheduler, lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_results["uar"])

        # Check early stopping
        if use_early_stopping:
            if early_stopping(val_results["uar"], epoch):
                print(f"   🛑 Stopping training at epoch {epoch+1}")
                break

    # Load best model and evaluate on test sets with plots
    model.load_state_dict(best_model_state)
    val_results = evaluate_model_multimodal(
        model,
        val_loader,
        criterion,
        device,
        config,
        text_encoder=text_encoder,
        create_plots=True,
        plot_title=f"Validation-{train_dataset.dataset_name}",
    )

    test_results = []
    for test_loader, test_name in zip(test_loaders, test_names):
        test_result = evaluate_model_multimodal(
            model,
            test_loader,
            criterion,
            device,
            config,
            text_encoder=text_encoder,
            create_plots=True,
            plot_title=f"CrossCorpus-{test_name}",
        )
        test_results.append({"dataset": test_name, "results": test_result})
        print(
            f"   {test_name}: Acc={test_result['accuracy']:.4f}, UAR={test_result['uar']:.4f}"
        )

    # Log to wandb
    if wandb.run:
        log_dict = {
            "validation/accuracy": val_results["accuracy"],
            "validation/uar": val_results["uar"],
        }

        # Add validation plots
        if "confusion_matrix" in val_results:
            log_dict["validation/confusion_matrix"] = val_results["confusion_matrix"]
        if "difficulty_plot" in val_results:
            log_dict["validation/difficulty_plot"] = val_results["difficulty_plot"]
            if "difficulty_analysis" in val_results:
                analysis = val_results["difficulty_analysis"]
                log_dict["validation/difficulty_correlation"] = analysis[
                    "difficulty_accuracy_correlation"
                ]

        # Add test results and plots
        for test_result in test_results:
            dataset_name = test_result["dataset"].lower()
            prefix = f"{train_dataset.dataset_name}_TO_{dataset_name}"
            results = test_result["results"]
            log_dict[f"{prefix}/accuracy"] = results["accuracy"]
            log_dict[f"{prefix}/uar"] = results["uar"]

            # Add test plots
            if "confusion_matrix" in results:
                log_dict[f"{prefix}/confusion_matrix"] = results["confusion_matrix"]
            if "difficulty_plot" in results:
                log_dict[f"{prefix}/difficulty_plot"] = results["difficulty_plot"]
                if "difficulty_analysis" in results:
                    analysis = results["difficulty_analysis"]
                    log_dict[f"{prefix}/difficulty_correlation"] = analysis[
                        "difficulty_accuracy_correlation"
                    ]

        wandb.log(log_dict)

    results = {
        "validation": val_results,
        "test_results": test_results,
        "best_val_accuracy": best_val_acc,
    }

    return results


def run_experiment_with_seeds(config):
    """
    Run experiment with multiple seeds and compute averaged results
    Creates WandB runs for each seed and a final averaged section
    """
    seeds = getattr(config, "seeds", [42])

    if len(seeds) == 1:
        # Single seed - run normally
        config.seed = seeds[0]
        return run_experiment(config)

    # Multi-seed experiment
    print(f"\n{'='*60}")
    print(f"🎲 MULTI-SEED EXPERIMENT: {config.experiment_name}")
    print(f"   Running with {len(seeds)} seeds: {seeds}")
    print(f"{'='*60}\n")

    all_results = []
    original_exp_name = config.experiment_name

    # Run experiment for each seed
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"🌱 SEED {seed_idx+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        # Update config for this seed
        config.seed = seed
        config.experiment_name = f"{original_exp_name}_seed{seed}"

        # Run experiment
        result = run_experiment(config)
        all_results.append(result)

    # Compute averaged results
    print(f"\n{'='*60}")
    print(f"📊 COMPUTING AVERAGED RESULTS ACROSS {len(seeds)} SEEDS")
    print(f"{'='*60}")

    averaged_results = compute_averaged_results(all_results, config)

    # Log averaged results to WandB
    log_averaged_results_to_wandb(averaged_results, config, original_exp_name, seeds)

    return averaged_results


def compute_averaged_results(all_results, config):
    """
    Compute mean and std across multiple seed runs
    """
    averaged = {}

    if config.evaluation_mode == "cross_corpus":
        # Extract validation metrics
        val_accs = [r["validation"]["accuracy"] for r in all_results]
        val_uars = [r["validation"]["uar"] for r in all_results]

        averaged["validation"] = {
            "accuracy_mean": np.mean(val_accs),
            "accuracy_std": np.std(val_accs),
            "uar_mean": np.mean(val_uars),
            "uar_std": np.std(val_uars),
        }

        # Extract test results for each dataset
        test_datasets = [tr["dataset"] for tr in all_results[0]["test_results"]]
        averaged["test_results"] = []

        for dataset_name in test_datasets:
            # Collect metrics for this dataset across all seeds
            accs = []
            uars = []
            f1s = []

            for result in all_results:
                for test_result in result["test_results"]:
                    if test_result["dataset"] == dataset_name:
                        accs.append(test_result["results"]["accuracy"])
                        uars.append(test_result["results"]["uar"])
                        f1s.append(test_result["results"]["f1_weighted"])
                        break

            averaged["test_results"].append(
                {
                    "dataset": dataset_name,
                    "accuracy_mean": np.mean(accs),
                    "accuracy_std": np.std(accs),
                    "uar_mean": np.mean(uars),
                    "uar_std": np.std(uars),
                    "f1_mean": np.mean(f1s),
                    "f1_std": np.std(f1s),
                }
            )

    elif config.evaluation_mode == "loso":
        # Extract LOSO metrics
        loso_acc_means = [r["loso_accuracy_mean"] for r in all_results]
        loso_acc_stds = [r["loso_accuracy_std"] for r in all_results]
        loso_uar_means = [r["loso_uar_mean"] for r in all_results]
        loso_uar_stds = [r["loso_uar_std"] for r in all_results]

        averaged["loso"] = {
            "accuracy_mean": np.mean(loso_acc_means),
            "accuracy_std_across_seeds": np.std(loso_acc_means),
            "accuracy_std_within_seeds": np.mean(loso_acc_stds),
            "uar_mean": np.mean(loso_uar_means),
            "uar_std_across_seeds": np.std(loso_uar_means),
            "uar_std_within_seeds": np.mean(loso_uar_stds),
        }

    elif config.evaluation_mode == "both":
        # Handle both evaluation modes
        # LOSO
        loso_acc_means = [r["loso"]["loso_accuracy_mean"] for r in all_results]
        loso_uar_means = [r["loso"]["loso_uar_mean"] for r in all_results]

        averaged["loso"] = {
            "accuracy_mean": np.mean(loso_acc_means),
            "accuracy_std": np.std(loso_acc_means),
            "uar_mean": np.mean(loso_uar_means),
            "uar_std": np.std(loso_uar_means),
        }

        # Cross-corpus
        val_accs = [r["cross_corpus"]["validation"]["accuracy"] for r in all_results]
        val_uars = [r["cross_corpus"]["validation"]["uar"] for r in all_results]

        averaged["cross_corpus"] = {
            "validation": {
                "accuracy_mean": np.mean(val_accs),
                "accuracy_std": np.std(val_accs),
                "uar_mean": np.mean(val_uars),
                "uar_std": np.std(val_uars),
            },
            "test_results": [],
        }

        # Test results
        test_datasets = [
            tr["dataset"] for tr in all_results[0]["cross_corpus"]["test_results"]
        ]
        for dataset_name in test_datasets:
            accs = []
            uars = []

            for result in all_results:
                for test_result in result["cross_corpus"]["test_results"]:
                    if test_result["dataset"] == dataset_name:
                        accs.append(test_result["results"]["accuracy"])
                        uars.append(test_result["results"]["uar"])
                        break

            averaged["cross_corpus"]["test_results"].append(
                {
                    "dataset": dataset_name,
                    "accuracy_mean": np.mean(accs),
                    "accuracy_std": np.std(accs),
                    "uar_mean": np.mean(uars),
                    "uar_std": np.std(uars),
                }
            )

    return averaged


def get_train_dataset_name(config):
    """
    Get a clean train dataset name for logging, handling both single and multiple datasets
    """
    if isinstance(config.train_dataset, list):
        return "+".join(config.train_dataset)
    else:
        return config.train_dataset


def log_averaged_results_to_wandb(averaged_results, config, experiment_name, seeds):
    """
    Log averaged results to WandB in special averaged sections
    """
    # Get train dataset name (handles both single and multiple datasets)
    train_dataset_name = get_train_dataset_name(config)

    # Create a new WandB run for averaged results
    wandb.init(
        project=config.wandb_project,
        name=f"{experiment_name}_AVERAGED_",
        config={
            **vars(config),
            "seeds": seeds,
            "num_seeds": len(seeds),
            "averaged_run": True,
        },
        tags=["averaged", "multi-seed"],
    )

    print(f"\n{'='*60}")
    print(f"AVERAGED RESULTS ACROSS {len(seeds)} SEEDS")
    print(f"{'='*60}")

    if config.evaluation_mode == "cross_corpus":
        # Log validation averages
        val = averaged_results["validation"]
        print(f"\nValidation:")
        print(f"  Accuracy: {val['accuracy_mean']:.4f} ± {val['accuracy_std']:.4f}")
        print(f"  UAR: {val['uar_mean']:.4f} ± {val['uar_std']:.4f}")

        wandb.log(
            {
                f"AVERAGED_{train_dataset_name}/validation_accuracy_mean": val[
                    "accuracy_mean"
                ],
                f"AVERAGED_{train_dataset_name}/validation_accuracy_std": val[
                    "accuracy_std"
                ],
                f"AVERAGED_{train_dataset_name}/validation_uar_mean": val["uar_mean"],
                f"AVERAGED_{train_dataset_name}/validation_uar_std": val["uar_std"],
            }
        )

        # Log test averages for each dataset
        for test_result in averaged_results["test_results"]:
            dataset_name = test_result["dataset"]
            print(f"\n{train_dataset_name} → {dataset_name}:")
            print(
                f"  Accuracy: {test_result['accuracy_mean']:.4f} ± {test_result['accuracy_std']:.4f}"
            )
            print(
                f"  UAR: {test_result['uar_mean']:.4f} ± {test_result['uar_std']:.4f}"
            )

            prefix = (
                f"AVERAGED_{train_dataset_name}/{train_dataset_name}to{dataset_name}"
            )
            wandb.log(
                {
                    f"{prefix}_accuracy_mean": test_result["accuracy_mean"],
                    f"{prefix}_accuracy_std": test_result["accuracy_std"],
                    f"{prefix}_uar_mean": test_result["uar_mean"],
                    f"{prefix}_uar_std": test_result["uar_std"],
                    f"{prefix}_f1_mean": test_result["f1_mean"],
                    f"{prefix}_f1_std": test_result["f1_std"],
                }
            )

    elif config.evaluation_mode == "loso":
        loso = averaged_results["loso"]
        print(f"\nLOSO:")
        print(
            f"  Accuracy: {loso['accuracy_mean']:.4f} ± {loso['accuracy_std_across_seeds']:.4f}"
        )
        print(f"  UAR: {loso['uar_mean']:.4f} ± {loso['uar_std_across_seeds']:.4f}")

        wandb.log(
            {
                f"AVERAGED_{train_dataset_name}/loso_accuracy_mean": loso[
                    "accuracy_mean"
                ],
                f"AVERAGED_{train_dataset_name}/loso_accuracy_std": loso[
                    "accuracy_std_across_seeds"
                ],
                f"AVERAGED_{train_dataset_name}/loso_uar_mean": loso["uar_mean"],
                f"AVERAGED_{train_dataset_name}/loso_uar_std": loso[
                    "uar_std_across_seeds"
                ],
            }
        )

    elif config.evaluation_mode == "both":
        # LOSO
        loso = averaged_results["loso"]
        print(f"\nLOSO:")
        print(f"  Accuracy: {loso['accuracy_mean']:.4f} ± {loso['accuracy_std']:.4f}")
        print(f"  UAR: {loso['uar_mean']:.4f} ± {loso['uar_std']:.4f}")

        wandb.log(
            {
                f"AVERAGED_{train_dataset_name}/loso_accuracy_mean": loso[
                    "accuracy_mean"
                ],
                f"AVERAGED_{train_dataset_name}/loso_accuracy_std": loso[
                    "accuracy_std"
                ],
                f"AVERAGED_{train_dataset_name}/loso_uar_mean": loso["uar_mean"],
                f"AVERAGED_{train_dataset_name}/loso_uar_std": loso["uar_std"],
            }
        )

        # Cross-corpus
        val = averaged_results["cross_corpus"]["validation"]
        print(f"\nValidation:")
        print(f"  Accuracy: {val['accuracy_mean']:.4f} ± {val['accuracy_std']:.4f}")
        print(f"  UAR: {val['uar_mean']:.4f} ± {val['uar_std']:.4f}")

        wandb.log(
            {
                f"AVERAGED_{train_dataset_name}/validation_accuracy_mean": val[
                    "accuracy_mean"
                ],
                f"AVERAGED_{train_dataset_name}/validation_accuracy_std": val[
                    "accuracy_std"
                ],
                f"AVERAGED_{train_dataset_name}/validation_uar_mean": val["uar_mean"],
                f"AVERAGED_{train_dataset_name}/validation_uar_std": val["uar_std"],
            }
        )

        for test_result in averaged_results["cross_corpus"]["test_results"]:
            dataset_name = test_result["dataset"]
            print(f"\n{train_dataset_name} → {dataset_name}:")
            print(
                f"  Accuracy: {test_result['accuracy_mean']:.4f} ± {test_result['accuracy_std']:.4f}"
            )
            print(
                f"  UAR: {test_result['uar_mean']:.4f} ± {test_result['uar_std']:.4f}"
            )

            prefix = (
                f"AVERAGED_{train_dataset_name}/{train_dataset_name}to{dataset_name}"
            )
            wandb.log(
                {
                    f"{prefix}_accuracy_mean": test_result["accuracy_mean"],
                    f"{prefix}_accuracy_std": test_result["accuracy_std"],
                    f"{prefix}_uar_mean": test_result["uar_mean"],
                    f"{prefix}_uar_std": test_result["uar_std"],
                }
            )

    print(f"{'='*60}\n")
    wandb.finish()


def main(config_path=None, experiment_id=None, all_experiments=False):
    """Main training function"""
    if config_path:
        if all_experiments:
            return run_all_experiments_from_yaml(config_path)
        else:
            config = load_config_from_yaml(config_path, experiment_id)
            return run_experiment_with_seeds(config)
    else:
        print("📄 Using default config")
        config = Config()
        return run_experiment_with_seeds(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal Emotion Recognition with Curriculum Learning"
    )
    parser.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Experiment ID/name for multi-experiment configs",
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Run all experiments in config file"
    )

    args = parser.parse_args()

    try:
        # Convert experiment to int if it's a number
        experiment_id = args.experiment
        if experiment_id and experiment_id.isdigit():
            experiment_id = int(experiment_id)

        main(args.config, experiment_id, args.all)
    except Exception as e:
        import traceback

        print(f"💥 Fatal error: {e}")
        print(f"🔍 Full traceback:")
        print(traceback.format_exc())
        sys.exit(1)
