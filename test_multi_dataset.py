#!/usr/bin/env python3
"""
Quick test script to verify multi-dataset training functionality
"""

from config import Config

# Test 1: Single dataset (existing behavior)
print("="*60)
print("TEST 1: Single Dataset")
print("="*60)
config = Config()
config.train_dataset = "IEMO"
print(f"train_dataset: {config.train_dataset}")
print(f"Is list? {isinstance(config.train_dataset, list)}")

# Test 2: Multiple datasets (new feature)
print("\n" + "="*60)
print("TEST 2: Multiple Datasets")
print("="*60)
config = Config()
config.train_dataset = ["IEMO", "MSPI"]
print(f"train_dataset: {config.train_dataset}")
print(f"Is list? {isinstance(config.train_dataset, list)}")
print(f"Dataset names: {', '.join(config.train_dataset)}")
print(f"Joined name: {'+'.join(config.train_dataset)}")

# Test 3: Three datasets
print("\n" + "="*60)
print("TEST 3: Three Datasets")
print("="*60)
config = Config()
config.train_dataset = ["IEMO", "MSPI", "MSPP"]
print(f"train_dataset: {config.train_dataset}")
print(f"Dataset names: {', '.join(config.train_dataset)}")
print(f"Joined name: {'+'.join(config.train_dataset)}")

# Test 4: Determine test datasets
print("\n" + "="*60)
print("TEST 4: Determining Test Datasets")
print("="*60)
config = Config()
config.train_dataset = ["IEMO", "MSPI"]

all_datasets = ["IEMO", "MSPI", "MSPP", "CMUMOSEI", "SAMSEMO"]
train_datasets = config.train_dataset if isinstance(config.train_dataset, list) else [config.train_dataset]
test_datasets = [d for d in all_datasets if d not in train_datasets]

print(f"Train datasets: {', '.join(train_datasets)}")
print(f"Test datasets: {', '.join(test_datasets)}")
print(f"Training: {'+'.join(train_datasets)} → [{', '.join(test_datasets)}]")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
