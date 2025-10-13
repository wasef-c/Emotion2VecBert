# Multi-Seed Experiments Guide

## Overview

The system now supports running experiments with multiple random seeds to ensure statistical significance of results. Each experiment can be run with multiple seeds, and the system will automatically:

1. Run the experiment once for each seed
2. Compute average metrics across all seeds
3. Compute standard deviations
4. Create a special "AVERAGED" WandB run with aggregated results

## Configuration

### Single Seed (Default)
```yaml
seeds: [42]  # or just: seed: 42
```

### Multiple Seeds
```yaml
seeds: [42, 123, 456]  # Run with 3 different seeds
```

### In Config Files

Example from `multimodal_baseline.yaml`:
```yaml
template_config: &template
  seeds: [42, 123, 456]  # Will run 3 times and average
  learning_rate: 9e-3
  # ... other params
```

## WandB Organization

### Individual Seed Runs

Each seed creates a separate WandB run named:
```
{experiment_name}_seed{seed_value}
```

For example:
- `Multimodal Full System IEMO_seed42`
- `Multimodal Full System IEMO_seed123`
- `Multimodal Full System IEMO_seed456`

### Averaged Run

After all seeds complete, an additional WandB run is created:
```
{experiment_name}_AVERAGED
```

This run is tagged with `["averaged", "multi-seed"]` for easy filtering.

## Logged Metrics

### For Cross-Corpus Evaluation

#### Individual Seeds
Each seed logs to its own run:
- `validation/accuracy`
- `validation/uar`
- `{train_dataset}_TO_{test_dataset}/accuracy`
- `{train_dataset}_TO_{test_dataset}/uar`

#### Averaged Run
The AVERAGED run logs:
- `AVERAGED/validation_accuracy_mean`
- `AVERAGED/validation_accuracy_std`
- `AVERAGED/validation_uar_mean`
- `AVERAGED/validation_uar_std`
- `AVERAGED/{train}to{test}_accuracy_mean`
- `AVERAGED/{train}to{test}_accuracy_std`
- `AVERAGED/{train}to{test}_uar_mean`
- `AVERAGED/{train}to{test}_uar_std`
- `AVERAGED/{train}to{test}_f1_mean`
- `AVERAGED/{train}to{test}_f1_std`

### Example: IEMO → MSPI

For a cross-corpus experiment (IEMO → MSPI, MSPP) with 3 seeds:

**Individual Seed Runs:**
```
IEMO_TO_mspi/accuracy  (seed 42)
IEMO_TO_mspi/accuracy  (seed 123)
IEMO_TO_mspi/accuracy  (seed 456)
```

**Averaged Run:**
```
AVERAGED/IEMOtoMSPI_accuracy_mean = 0.723
AVERAGED/IEMOtoMSPI_accuracy_std = 0.015
AVERAGED/IEMOtoMSPI_uar_mean = 0.698
AVERAGED/IEMOtoMSPI_uar_std = 0.012
```

### For LOSO Evaluation

#### Averaged Run Logs:
- `AVERAGED/loso_accuracy_mean`
- `AVERAGED/loso_accuracy_std`
- `AVERAGED/loso_uar_mean`
- `AVERAGED/loso_uar_std`

## Output Format

### Console Output

```
============================================================
🎲 MULTI-SEED EXPERIMENT: Multimodal Full System IEMO
   Running with 3 seeds: [42, 123, 456]
============================================================

============================================================
🌱 SEED 1/3: 42
============================================================
🔢 Random seed set to: 42
...
[Training output for seed 42]
...

============================================================
🌱 SEED 2/3: 123
============================================================
🔢 Random seed set to: 123
...
[Training output for seed 123]
...

============================================================
🌱 SEED 3/3: 456
============================================================
🔢 Random seed set to: 456
...
[Training output for seed 456]
...

============================================================
📊 COMPUTING AVERAGED RESULTS ACROSS 3 SEEDS
============================================================

============================================================
AVERAGED RESULTS ACROSS 3 SEEDS
============================================================

Validation:
  Accuracy: 0.7543 ± 0.0123
  UAR: 0.7234 ± 0.0098

IEMO → MSPI:
  Accuracy: 0.7234 ± 0.0156
  UAR: 0.6987 ± 0.0134

IEMO → MSPP:
  Accuracy: 0.6876 ± 0.0142
  UAR: 0.6654 ± 0.0128
============================================================
```

## Usage Examples

### Example 1: Run with Default Seeds from Config
```bash
python main.py --config configs/multimodal_baseline.yaml --experiment 0
```

This will use the seeds specified in the config file (e.g., `[42, 123, 456]`).

### Example 2: Override Seeds via Config Edit

Edit your config file:
```yaml
experiments:
  - <<: *template
    name: "My Experiment"
    seeds: [42, 100, 200, 300, 400]  # 5 seeds for more robust results
```

### Example 3: Single Seed (No Averaging)
```yaml
experiments:
  - <<: *template
    name: "Quick Test"
    seeds: [42]  # Only one seed, no averaging
```

## Statistical Interpretation

### Standard Deviation
The std values show the variability across different random initializations:
- **Low std (< 0.01)**: Results are very stable across seeds
- **Medium std (0.01-0.03)**: Normal variability
- **High std (> 0.03)**: Results may be sensitive to initialization

### Reporting Results

When reporting in papers, use:
```
"We report mean ± std over 3 random seeds."

Example: "Our multimodal system achieves 72.34% ± 1.56% accuracy
on IEMO→MSPI cross-corpus evaluation."
```

## Best Practices

### Number of Seeds

- **Prototype/Debug**: 1 seed (faster iteration)
- **Development**: 3 seeds (good balance)
- **Publication**: 5-10 seeds (more robust statistics)

### Computational Cost

Running with N seeds multiplies training time by N:
- 1 seed: 1x time
- 3 seeds: 3x time
- 5 seeds: 5x time

**Tip**: Use fewer seeds during development, increase for final evaluation.

### Seed Selection

Common choices:
- `[42]` - Single seed (de facto standard)
- `[42, 123, 456]` - Three seeds (common practice)
- `[42, 123, 456, 789, 1024]` - Five seeds (more robust)
- `[0, 1, 2, 3, 4]` - Sequential (easier to remember)

## WandB Filtering

### View Only Averaged Runs
Filter by tag: `averaged`

### View Individual Seed Runs
Exclude tag: `averaged` OR search for `_seed` in name

### Compare Across Experiments
1. Filter to show only AVERAGED runs
2. Compare `IEMOtoMSPI_accuracy_mean` across different models/configs

## Configuration Examples

### Minimal Config (Single Seed)
```yaml
experiments:
  - name: "Test"
    learning_rate: 9e-3
    seeds: [42]
```

### Production Config (Multiple Seeds)
```yaml
experiments:
  - name: "Production Run"
    learning_rate: 9e-3
    seeds: [42, 123, 456, 789, 1024]
    # ... other params
```

### Mixed Seeds per Experiment
```yaml
experiments:
  - name: "Quick Baseline"
    seeds: [42]  # Fast

  - name: "Final Multimodal"
    seeds: [42, 123, 456]  # Robust
```

## Troubleshooting

### WandB Shows Too Many Runs
Each seed creates a run, plus one AVERAGED run.
- 3 seeds = 4 total runs (3 individual + 1 averaged)
- Filter by tag `averaged` to see only summary runs

### Inconsistent Results
If std is very high:
1. Check if model/training is stable
2. Increase number of seeds
3. Check for bugs in data loading/processing

### Memory Issues
If running multiple seeds sequentially causes memory issues:
- System cleans up between seeds automatically
- Check for memory leaks in custom code
- Reduce batch size if needed

## Code Implementation

### Key Functions

1. **run_experiment_with_seeds(config)**
   - Main entry point
   - Runs experiment for each seed
   - Collects results

2. **compute_averaged_results(all_results, config)**
   - Computes mean and std
   - Handles different evaluation modes

3. **log_averaged_results_to_wandb(averaged_results, config, experiment_name, seeds)**
   - Creates AVERAGED WandB run
   - Logs all aggregated metrics

### Backward Compatibility

The system is fully backward compatible:
- Old configs with `seed: 42` still work
- Automatically converted to `seeds: [42]`
- Single-seed experiments skip averaging

## Future Enhancements

Potential additions:
- Confidence intervals
- Significance testing between models
- Seed-level result visualization
- Bootstrap resampling
- Cross-seed correlation analysis

---

**Quick Reference:**

| Seeds | Purpose | Time Cost | Use Case |
|-------|---------|-----------|----------|
| [42] | Debug/Fast | 1x | Development |
| [42, 123, 456] | Standard | 3x | Most experiments |
| [42, 123, 456, 789, 1024] | Robust | 5x | Publications |

**WandB Naming:**
- Individual: `{name}_seed{N}`
- Averaged: `{name}_AVERAGED`

**Metrics Format:**
- Individual: `metric_name`
- Averaged: `AVERAGED/{section}_metric_mean` and `AVERAGED/{section}_metric_std`
