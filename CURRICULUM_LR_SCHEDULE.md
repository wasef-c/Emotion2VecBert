# Curriculum Learning Rate Schedule (Sinusoidal Pattern)

## Overview

The curriculum LR schedule implements a **sinusoidal learning rate pattern** that aligns with curriculum learning:

```
LR Pattern:
    HIGH ━━━━╮         ╭━━━━ MEDIUM-HIGH
              ╲       ╱
               ╲     ╱
                ╲   ╱
                 ╲ ╱
                  ╳  LOW
    ├─────────────┼────────────┤
      Curriculum  │  Full Data
      (easy→hard) │  Training
```

## Rationale

1. **Start HIGH** (beginning of curriculum, easy samples)
   - Easy samples are less noisy and more consistent
   - Can tolerate larger updates without destabilizing training
   - Fast learning on clear examples

2. **Descend to LOW** (end of curriculum, hard samples)
   - Hard samples are more ambiguous and require careful learning
   - Lower LR prevents overfitting to difficult/noisy samples
   - More stable gradients on challenging data

3. **Jump to MEDIUM-HIGH** (post-curriculum, full dataset)
   - Full dataset provides more comprehensive signal
   - Can use higher LR with complete information
   - Faster convergence with all data available

## Configuration

Add these parameters to your YAML config:

```yaml
# Enable the sinusoidal LR schedule
use_curriculum_lr_schedule: true

# Start of curriculum (easy samples): HIGH learning rate
curriculum_start_lr_multiplier: 3.0    # 3x base LR

# End of curriculum (hard samples): LOW learning rate
curriculum_end_lr_multiplier: 0.2      # 0.2x base LR

# After curriculum (full dataset): MEDIUM-HIGH learning rate
post_curriculum_lr_multiplier: 2.0     # 2x base LR

# Decay pattern during curriculum
curriculum_lr_decay: "cosine"  # Options: "cosine", "linear", "sqrt"
```

## Example Timeline

With `base_lr = 5e-6`, `curriculum_epochs = 15`, `total_epochs = 30`:

```
Epoch  0: LR = 1.5e-5  (5e-6 × 3.0)  ← HIGH (easy samples start)
Epoch  1: LR = 1.4e-5  (cosine decay)
Epoch  5: LR = 9.0e-6  (cosine decay)
Epoch 10: LR = 3.5e-6  (cosine decay)
Epoch 14: LR = 1.0e-6  (5e-6 × 0.2)  ← LOW (hard samples end)
Epoch 15: LR = 1.0e-5  (5e-6 × 2.0)  ← JUMP UP (full dataset)
Epoch 16: LR = 1.0e-5  (stays constant)
...
Epoch 29: LR = 1.0e-5  (stays constant)
```

## Decay Types

### Cosine Decay (Recommended)
```python
curriculum_lr_decay: "cosine"
```
- Smooth, gradual descent
- More time spent at higher LR (good for easy samples)
- Accelerates descent near end (prepares for hard samples)

### Linear Decay
```python
curriculum_lr_decay: "linear"
```
- Constant rate of decay
- Predictable, simple behavior

### Square Root Decay
```python
curriculum_lr_decay: "sqrt"
```
- Slow decay at start
- Faster decay at end
- Good if early samples are very easy

## How It Works

The system automatically:

1. **During curriculum** (epochs 0 to `curriculum_epochs - 1`):
   - Calculates LR for each epoch using decay formula
   - Updates optimizer LR at the start of each epoch
   - Prints LR every 5 epochs: `Epoch X: Curriculum LR = Y.YYe-Z`

2. **Transition** (epoch `curriculum_epochs`):
   - Jumps LR to `base_lr * post_curriculum_lr_multiplier`
   - Prints: `Epoch X: Post-curriculum LR jump to Y.YYe-Z`
   - Also increases dropout at this point

3. **After curriculum** (epochs > `curriculum_epochs`):
   - LR stays constant at post-curriculum level
   - Can still be modified by standard LR scheduler if enabled

## Integration with Standard Schedulers

The curriculum LR schedule works **before** standard schedulers:

```
Curriculum Schedule (epochs 0-14):
  - Manual LR control with sinusoidal pattern

Post-Curriculum (epochs 15+):
  - Standard scheduler takes over (cosine, step, plateau, etc.)
  - Starts from post_curriculum_lr_multiplier value
```

If you use a standard scheduler (e.g., `lr_scheduler: "cosine"`), it will:
- Be **ignored** during curriculum phase
- **Activate** after curriculum ends

## Recommended Settings

### Conservative (Safe Start)
```yaml
curriculum_start_lr_multiplier: 2.0
curriculum_end_lr_multiplier: 0.3
post_curriculum_lr_multiplier: 1.5
curriculum_lr_decay: "cosine"
```

### Aggressive (If Easy Samples Are Very Clean)
```yaml
curriculum_start_lr_multiplier: 5.0
curriculum_end_lr_multiplier: 0.1
post_curriculum_lr_multiplier: 2.5
curriculum_lr_decay: "cosine"
```

### Gentle (If Hard Samples Are Very Noisy)
```yaml
curriculum_start_lr_multiplier: 1.5
curriculum_end_lr_multiplier: 0.5
post_curriculum_lr_multiplier: 1.2
curriculum_lr_decay: "linear"
```

## Monitoring

During training, you'll see:

```
🌊 Curriculum LR schedule enabled (sinusoidal pattern):
   Start of curriculum: 1.50e-05 (3.0x base)
   End of curriculum: 1.00e-06 (0.2x base)
   After curriculum: 1.00e-05 (2.0x base)
   Decay type: cosine

...

   Epoch 1: Curriculum LR = 1.50e-05
   Epoch 6: Curriculum LR = 8.50e-06
   Epoch 11: Curriculum LR = 3.00e-06
   Epoch 16: Post-curriculum LR jump to 1.00e-05
```

Also logged to WandB as `lr` in the training logs.

## Disabling the Feature

To use standard constant/scheduled LR instead:

```yaml
use_curriculum_lr_schedule: false
```

The system will fall back to:
- Using `learning_rate` throughout training
- Applying standard `lr_scheduler` if configured

## Theory

This approach is inspired by:

1. **Cyclical Learning Rates** (Smith, 2017) - varying LR improves generalization
2. **Curriculum Learning** (Bengio et al., 2009) - easy-to-hard sample ordering
3. **Learning Rate Warmup** (Goyal et al., 2017) - but inverted for curriculum

The key insight: **Easy samples need less cautious learning** than hard samples. Traditional warmup assumes all data is equally difficult. Curriculum learning lets us adapt LR to sample difficulty.

## Troubleshooting

**Training unstable during early curriculum?**
- Reduce `curriculum_start_lr_multiplier` (try 2.0 instead of 3.0)

**Not learning enough during curriculum?**
- Increase `curriculum_end_lr_multiplier` (try 0.5 instead of 0.2)
- Use "linear" decay instead of "cosine"

**Poor performance after curriculum ends?**
- Increase `post_curriculum_lr_multiplier` (try 2.5 or 3.0)
- The model may need stronger updates with full data

**Want to visualize the LR schedule?**
- Check your WandB dashboard, LR is logged each epoch
- Or add `print(f"Current LR: {optimizer.param_groups[0]['lr']}")` to track it
