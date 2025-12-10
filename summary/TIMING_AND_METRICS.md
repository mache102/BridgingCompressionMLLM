# Timing and Metrics Implementation Summary

## Changes Made

### 1. Timing in ELIC.py `forward()` Method

Added precise timing measurements for each component:
- **g_a** (analysis transform): Input → latent representation
- **h_a** (hyper-encoder): Latent → hyper-latent
- **h_s** (hyper-decoder): Hyper-latent → parameters **[DECODER]**
- **g_s** (synthesis transform): Latent → reconstructed output

All timing uses `time.perf_counter()` with CUDA synchronization for GPU operations.

**Location**: `compressai/models/ELIC.py`, line ~195-348

**Output format**:
```python
{
    "x_hat": reconstructed_image,
    "y_hat": latent_representation,
    "llm_emb": llm_embedding,
    "likelihoods": {...},
    "time": {
        'g_a': float,  # seconds
        'h_a': float,  # seconds
        'h_s': float,  # seconds
        'g_s': float,  # seconds
    }
}
```

### 2. Component Definitions

**Encoder** = g_a + h_a + g_s
- Analysis transform (g_a)
- Hyper-encoder (h_a)
- Synthesis transform (g_s)

**Decoder** = h_s
- Hyper-decoder only

### 3. Training Loop Timing (train.py)

#### Per-batch Timing Collection
- Extracts timing from `out_net['time']`
- Computes encoder time: `(g_a + h_a + g_s) * 1e6 / batch_size` → **microseconds per image**
- Computes decoder time: `h_s * 1e6 / batch_size` → **microseconds per image**
- Updates `AverageMeter` instances for epoch-level aggregation

**Location**: `examples/train.py`, line ~420-430 (training), line ~510-520 (testing)

#### End-of-Epoch Logging
Logs to both:
1. **Comet ML** (experiment tracking)
2. **Standard Python logging** (logging.info)

Format:
```
Train epoch 5: total_loss=1.234567, bpp=0.5678, psnr=28.34, top1=0.6543, encode_us=55692.548, decode_us=5750.010
```

Precision:
- Timing: 3 decimal places (microseconds)
- Loss/metrics: 4-6 decimal places

**Location**: `examples/train.py`, line ~455-460 (train), line ~548-553 (test)

### 4. Model Architecture Metrics (One-Time)

Added helper functions:
- `count_parameters(model)` - Count trainable parameters
- `estimate_macs(model, input_shape, device)` - Estimate MACs using `thop`

Logged once at model initialization:
- Encoder (g_a + h_a + g_s) trainable params
- Decoder (h_s) trainable params
- Adapter trainable params
- Per-component MACs (GMACs)
  - g_a MACs
  - h_a MACs
  - h_s MACs (decoder)
  - g_s MACs
  - Total encoder MACs
  - Adapter MACs

**Location**: `examples/train.py`, line ~35-50 (helpers), line ~678-732 (logging)

**Output format**:
```
============================================================
MODEL ARCHITECTURE STATISTICS (one-time measurement)
============================================================
Encoder (g_a + h_a + g_s) trainable params: 12,345,678
Decoder (h_s) trainable params: 1,234,567
Adapter trainable params: 234,567
g_a MACs: 45.678 GMACs
h_a MACs: 2.345 GMACs
h_s MACs (decoder): 3.456 GMACs
g_s MACs: 43.210 GMACs
Total Encoder MACs: 91.233 GMACs
Adapter MACs: 1.234 GMACs
============================================================
```

## Dependencies

- `thop` (optional): For MACs estimation. Falls back gracefully if not available.
- `torch`: CUDA synchronization for accurate GPU timing
- `logging`: Standard Python logging for console/file output

## Testing

Test script: `test_timing.py`

Example output:
```
g_a time: 59.833 ms
h_a time: 1.539 ms
h_s time: 11.500 ms
g_s time: 50.012 ms

Encoder (g_a + h_a + g_s): 111.385 ms
Decoder (h_s): 11.500 ms

Per-image encoder time: 55692.548 μs
Per-image decoder time: 5750.010 μs
```

## Usage in Training

The timing information is automatically collected during training:

1. **Forward pass**: Model returns timing in `out_net['time']`
2. **Per-batch**: Times are aggregated as per-image microseconds
3. **Per-epoch**: Average times are logged to Comet and console

No manual intervention required - just run your training script as usual.

## Notes

- All timing measurements use CUDA synchronization when GPU is available
- Times are converted to per-image microseconds for consistent reporting
- Encoder includes g_s (synthesis) as it's part of the end-to-end encoding process
- Decoder is only h_s (hyper-decoder) as defined by user requirements
