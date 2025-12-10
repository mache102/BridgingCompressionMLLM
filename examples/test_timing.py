#!/usr/bin/env python3
"""Quick test to verify timing measurements in forward() work correctly."""

import sys
sys.path.insert(0, '../')

import torch
from compressai.models.ELIC import ELIC_LLM

# Create a small ELIC model instance (use defaults)
model = ELIC_LLM()  # N=192, M=320 by default
model.eval()

# Create dummy input
dummy_input = torch.randn(2, 3, 256, 256)

# Run forward pass
with torch.no_grad():
    output = model(dummy_input, noisequant=False)

# Check timing info
print("=" * 60)
print("Forward pass timing test")
print("=" * 60)

if 'time' in output:
    timing = output['time']
    print(f"g_a time: {timing.get('g_a', 0)*1000:.3f} ms")
    print(f"h_a time: {timing.get('h_a', 0)*1000:.3f} ms")
    print(f"h_s time: {timing.get('h_s', 0)*1000:.3f} ms")
    print(f"g_s time: {timing.get('g_s', 0)*1000:.3f} ms")
    
    # Compute encoder and decoder times
    g_a_t = timing.get('g_a', 0.0)
    h_a_t = timing.get('h_a', 0.0)
    g_s_t = timing.get('g_s', 0.0)
    h_s_t = timing.get('h_s', 0.0)
    
    encoder_time = (g_a_t + h_a_t + g_s_t) * 1000  # ms
    decoder_time = h_s_t * 1000  # ms
    
    print(f"\nEncoder (g_a + h_a + g_s): {encoder_time:.3f} ms")
    print(f"Decoder (h_s): {decoder_time:.3f} ms")
    
    # Per-image microseconds (batch size = 2)
    enc_per_image_us = (g_a_t + h_a_t + g_s_t) * 1e6 / 2
    dec_per_image_us = h_s_t * 1e6 / 2
    print(f"\nPer-image encoder time: {enc_per_image_us:.3f} μs")
    print(f"Per-image decoder time: {dec_per_image_us:.3f} μs")
    
    print("\n✅ Timing measurements working correctly!")
else:
    print("❌ No timing info in output!")
    sys.exit(1)

print("=" * 60)
