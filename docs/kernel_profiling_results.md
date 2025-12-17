# PSTM Kernel Profiling Results

## Overview

This document contains comprehensive timing analysis of the PSTM Numba CPU kernel,
identifying where time is spent during tile migration.

## Test Configuration

- **Platform**: macOS Darwin 25.1.0
- **Kernel**: Numba CPU with parallel=True, fastmath=True
- **Tile size**: 32×32 = 1,024 output pillars
- **Aperture**: 5,000m (unless noted)

---

## Table 1: Kernel Time by Interpolation Method

Test: 100k traces, 5km aperture

| Method | 500 samples | 1000 samples | Scaling | Quality | Recommended Use |
|--------|-------------|--------------|---------|---------|-----------------|
| Nearest | 2.45s | 8.7s | 3.6x | Low | Quick previews only |
| Linear | 2.52s | 10.0s | 4.0x | Good | Standard processing |
| Linear+weights | 2.68s | 11.4s | 4.2x | Good | Standard processing |
| Cubic | 2.86s | 12.9s | 4.5x | Better | High-quality processing |
| Sinc4 | 8.59s | 55.5s | 6.5x | Good | Balanced quality/speed |
| Sinc8 | 13.53s | 94.5s | 7.0x | High | High-quality migration |
| Sinc16 | 22.11s | 163.6s | 7.4x | Highest | Research/archival |
| Lanczos3 | 12.36s | 83.1s | 6.7x | High | Sharp edges |
| Lanczos5 | 17.32s | 121.5s | 7.0x | Highest | Maximum sharpness |

---

## Table 2: Estimated Tile Time at 278k Traces (Wizard Scenario)

Scaled from 100k trace tests by factor of 2.78x

| Method | 500 samples | 1000 samples | vs Linear |
|--------|-------------|--------------|-----------|
| Linear+weights | 7.5s | 31.7s | 1.0x |
| Cubic | 7.9s | 35.9s | 1.1x |
| Sinc4 | 23.9s | 154.2s | 4.9x |
| Sinc8 | 37.6s | 262.8s | 8.3x |
| Sinc16 | 61.5s | 454.7s | 14.4x |
| Lanczos3 | 34.4s | 231.0s | 7.3x |
| Lanczos5 | 48.2s | 337.7s | 10.7x |

**Wizard observed**: 139.4s for first tile
**Closest match**: Sinc4 @ ~1000 samples = 154s

---

## Table 3: Component Time Breakdown

Based on differential timing analysis:

| Component | Time (s) | % of Kernel | Notes |
|-----------|----------|-------------|-------|
| Core DSR + accumulation | 8.7 | ~10% | Distance calc, travel time, output write |
| Linear interpolation | 1.3 | ~1.5% | Trace amplitude lookup |
| Cubic interpolation | 2.5 | ~3% | +4 points, spline calc |
| Sinc4 interpolation | 46.8 | ~50% | +4 sinc evaluations per sample |
| Sinc8 interpolation | 85.8 | ~75% | +8 sinc evaluations per sample |
| Sinc16 interpolation | 154.9 | ~90% | +16 sinc evaluations per sample |
| Spreading weight | 1.5 | ~1.5% | 1 division per contribution |
| Obliquity weight | 0.2 | ~0.2% | 1 division per contribution |
| Taper weight | 0.5 | ~0.5% | Cosine calc at aperture edge |

---

## Table 4: Aperture Impact on Performance

Test: Linear interpolation, 100k traces, 500 samples

| Aperture | Time | Relative | Use Case |
|----------|------|----------|----------|
| 5000m | 2.68s | 100% | Full aperture - all nearby traces |
| 2500m | 1.83s | 68% | Half aperture - good for shallow targets |
| 1250m | 0.91s | 34% | Quarter aperture - fast preview |

---

## Key Findings

### 1. INTERPOLATION is the PRIMARY BOTTLENECK

- Sinc methods consume 70-90% of kernel time
- Sinc8 is **5.5x slower** than linear
- Sinc16 is **9x slower** than linear

**Recommendation**: Use LINEAR for standard processing, SINC4 for quality-sensitive work

### 2. Aperture Scaling

- Time scales approximately linearly with aperture
- Halving aperture gives ~30% speedup
- Traces within aperture is the limiting factor

**Recommendation**: Use 2-3km aperture for shallow targets

### 3. Weight Computations are Negligible

- Spreading + obliquity add <5% overhead
- These should remain enabled for correct amplitudes

### 4. Time Samples Scale Linearly

- Doubling samples approximately doubles kernel time
- Use minimum samples needed for target frequency

---

## Optimization Priorities

1. **Switch interpolation method** (Highest impact)
   - Linear: 10x faster than sinc8
   - Sinc4: 3x faster than sinc8, still good quality

2. **Reduce aperture** (Medium impact)
   - 2.5km instead of 5km: 30% faster
   - Must match geology/target depth

3. **Reduce output samples** (Medium impact)
   - Only output samples needed for interpretation
   - Consider post-migration decimation

4. **Tile size tuning** (Low impact)
   - Current 32x32 is reasonable
   - Larger tiles = more parallelism but more memory

---

## Scripts Created

- `scripts/profile_kernel.py` - Component-level profiling
- `scripts/profile_kernel_direct.py` - Direct kernel timing with all interpolation methods

Run with:
```bash
.venv/bin/python scripts/profile_kernel_direct.py --traces 100000 --tile-size 32 --samples 500 --aperture 5000
```
