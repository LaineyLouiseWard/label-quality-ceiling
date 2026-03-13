# Bootstrap Confidence Intervals (Tile-Level Resampling)

Resamples: 2000 | Seed: 42 | Level: 95%

## stage1_baseline/val
- Tiles: 231
- **mIoU**: 73.3%  [70.2%, 75.8%]
- **mF1**:  84.1%  [81.7%, 85.9%]
- **OA**:   90.8%  [89.6%, 92.0%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 71.5% | [69.1%, 73.7%] |
| Grassland | 89.7% | [88.0%, 91.2%] |
| Cropland | 81.2% | [70.8%, 88.0%] |
| Settlement | 67.7% | [64.5%, 70.6%] |
| Seminatural | 56.5% | [45.1%, 66.0%] |

## stage1_baseline/test
- Tiles: 230
- **mIoU**: 72.5%  [68.5%, 75.8%]
- **mF1**:  83.7%  [80.6%, 86.0%]
- **OA**:   89.6%  [87.8%, 91.1%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 73.3% | [70.8%, 75.6%] |
| Grassland | 88.0% | [85.6%, 90.0%] |
| Cropland | 74.3% | [63.2%, 82.2%] |
| Settlement | 69.9% | [67.5%, 72.2%] |
| Seminatural | 57.1% | [42.6%, 69.0%] |

## stage2_replication/val
- Tiles: 231
- **mIoU**: 73.7%  [70.6%, 76.4%]
- **mF1**:  84.4%  [81.9%, 86.3%]
- **OA**:   91.2%  [89.9%, 92.3%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 72.6% | [70.3%, 74.7%] |
| Grassland | 90.1% | [88.5%, 91.6%] |
| Cropland | 78.8% | [68.0%, 86.6%] |
| Settlement | 69.9% | [66.7%, 72.7%] |
| Seminatural | 57.0% | [46.0%, 66.0%] |

## stage2_replication/test
- Tiles: 230
- **mIoU**: 73.2%  [69.4%, 76.6%]
- **mF1**:  84.2%  [81.2%, 86.6%]
- **OA**:   90.1%  [88.3%, 91.6%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 74.5% | [72.2%, 76.7%] |
| Grassland | 88.3% | [85.9%, 90.3%] |
| Cropland | 72.4% | [60.8%, 81.3%] |
| Settlement | 71.0% | [68.6%, 73.1%] |
| Seminatural | 59.7% | [45.2%, 72.2%] |

## stage3b_finetune/val
- Tiles: 231
- **mIoU**: 79.0%  [76.4%, 81.2%]
- **mF1**:  88.0%  [86.3%, 89.4%]
- **OA**:   92.8%  [91.8%, 93.7%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 74.6% | [72.4%, 76.6%] |
| Grassland | 91.7% | [90.4%, 92.8%] |
| Cropland | 87.5% | [79.1%, 92.8%] |
| Settlement | 72.5% | [69.4%, 75.2%] |
| Seminatural | 69.0% | [59.5%, 76.9%] |

## stage3b_finetune/test
- Tiles: 230
- **mIoU**: 78.7%  [75.3%, 81.3%]
- **mF1**:  87.9%  [85.5%, 89.5%]
- **OA**:   92.1%  [90.8%, 93.3%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 76.6% | [74.5%, 78.6%] |
| Grassland | 90.7% | [88.8%, 92.2%] |
| Cropland | 82.1% | [74.7%, 87.8%] |
| Settlement | 73.4% | [71.2%, 75.4%] |
| Seminatural | 70.5% | [57.8%, 80.5%] |

## stage4_sampling/val
- Tiles: 231
- **mIoU**: 81.8%  [79.6%, 83.5%]
- **mF1**:  89.8%  [88.4%, 90.8%]
- **OA**:   93.8%  [93.2%, 94.4%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 75.4% | [73.3%, 77.4%] |
| Grassland | 92.9% | [92.0%, 93.7%] |
| Cropland | 88.6% | [81.1%, 93.3%] |
| Settlement | 74.3% | [71.3%, 76.9%] |
| Seminatural | 77.8% | [70.5%, 83.5%] |

## stage4_sampling/test
- Tiles: 230
- **mIoU**: 82.0%  [79.6%, 83.7%]
- **mF1**:  90.0%  [88.5%, 91.0%]
- **OA**:   93.4%  [92.5%, 94.1%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 77.6% | [75.5%, 79.5%] |
| Grassland | 92.1% | [90.9%, 93.1%] |
| Cropland | 85.2% | [79.3%, 89.6%] |
| Settlement | 75.5% | [73.4%, 77.5%] |
| Seminatural | 79.5% | [69.8%, 86.1%] |

## stage5_kd/val
- Tiles: 231
- **mIoU**: 83.0%  [80.8%, 84.7%]
- **mF1**:  90.6%  [89.2%, 91.6%]
- **OA**:   94.2%  [93.6%, 94.8%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 76.4% | [74.4%, 78.4%] |
| Grassland | 93.3% | [92.4%, 94.1%] |
| Cropland | 88.8% | [81.1%, 93.7%] |
| Settlement | 76.0% | [73.1%, 78.6%] |
| Seminatural | 80.5% | [73.8%, 85.7%] |

## stage5_kd/test
- Tiles: 230
- **mIoU**: 83.1%  [80.6%, 84.9%]
- **mF1**:  90.6%  [89.1%, 91.7%]
- **OA**:   93.8%  [92.9%, 94.6%]

| Class | IoU | 95% CI |
|-------|-----|--------|
| Forest | 78.7% | [76.7%, 80.5%] |
| Grassland | 92.6% | [91.3%, 93.7%] |
| Cropland | 86.6% | [81.6%, 90.4%] |
| Settlement | 76.7% | [74.6%, 78.7%] |
| Seminatural | 80.7% | [70.7%, 88.0%] |

## Stage 1→5 improvement (val)
- **ΔmIoU**: +9.7%  (individual CI widths: ±2.8%, ±2.0%)
- **ΔmF1**: +6.5%  (individual CI widths: ±2.1%, ±1.2%)
- **ΔOA**: +3.4%  (individual CI widths: ±1.2%, ±0.6%)

## Stage 1→5 improvement (test)
- **ΔmIoU**: +10.5%  (individual CI widths: ±3.6%, ±2.2%)
- **ΔmF1**: +7.0%  (individual CI widths: ±2.7%, ±1.3%)
- **ΔOA**: +4.2%  (individual CI widths: ±1.6%, ±0.8%)
