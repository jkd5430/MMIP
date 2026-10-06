# Test Top-3 analysis

| Run | Top-1 | Top-3 | Top-5 | Top-3 correct | Top-3 missed |
|---|---:|---:|---:|---:|---:|
| q2_plain_cnn | 88.65% | 97.53% | 99.32% | 17,339 | 439 |
| q2_lenet5 | 77.65% | 92.20% | 97.20% | 16,392 | 1,386 |
| q3_plain_cnn_aug | 83.04% | 95.80% | 98.74% | 17,031 | 747 |
| q3_lenet5_aug | 73.82% | 90.75% | 96.54% | 16,133 | 1,645 |

- Best Top-3: q2_plain_cnn (97.53%).
- Augmentation changed plain CNN Top-3 by -1.73 percentage points.
- Augmentation changed LeNet-5 Top-3 by -1.46 percentage points.

## Lowest per-class Top-3

- q2_plain_cnn: heart=0.8803; femur-left=0.9477; kidney-left=0.9598
- q2_lenet5: pancreas=0.8434; femur-left=0.8520; heart=0.8573
- q3_plain_cnn_aug: spleen=0.8429; heart=0.8637; femur-right=0.9281
- q3_lenet5_aug: spleen=0.7389; bladder=0.8166; heart=0.8752
