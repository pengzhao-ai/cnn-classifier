# CIFAR-10 core ablations

| Run | Test accuracy | 95% CI | Macro F1 | Trainable params | Train time | Latency (ms) |
|---|---:|---|---:|---:|---:|---:|
| augmentation | 0.6792 | 0.6706 - 0.6880 | 0.6787 | 482,730 | 0.3 min | n/a |
| none_augmentation | 0.7106 | 0.7019 - 0.7191 | 0.7096 | 482,730 | 0.3 min | n/a |
| batchnorm | 0.6792 | 0.6706 - 0.6880 | 0.6787 | 482,730 | 0.3 min | n/a |
| no_batchnorm | 0.4855 | 0.4758 - 0.4946 | 0.4732 | 482,058 | 0.3 min | n/a |
| residual | 0.6792 | 0.6706 - 0.6880 | 0.6787 | 482,730 | 0.3 min | n/a |
| no_residual | 0.6860 | 0.6775 - 0.6947 | 0.6852 | 482,730 | 0.3 min | n/a |
| cosine | 0.6792 | 0.6706 - 0.6880 | 0.6787 | 482,730 | 0.3 min | n/a |
| constant | 0.5944 | 0.5860 - 0.6037 | 0.5918 | 482,730 | 0.3 min | n/a |

Pairs: augmentation on/off, BatchNorm on/off, residual on/off, cosine/constant schedule.
All eight runs use the same 10,000/2,000 reduced split, seed 42, 8-epoch budget, and model shape.
Per-run bootstrap intervals are not a paired significance test; overlapping intervals are not evidence of no effect.
This is one seed and a reduced budget, not a universal ranking.
