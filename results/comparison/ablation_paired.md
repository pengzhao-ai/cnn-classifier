# Ablation paired bootstrap (accuracy)

| Comparison | Difference | 95% CI |
|---|---:|---|
| augmentation (on - off) | -0.0314 | -0.0390 - -0.0242 |
| batchnorm (on - off) | +0.1937 | +0.1830 - +0.2042 |
| residual (on - off) | -0.0068 | -0.0136 - +0.0003 |
| cosine vs constant (on - off) | +0.0848 | +0.0760 - +0.0940 |

Paired differences use 1,000 stratified bootstrap resamples on the shared test predictions. A positive difference means the first run has higher accuracy.