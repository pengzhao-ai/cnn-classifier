# Transfer paired bootstrap (accuracy)

| Comparison | Difference | 95% CI |
|---|---:|---|
| linear_probe vs scratch | -0.2822 | -0.2919 - -0.2721 |
| finetune vs scratch | +0.0410 | +0.0349 - +0.0469 |
| finetune vs linear_probe | +0.3232 | +0.3135 - +0.3323 |

Paired differences use 1,000 stratified bootstrap resamples on the shared test predictions. A positive difference means the first run has higher accuracy.