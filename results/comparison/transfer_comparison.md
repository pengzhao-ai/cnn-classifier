# 32×32 ResNet-18 transfer benchmark

| Run | Test accuracy | 95% CI | Macro F1 | Trainable params | Train time | Latency (ms) |
|---|---:|---|---:|---:|---:|---:|
| scratch | 0.9088 | 0.9033–0.9143 | 0.9090 | 11,181,642 | 19.6 min | 3.84 |
| linear_probe | 0.6266 | 0.6175–0.6363 | 0.6254 | 5,130 | 8.4 min | 3.75 |
| finetune | 0.9498 | 0.9457–0.9541 | 0.9497 | 11,181,642 | 19.5 min | 3.70 |

All three runs share the CIFAR-10 test set; the recorded test fingerprint is identical across rows.
The scratch control receives ImageNet normalization to match the pretrained input contract; the reference CIFAR baseline uses CIFAR normalization. The difference is intentional and labeled.
BatchNorm running statistics continue to update while the linear-probe parameters are frozen; the probe is not a strict frozen-feature probe.
Per-run bootstrap intervals are not a paired significance test.
