from torch import nn

from cnn_classifier.evaluation.latency import measure_latency


def test_latency_report_is_ordered_and_positive():
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 2))
    report = measure_latency(model, (1, 1, 1), "cpu", warmup=2, iterations=5)
    assert report.mean_ms > 0
    assert report.p50_ms <= report.p95_ms
    assert report.throughput > 0
