from cnn_classifier.config.schema import ModelConfig
from cnn_classifier.models import build_model, model_summary


def test_model_summary_reports_params_and_output_shape():
    model = build_model(
        ModelConfig(name="simple_cnn", params={"channels": [4, 8]}),
        num_classes=3,
        in_channels=3,
        image_size=(8, 8),
    )
    summary = model_summary(model, (3, 8, 8))
    assert summary["total_params"] == sum(p.numel() for p in model.parameters())
    assert summary["trainable_params"] == summary["total_params"]
    assert summary["output_shape"] == [1, 3]
    assert summary["input_shape"] == [3, 8, 8]
    assert summary["modules"]
