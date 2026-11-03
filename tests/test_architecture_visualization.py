from __future__ import annotations

from cnn_classifier.config import load_config
from cnn_classifier.config.schema import ModelConfig
from cnn_classifier.diagnostics.architecture import architecture_stages
from cnn_classifier.models import build_model
from cnn_classifier.viz.architecture import plot_cnn_pipeline


def test_simple_cnn_stages_have_shapes_and_params():
    model = build_model(
        ModelConfig(name="simple_cnn", params={"channels": [8, 16], "dropout": 0.0}),
        num_classes=3,
        in_channels=1,
        image_size=(16, 16),
    )

    stages = architecture_stages(model, (1, 16, 16))

    assert [stage.name for stage in stages] == ["stage 1", "stage 2", "head"]
    assert stages[0].output_shape == (8, 8, 8)
    assert stages[1].output_shape == (16, 4, 4)
    assert stages[-1].output_shape == (3,)
    assert sum(stage.params for stage in stages) == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_residual_cnn_stages_match_reference_shapes():
    config = load_config("configs/cifar10/scratch_baseline.yaml")
    model = build_model(
        config.model,
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )

    stages = architecture_stages(model, (3, 32, 32))

    assert [stage.name for stage in stages] == [
        "stem",
        "stage 1",
        "stage 2",
        "stage 3",
        "stage 4",
        "head",
    ]
    assert [stage.output_shape for stage in stages[:5]] == [
        (64, 32, 32),
        (64, 16, 16),
        (128, 8, 8),
        (256, 4, 4),
        (512, 4, 4),
    ]
    assert stages[-1].output_shape == (10,)
    assert all(stage.residual for stage in stages[1:5])
    assert sum(stage.params for stage in stages) == 7_827_786


def test_transfer_stages_mark_frozen_backbone_and_trainable_head():
    config = load_config("configs/cifar10/resnet18_linear_probe_cifar.yaml")
    transfer = config.model.transfer.model_copy(update={"weights": None})
    model_config = config.model.model_copy(update={"transfer": transfer})
    model = build_model(
        model_config,
        num_classes=10,
        in_channels=3,
        image_size=(32, 32),
    )

    stages = architecture_stages(model, (3, 32, 32))

    assert stages[0].name == "stem"
    assert stages[0].frozen
    assert len([stage for stage in stages if stage.name.startswith("layer")]) == 4
    assert all(stage.frozen for stage in stages[:-1])
    assert stages[-1].name == "head"
    assert stages[-1].trainable_params == 5_130


def test_pipeline_figure_writes_png_and_svg(tmp_path):
    model = build_model(
        ModelConfig(name="simple_cnn", params={"channels": [8, 16], "dropout": 0.0}),
        num_classes=3,
        in_channels=1,
        image_size=(16, 16),
    )
    stages = architecture_stages(model, (1, 16, 16))

    path = plot_cnn_pipeline(
        stages,
        input_shape=(1, 16, 16),
        num_classes=3,
        path=tmp_path / "architecture.png",
        title="Simple CNN",
    )

    assert path.exists()
    assert path.stat().st_size > 0
    assert (tmp_path / "architecture.svg").exists()
