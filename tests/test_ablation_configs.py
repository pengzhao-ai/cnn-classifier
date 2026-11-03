from cnn_classifier.config import load_config

ABLATION_CONFIGS = (
    "configs/ablations/augmentation.yaml",
    "configs/ablations/none_augmentation.yaml",
    "configs/ablations/batchnorm.yaml",
    "configs/ablations/no_batchnorm.yaml",
    "configs/ablations/residual.yaml",
    "configs/ablations/no_residual.yaml",
    "configs/ablations/cosine.yaml",
    "configs/ablations/constant.yaml",
)


def test_ablation_pairs_differ_only_in_intended_knob():
    aug_a = load_config("configs/ablations/augmentation.yaml")
    aug_b = load_config("configs/ablations/none_augmentation.yaml")
    assert aug_a.training.epochs == aug_b.training.epochs
    assert aug_a.data.transforms != aug_b.data.transforms
    assert aug_a.model == aug_b.model
    assert aug_a.training == aug_b.training

    bn_a = load_config("configs/ablations/batchnorm.yaml")
    bn_b = load_config("configs/ablations/no_batchnorm.yaml")
    assert bn_a.model.params["norm"] != bn_b.model.params["norm"]
    assert bn_a.data == bn_b.data
    assert bn_a.training == bn_b.training

    residual_a = load_config("configs/ablations/residual.yaml")
    residual_b = load_config("configs/ablations/no_residual.yaml")
    assert residual_a.model.params["residual"] != residual_b.model.params["residual"]
    assert residual_a.data == residual_b.data
    assert residual_a.training == residual_b.training

    cosine = load_config("configs/ablations/cosine.yaml")
    constant = load_config("configs/ablations/constant.yaml")
    assert cosine.training.scheduler.name == "cosine"
    assert constant.training.scheduler.name == "constant"
    assert cosine.data == constant.data
    assert cosine.model == constant.model


def test_augmentation_pair_changes_only_the_train_pipeline():
    aug_a = load_config("configs/ablations/augmentation.yaml")
    aug_b = load_config("configs/ablations/none_augmentation.yaml")

    train_a = {transform.name for transform in aug_a.data.transforms["train"]}
    train_b = {transform.name for transform in aug_b.data.transforms["train"]}
    assert {"RandomHorizontalFlip", "RandomCrop"} <= train_a
    assert train_b == {"ToTensor", "Normalize"}
    assert aug_a.data.transforms["eval"] == aug_b.data.transforms["eval"]


def test_all_ablation_configs_share_reduced_budget_and_seed():
    configs = [load_config(path) for path in ABLATION_CONFIGS]

    assert len({config.experiment.name for config in configs}) == len(configs)
    assert all(config.experiment.seed == 42 for config in configs)
    assert all(config.model.name == "residual_cnn" for config in configs)
    assert all(config.training.epochs == 8 for config in configs)
    assert all(config.data.adapter_params["max_train_samples"] == 10_000 for config in configs)
    assert all(config.data.adapter_params["max_val_samples"] == 2_000 for config in configs)
