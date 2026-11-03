from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_PATH = REPO_ROOT / "demos" / "bring_your_own_data" / "scripts" / "run_quickstart.py"
FIXTURE_ROOT = REPO_ROOT / "demos" / "bring_your_own_data" / "fixture"


def _load_quickstart():
    spec = importlib.util.spec_from_file_location("byo_run_quickstart", QUICKSTART_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_byo_fixture_is_sixteen_bit_and_complete():
    images = sorted(FIXTURE_ROOT.glob("train/*/*.png")) + sorted(FIXTURE_ROOT.glob("val/*/*.png"))
    assert len(images) == 60
    assert {path.parent.name for path in images} == {"circle", "square", "stripe"}
    with Image.open(images[0]) as image:
        assert image.mode in {"I", "I;16", "I;16B", "I;16L"}


def test_byo_quickstart_trains_and_writes_contract(tmp_path, byo_config_path):
    module = _load_quickstart()

    result = module.run_quickstart(byo_config_path, output_root=tmp_path)

    contract = json.loads((result.run_dir / "data" / "model_contract.json").read_text())
    assert contract["dtype"] == "float32"
    assert contract["input_shape"] == [1, 16, 16]
    assert contract["label_map"] == {"circle": 0, "square": 1, "stripe": 2}
    assert contract["channel_policy"] == "gray"
    assert contract["value_range"] == [0.0, 1.0]
    assert (result.run_dir / "report.html").exists()
    assert result.report_path == result.run_dir / "report.html"
