from pathlib import Path

import torch

from models.registry import _clean_state_dict_keys, load_models


def test_clean_state_dict_keys_strips_model_and_module_prefixes():
    state_dict = {
        "model.conv1.weight": torch.zeros(1),
        "module.model.layer1.0.conv1.weight": torch.ones(1),
        "fc.weight": torch.full((1,), 2.0),
    }

    cleaned = _clean_state_dict_keys(state_dict)

    assert "conv1.weight" in cleaned
    assert "layer1.0.conv1.weight" in cleaned
    assert "fc.weight" in cleaned
    assert "model.conv1.weight" not in cleaned
    assert "module.model.layer1.0.conv1.weight" not in cleaned


def test_load_models_returns_empty_when_models_dir_missing(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    result = load_models(models_dir=str(missing), device=torch.device("cpu"))
    assert result == []


def test_load_models_returns_empty_when_no_valid_model_files(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "README.txt").write_text("no model files here", encoding="utf-8")

    result = load_models(models_dir=str(models_dir), device=torch.device("cpu"))
    assert result == []
