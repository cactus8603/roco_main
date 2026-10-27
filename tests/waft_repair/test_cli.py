"""Exercise command wiring, including serializable preparation results."""
from pathlib import Path
import json

from waft_repair.cli import main


def test_help_has_descriptive_train_and_inference_commands(capsys):
    import pytest
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "train" in output and "infer-images" in output and "prepare" in output


def test_prepare_command_prints_manifest_path(monkeypatch, tmp_path, capsys):
    from waft_repair import backend
    output = tmp_path / "prepared" / "manifest.json"
    monkeypatch.setattr(backend, "prepare_dataset", lambda *args, **kwargs: output)
    main(["prepare", "--spec", "images.json", "--output", str(output.parent),
          "--waft-source", "external/WAFT", "--waft-checkpoint", "weights.pth"])
    assert json.loads(capsys.readouterr().out) == {"manifest": str(output), "target": "cheap"}
