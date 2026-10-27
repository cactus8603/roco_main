"""Exercise command wiring, including serializable preparation results."""
from pathlib import Path
import json
import pytest

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


@pytest.mark.parametrize("steps",[None,32,256,512])
def test_prepare_forwards_expensive_budget(monkeypatch,tmp_path,capsys,steps):
    from waft_repair import backend
    captured={}
    def prepare(*args,**kwargs):
        captured.update(kwargs)
        return tmp_path/"manifest.json"
    monkeypatch.setattr(backend,"prepare_dataset",prepare)
    argv=["prepare","--spec","images.json","--output",str(tmp_path),
          "--target","expensive","--waft-source","source","--waft-checkpoint","weights"]
    if steps is not None:
        argv+=["--target-steps",str(steps)]
    main(argv)
    assert captured["target_steps"]==steps and captured["target_kind"]=="expensive"
    assert json.loads(capsys.readouterr().out)["target"]=="expensive"


@pytest.mark.parametrize("extra",[["--target-steps","64"],
                                  ["--target-steps","32.0"],
                                  ["--target","cheap","--target-steps","32"]])
def test_invalid_cli_budget_never_starts_preparation(monkeypatch,tmp_path,capsys,extra):
    from waft_repair import backend
    def forbidden(*args,**kwargs):
        raise AssertionError("preparation started")
    monkeypatch.setattr(backend,"prepare_dataset",forbidden)
    with pytest.raises(SystemExit) as result:
        main(["prepare","--spec","images.json","--output",str(tmp_path),
              "--waft-source","source","--waft-checkpoint","weights",*extra])
    assert result.value.code==2
