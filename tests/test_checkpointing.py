"""
Tests the functions that are used in the checkpointing, during file 
extraction
"""

import sys
import os
import json
from pathlib import Path
from pytest import fixture, MonkeyPatch

sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from data_handling.get_data import (
    save_checkpoint,
    load_checkpoint
)

@fixture
def config_patch_fixture(monkeypatch: MonkeyPatch, tmp_path: str) -> dict:
    """
    Fixture for a fake configurations file containing a checkpoint path

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file
        tmp_path (str): temporary path to the file

    Returns:
        fake_config (dict): dictionary with the fake configurations
    """
    fake_config = {"checkpoints_path": Path(tmp_path) / "checkpoint.json"}
    monkeypatch.setattr("data_handling.get_data.config", fake_config)
    return fake_config

# Testing function load_checkpoint

def test_load_checkpoint_file_exists(config_patch_fixture: dict) -> None:
    """
    Tests the loading of the existing checkpoint file

    Args:
        config_patch_fixture (dict): fixture defined to mock 
            the configurations

    Returns:
        None
    """
    data = ["file1.xml", "file2.xml"]
    with open(config_patch_fixture["checkpoints_path"], "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj)

    result = load_checkpoint()
    assert isinstance(result, set)
    assert result == {"file1.xml", "file2.xml"}

def test_load_checkpoint_file_missing(config_patch_fixture: dict) -> None:
    """
    Tests the attempt of loading a checkpoint file when it 
    does not exist

    Args:
        config_patch_fixture (dict): fixture defined to mock 
            the configurations

    Returns:
        None
    """

    if os.path.exists(config_patch_fixture["checkpoints_path"]):
        os.remove(config_patch_fixture["checkpoints_path"])
    result = load_checkpoint()
    assert result == set()

# Testing function save_checkpoint

def test_save_checkpoint_and_load_back(config_patch_fixture: dict) -> None:
    """
    Tests the saving of a checkpoint and loading it back

    Args:
        config_patch_fixture (dict): fixture defined to mock 
            the configurations

    Returns:
        None
    """
    files_set = {"a.xml", "b.xml"}
    save_checkpoint(files_set)

    with open(config_patch_fixture["checkpoints_path"], "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    assert set(data) == files_set
    assert load_checkpoint() == files_set

def test_save_checkpoint_overwrites_file(config_patch_fixture: dict) -> None:
    """
    Tests the overwritting of a checkpoint

    Args:
        config_patch_fixture (dict): fixture defined to mock 
            the configurations

    Returns:
        None
    """
    with open(config_patch_fixture["checkpoints_path"], "w", encoding="utf-8") as file_obj:
        json.dump(["old.xml"], file_obj)

    new_set = {"new1.xml", "new2.xml"}
    save_checkpoint(new_set)

    with open(config_patch_fixture["checkpoints_path"], "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    assert set(data) == new_set
    assert set(data) == new_set
