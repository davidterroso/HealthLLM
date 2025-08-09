import json
import tempfile
from data_handling.get_data import save_checkpoint, load_checkpoint

def test_checkpoint_save_and_load(tmp_path):
    checkpoint_file = tmp_path / "checkpoint.json"
    data = {"last_processed": 42}

    save_checkpoint(checkpoint_file, data)
    assert checkpoint_file.exists()

    loaded = load_checkpoint(checkpoint_file)
    assert loaded == data
