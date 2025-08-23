"""
File used in the testing of the functions that are in the utils folder
"""

import json
import sys
import logging
from unittest.mock import patch, mock_open
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from utils.get_embeddings_dims import get_embeddings_dims
from utils.logging_config import setup_logging
from utils.load_config import load_config

config = load_config()

def test_get_embeddings_dims_success() -> None:
    """
    Tests the function that gets the embeddings 
    dimensions when the conditions are correct

    Args:
        None

    Returns:
        None
    """
    mock_config: dict[str, object] = {"hf_embedding_model": "some-model"}

    fake_embedding_vector = [0.1] * config["embedding_dim"]

    m_open = mock_open(read_data=json.dumps(mock_config))

    with patch("builtins.open", m_open), \
        patch("json.load", return_value=mock_config), \
        patch("json.dump") as mock_json_dump, \
        patch("utils.get_embeddings_dims.HuggingFaceEmbeddings") as mock_hf_embed:

        mock_instance = mock_hf_embed.return_value
        mock_instance.embed_query.return_value = fake_embedding_vector

        embedding_dim = get_embeddings_dims()

        assert embedding_dim == len(fake_embedding_vector)

        updated_config = mock_config.copy()
        updated_config["embedding_dim"] = embedding_dim
        mock_json_dump.assert_called_once_with(updated_config, m_open(), indent=4)

def test_get_embeddings_dims_raises_without_model() -> None:
    """
    Tests the function that gets the embeddings dimensions 
    whenever no model has been defined

    Args:
        None

    Returns:
        None
    """
    mock_config: dict[str, object] = {}

    m_open = mock_open(read_data=json.dumps(mock_config))

    with patch("builtins.open", m_open), patch("json.load", return_value=mock_config):
        with pytest.raises(ValueError, match="No 'hf_embedding_model' found"):
            get_embeddings_dims()

def test_setup_logging_invalid_level() -> None:
    """
    Tests what happens when the passed value is not 
    a key for the dictionary that converts the given 
    string into a logging.level

    Args:
        None
    
    Returns:
        None
    """
    with pytest.raises(KeyError):
        setup_logging(level_str="nonexistent_level")

def test_setup_logging_valid_level(caplog: pytest.LogCaptureFixture) -> None:
    """
    Args:
        caplog (LogCaptureFixture): object that 
            captures the logged information
    
    Returns:
        None
    """
    setup_logging("debug")

    with caplog.at_level(logging.DEBUG):
        logging.debug("debug message")

    assert "debug message" in caplog.text
    assert caplog.records[0].levelno == logging.DEBUG
