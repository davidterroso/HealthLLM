"""
File used in the testing of the functions that are in the utils folder
"""

import os
import json
import sys
from unittest.mock import patch, mock_open
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

config_path = os.path.join(os.path.dirname(__file__),
                           '..', 'data_handling', 'config.json')

with open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
    config = json.load(f)

# pylint: disable=wrong-import-position
from utils.get_embeddings_dims import get_embeddings_dims

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
