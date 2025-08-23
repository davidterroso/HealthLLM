"""
Tests the chunking functions during data extraction
"""

import sys
from pathlib import Path
from langchain.schema import Document
from pytest import MonkeyPatch, raises

sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from data_handling.get_data import text_chunker
from utils.load_config import load_config

config = load_config()

def test_text_chunker_basic(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the chunker basic functions

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file

    Returns:
        None
    """
    monkeypatch.setitem(config, "chunk_size", 10)
    monkeypatch.setitem(config, "chunk_overlap", 2)

    text = "This is a simple test string to be chunked into parts."
    metadata = {"source": "unit_test"}

    docs = text_chunker(text, metadata)

    assert isinstance(docs, list)
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.metadata == metadata for d in docs)
    assert any("test" in d.page_content for d in docs)

def test_text_chunker_empty_string(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the chunker when given an empty string

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file

    Returns:
        None
    """
    monkeypatch.setitem(config, "chunk_size", 10)
    monkeypatch.setitem(config, "chunk_overlap", 2)

    docs = text_chunker("", None)
    assert docs == []

def test_text_chunker_none_metadata(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the chunker when given an empty metadata
    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file

    Returns:
        None
    """
    monkeypatch.setitem(config, "chunk_size", 10)
    monkeypatch.setitem(config, "chunk_overlap", 2)

    text = "Short text"
    docs = text_chunker(text, None)
    assert all(isinstance(d.metadata, dict) for d in docs)

def test_text_chunker_invalid_full_text_type():
    """
    Tests the chunker when given an invalid text type

    Args:
        None

    Returns:
        None
    """
    with raises(TypeError):
        text_chunker(1234, None)

def test_text_chunker_invalid_metadata_type():
    """
    Tests the chunker when given an invalid metadata type

    Args:
        None

    Returns:
        None
    """
    with raises(TypeError):
        text_chunker("text", metadata="not a dict")
