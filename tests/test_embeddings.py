"""
Tests the functions behind the embeddings of the text
"""

import os
import json
import sys
import logging
from typing import List
from pathlib import Path
from unittest.mock import MagicMock, patch
from pytest import fixture, raises, LogCaptureFixture, MonkeyPatch
from langchain.schema import Document

sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
import data_handling.embedding_functions as emb_func_module
from data_handling.embedding_functions import embed_chunks, embed_docs

config_path = os.path.join(os.path.dirname(__file__),
                           '..', 'data_handling', 'config.json')

with open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
    config = json.load(f)

@fixture
def sample_docs() -> List[Document]:
    """
    Pytest fixture to create fake LangChain documents

    Args:
        None

    Returns:
        List[Document]: list of fake LangChain 
            documents 
    """
    return [
        Document(page_content="This is a test.", metadata={"pmid": "123", "title": "Test title 1"}),
        Document(page_content="Another chunk.", metadata={"pmid": "123", "title": "Test title 2"})
    ]

def test_embed_docs_empty_list(caplog: LogCaptureFixture) -> None:
    """
    Tests the embedding function when it receives a
    list of empy documents

    Args:
        caplog (LogCaptureFixture): object that 
            captures the logged information

    Returns:
        None
    """
    caplogging.set_level(logging.WARNING)
    client = MagicMock()

    embed_docs([], client, "test_collection")
    assert "No documents provided for embedding." in caplogging.text

def test_embed_docs_empty_content(sample_docs: List[Document], # pylint: disable=redefined-outer-name, unused-argument
                                  caplog: LogCaptureFixture) -> None:
    """
    Tests the embedding function when it receives multiple empty
    documents

    Args:
        sample_docs: list of LangChain Documents
        caplog (LogCaptureFixture): object that captures the 
            logged information

    Returns:
        None
    """
    caplogging.set_level(logging.WARNING)
    empty_docs = [
        Document(page_content="   ", metadata={"pmid": "123"}),
        Document(page_content="", metadata={"pmid": "123"})
    ]
    client = MagicMock()

    with patch("data_handling.embedding_functions.embed_chunks", return_value=[[0.1, 0.2]]):
        with patch("data_handling.upload_to_vectordb.upload_docs_to_qdrant"):
            embed_docs(empty_docs, client, "test_collection")
    assert "All provided documents have empty content." in caplogging.text

def test_embed_docs_missing_pmid_raises(sample_docs: List[Document]) -> None: # pylint: disable=redefined-outer-name, unused-argument
    """
    Tests the embedding function when given files 
    without PMID

    Args:
        sample_docs: list of LangChain Documents

    Returns:
        None
    """
    docs = [
        Document(page_content="Valid text", metadata={})
    ]
    client = MagicMock()

    with patch("data_handling.embedding_functions.embed_chunks", return_value=[[0.1, 0.2]]):
        with raises(KeyError, match="missing 'pmid'"):
            embed_docs(docs, client, "test_collection")

def test_embed_docs_calls_upload_docs_to_qdrant(sample_docs: List[Document]) -> None: # pylint: disable=redefined-outer-name, unused-argument
    """
    Tests the embedding function when calling the
    uploading functions

    Args:
        sample_docs: list of LangChain Documents

    Returns:
        None
    """
    mock_embeddings = [[0.1] * 768, [0.2] * 768]
    client = MagicMock()

    with patch("data_handling.embedding_functions.embed_chunks",
               return_value=mock_embeddings) as mock_embed_chunks:
        with patch("data_handling.embedding_functions.upload_docs_to_qdrant") as mock_upload:
            embed_docs(sample_docs, client, "my_collection")

    mock_embed_chunks.assert_called_once_with(chunks=["This is a test.", "Another chunk."])
    mock_upload.assert_called_once()
    _, kwargs = mock_upload.call_args
    assert kwargs["collection_name"] == "my_collection"
    assert kwargs["embeddings"] == mock_embeddings
    assert kwargs["base_id"] == "123"

def test_embed_chunks_returns_vectors_and_checks_size(monkeypatch: MonkeyPatch)-> None:
    """
    Tests the returns of the chunking function

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file

    Returns:
        None 
    """
    monkeypatch.setattr(emb_func_module,
                        "embedding_function",
                        MagicMock(
        embed_documents=lambda x: [[0.1] * config["embedding_dim"] for _ in x]
    ))
    chunks = ["text1", "text2"]
    vectors = embed_chunks(chunks)
    assert len(vectors) == 2
    assert all(len(vec) == config["embedding_dim"] for vec in vectors)

def test_embed_chunks_on_wrong_size(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the embedding function when the chunking function 
    returns an unexpected size

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file

    Returns:
        None 
    """
    monkeypatch.setattr(emb_func_module,
                        "embedding_function",
                        MagicMock(
        embed_documents=lambda x: [[0.1] * (config["embedding_dim"] - 1) for _ in x]
    ))
    with raises(ValueError, match="Invalid vector size"):
        embed_chunks(["bad chunk"])

def test_embed_chunks_logs_and_raises_on_error(monkeypatch: MonkeyPatch,
                                               caplog: LogCaptureFixture) -> None:
    """
    Tests the logs and errors raised when something is wrong 
    with the embedding

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file
        caplog (LogCaptureFixture): object that captures the 
            logged information

    Returns:
        None
    """
    def raise_error(_: List[str]) -> None:
        """
        Function called to raise an error

        Args:
            _ (List[str]): throwaway variable

        Returns:
            None
        """
        raise RuntimeError("embedding failed")
    monkeypatch.setattr(emb_func_module,
                        "embedding_function",
                        MagicMock(embed_documents=raise_error))
    caplogging.set_level(logging.ERROR)

    with raises(RuntimeError):
        embed_chunks(["text"])
    assert "Batch embedding failed" in caplogging.text
