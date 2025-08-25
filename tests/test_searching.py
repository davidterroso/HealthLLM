"""
Tests the functions in db_searching.py file
"""

import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
from pytest import MonkeyPatch, raises
from langchain_core.documents import Document

sys.path.append(str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from data_handling.db_searching import (
    create_embedding_function,
    get_qdrant_store,
    search_docs
)

# Tests the create_embedding_function function

def test_create_embedding_function_success(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the function that creates an embedding in 
    the success scenario

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file 

    Returns:
        None
    """
    monkeypatch.setitem(
        sys.modules["data_handling.db_searching"].config,
        "hf_embedding_model", "fake-model"
    )
    monkeypatch.setitem(
        sys.modules["data_handling.db_searching"].config,
        "collection_name", "test-collection"
    )
    with patch("data_handling.db_searching.HuggingFaceEmbeddings") as mock_embed:
        mock_embed.return_value = MagicMock()
        embedding_func, collection_name = create_embedding_function()
        assert embedding_func is not None
        assert isinstance(collection_name, str)
        assert collection_name == "test-collection"


def test_create_embedding_function_missing_key(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the function that when the function that creates 
    the embeddings is missing a key

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
            configurations file 

    Returns:
        None
    """
    monkeypatch.setattr("data_handling.db_searching.config", {}, raising=False)
    with raises(KeyError):
        create_embedding_function()

# Tests the get_qdrant_store function

@patch("data_handling.db_searching.QdrantClient")
@patch("data_handling.db_searching.QdrantVectorStore")
def test_get_qdrant_store_success(mock_store: MagicMock, mock_client: MagicMock) -> None:
    """
    Tests the function that initiates the Qdrant store in 
    a success scenario

    Args:
        mock_store (MagicMock): mock of the Qdrant
            vector store
        mock_client (MagicMock): mock of the Qdrant 
            client

    Returns:
        None
    """
    fake_embedding = MagicMock()
    store_instance = MagicMock()
    mock_store.return_value = store_instance

    result = get_qdrant_store(fake_embedding, "test-collection")

    assert result == store_instance
    mock_client.assert_called_once()
    mock_store.assert_called_once()


@patch("data_handling.db_searching.QdrantClient", side_effect=Exception("boom"))
def test_get_qdrant_store_unexpected_error(mock_client: MagicMock) -> None: # pylint: disable=unused-argument
    """
    Tests the Qdrant vector store initializer when it 
    gets an unexpected error

    Args:
        mock_client (MagicMock): mock of the Qdrant 
            client

    Returns:
        None
    """
    with raises(RuntimeError):
        get_qdrant_store(MagicMock(), "test-collection")

# Tests the search_docs function

@patch("data_handling.db_searching.QdrantClient")
def test_search_docs_success(mock_client: MagicMock) -> None:
    """
    Tests the function that gets the documents in a success 
    scenario

    Args:
        mock_client (MagicMock): mock of the Qdrant 
            client
    
    Returns:
        None
    """
    fake_embedding_func = MagicMock()
    fake_embedding_func.embed_query.return_value = [0.1, 0.2]

    with patch("data_handling.db_searching.create_embedding_function",
               return_value=(fake_embedding_func, "test-collection")):

        fake_hit = MagicMock()
        fake_hit.payload = {"text": "content", "title": "Paper A"}
        fake_hit.score = 0.95
        mock_client.return_value.search.return_value = [fake_hit]

        results = search_docs("test query", k=1)

        assert len(results) == 1
        doc = results[0]
        assert isinstance(doc, Document)
        assert doc.page_content == "content"
        assert doc.metadata["title"] == "Paper A"
        assert "similarity_score" in doc.metadata


@patch("data_handling.db_searching.QdrantClient")
def test_search_docs_with_none_payload(mock_client: MagicMock) -> None:
    """
    Tests the function that gets the documents but 
    they appear without payload

    Args:
        mock_client (MagicMock): mock of the Qdrant 
            client
    
    Returns:
        None
    """
    fake_embedding_func = MagicMock()
    fake_embedding_func.embed_query.return_value = [0.1, 0.2]

    with patch("data_handling.db_searching.create_embedding_function",
               return_value=(fake_embedding_func, "test-collection")):

        fake_hit = MagicMock()
        fake_hit.payload = None
        fake_hit.score = 0.8
        mock_client.return_value.search.return_value = [fake_hit]

        results = search_docs("test query", k=1)

        assert len(results) == 1
        doc = results[0]
        assert doc.page_content == ""
        assert doc.metadata["similarity_score"] == 0.8


@patch("data_handling.db_searching.QdrantClient")
def test_search_docs_failure(mock_client: MagicMock) -> None:
    """
    Tests the function that search documents in 
    a case of failure

    Args:
        mock_client (MagicMock): mock of the Qdrant 
            client
    
    Returns:
        None
    """
    fake_client = MagicMock()
    fake_client.search.side_effect = Exception("search error")
    mock_client.return_value = fake_client

    with patch("data_handling.db_searching.create_embedding_function",
               return_value=(MagicMock(), "test-collection")):
        results = search_docs("test query")
        assert not results
