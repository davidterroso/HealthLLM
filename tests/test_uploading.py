"""
Tests the functions responsible for uploading the embeddings to Qdrant
"""

import logging
from unittest.mock import MagicMock, patch
from httpx import Headers
from pytest import raises, MonkeyPatch, LogCaptureFixture
from langchain.schema import Document
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

import data_handling.upload_to_vectordb as upload_mod

upload_mod.config = {
    "embedding_dim": 768,
    "batch_size": 2,
}

def test_initiate_qdrant_session_success(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the function that initiates the Qdrant session, in 
    a successful scenario

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
                configurations file

    Returns:
        None
    """
    monkeypatch.setenv("QDRANT_HOST", "http://fake-url")
    monkeypatch.setenv("QDRANT_API_KEY", "fake-key")

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("data_handling.upload_to_vectordb.QdrantClient",
               return_value=mock_client) as mock_qdrant:
        client = upload_mod.initiate_qdrant_session("my_collection")
        assert client is mock_client
        mock_qdrant.assert_called_once_with(url="http://fake-url", api_key="fake-key")
        mock_client.recreate_collection.assert_called_once()
        _, call_kwargs = mock_client.recreate_collection.call_args
        assert call_kwargs["collection_name"] == "my_collection"
        assert isinstance(call_kwargs["vectors_config"], VectorParams)
        assert call_kwargs["vectors_config"].size == upload_mod.config["embedding_dim"]
        assert call_kwargs["vectors_config"].distance == Distance.COSINE

def test_initiate_qdrant_session_missing_env(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the function that initiates the Qdrant session, 
    when the environment with the keys is missing

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
                configurations file

    Returns:
        None
    """
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "key")
    with raises(ValueError, match="QDRANT_HOST is not set"):
        upload_mod.initiate_qdrant_session("test")

    monkeypatch.setenv("QDRANT_HOST", "url")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    with raises(ValueError, match="QDRANT_API_KEY is not set"):
        upload_mod.initiate_qdrant_session("test")

def test_initiate_qdrant_session_missing_embedding_dim(monkeypatch: MonkeyPatch) -> None:
    """
    Tests the function that initiates the Qdrant session, 
    when the dimensions of the embeddings is missing

    Args:
        monkeypatch (MonkeyPatch): patch used to replace the 
                configurations file

    Returns:
        None
    """
    monkeypatch.setenv("QDRANT_HOST", "url")
    monkeypatch.setenv("QDRANT_API_KEY", "key")

    orig_config = upload_mod.config.copy()
    upload_mod.config.pop("embedding_dim", None)

    with raises(KeyError, match="Missing 'embedding_dim'"):
        upload_mod.initiate_qdrant_session("test")

    upload_mod.config = orig_config

def test_upload_docs_to_qdrant_success(caplog: LogCaptureFixture) -> None:
    """
    Tests the function that uploads the embeddings to the 
    Qdrant server in the correct conditions

    Args:
        caplog (LogCaptureFixture): object that captures 
            the logged information

    Returns:
        None
    """
    caplogging.set_level(logging.WARNING)
    client = MagicMock()
    docs = [
        Document(page_content="chunk1 text", metadata={"title": "Title1", "pmid": "pmid1"}),
        Document(page_content="chunk2 text", metadata={"title": "Title2", "pmid": "pmid1"}),
    ]
    embeddings = [
        [0.1] * upload_mod.config["embedding_dim"],
        [0.2] * upload_mod.config["embedding_dim"],
    ]

    upload_mod.upload_docs_to_qdrant(docs, embeddings, "pmid1", client, "my_collection")

    assert client.upsert.called
    calls = client.upsert.call_args_list

    all_points = []
    for call in calls:
        points = call.kwargs["points"]
        all_points.extend(points)
        for point in points:
            assert isinstance(point, PointStruct)
            assert isinstance(point.id, str)
            assert isinstance(point.payload, dict) and "title" in point.payload
            assert "chunk_index" in point.payload
            assert "text_preview" in point.payload
    assert len(all_points) == len(docs)

def test_upload_docs_to_qdrant_missing_embedding_dim() -> None:
    """
    Tests the function that uploads the embeddings to the 
    Qdrant server when the embedding dimension is missing
    from the configurations

    Returns:
        None
    """
    orig_config = upload_mod.config.copy()
    upload_mod.config.pop("embedding_dim", None)

    try:
        with raises(KeyError, match="Missing 'embedding_dim'"):
            upload_mod.upload_docs_to_qdrant([], [], "base", MagicMock(), "collection")
    finally:
        upload_mod.config = orig_config

def test_upload_docs_to_qdrant_bad_embedding_size(caplog: LogCaptureFixture) -> None:
    """
    Tests the function that uploads the embeddings to the 
    Qdrant server when the embedding's dimension does not
    correspond to the expected

    Args:
        caplog (LogCaptureFixture): object that captures 
            the logged information

    Returns:
        None
    """
    caplogging.set_level(logging.WARNING)
    client = MagicMock()
    docs = [Document(page_content="text", metadata={"title": "Title", "pmid": "pmid"})]
    embeddings = [[0.1] * (upload_mod.config["embedding_dim"] - 1)]

    upload_mod.upload_docs_to_qdrant(docs, embeddings, "pmid", client, "collection")

    assert "Invalid vector size" in caplogging.text
    client.upsert.assert_not_called()

def test_upload_docs_to_qdrant_missing_title(caplog: LogCaptureFixture) -> None:
    """
    Tests the function that uploads the embeddings to the 
    Qdrant server when the title is missing

    Args:
        caplog (LogCaptureFixture): object that captures 
            the logged information

    Returns:
        None
    """
    caplogging.set_level(logging.WARNING)
    client = MagicMock()
    docs = [Document(page_content="text", metadata={"pmid": "pmid"})]
    embeddings = [[0.1] * upload_mod.config["embedding_dim"]]

    upload_mod.upload_docs_to_qdrant(docs, embeddings, "pmid", client, "collection")

    assert "Missing 'title'" in caplogging.text
    client.upsert.assert_not_called()

def test_upload_docs_to_qdrant_upsert_raises() -> None:
    """
    Tests the function that uploads the embeddings to the 
    Qdrant server when the embedding's upsert raises an error

    Returns:
        None
    """
    client = MagicMock()
    docs = [Document(page_content="text", metadata={"title": "Title", "pmid": "pmid"})]
    embeddings = [[0.1] * upload_mod.config["embedding_dim"]]

    def raise_unexpected_response(*_args, **_kwargs) -> None:
        """
        Function used in the raising of an unexpected response 
        error

        Returns:
            None
        """
        raise UnexpectedResponse(
            500,
            "fail content",
            b'{"header": "value"}',
            Headers({"Content-Type": "application/json"})
        )

    client.upsert.side_effect = raise_unexpected_response
    with raises(RuntimeError, match="Qdrant upsert failed"):
        upload_mod.upload_docs_to_qdrant(docs, embeddings, "pmid", client, "collection")

def test_upload_docs_to_qdrant_connection_error() -> None:
    """
    Tests the function that uploads the embeddings to the 
    Qdrant server when a connection error with the server 
    occurs

    Returns:
        None
    """
    client = MagicMock()
    docs = [Document(page_content="text", metadata={"title": "Title", "pmid": "pmid"})]
    embeddings = [[0.1] * upload_mod.config["embedding_dim"]]

    def raise_connection_error(*_args, **_kwargs) -> None:
        """
        Function used in the raising of a connection error

        Returns:
            None
        """
        raise ConnectionError("Connection fail")

    client.upsert.side_effect = raise_connection_error
    with raises(ConnectionError, match="Failed to upload points"):
        upload_mod.upload_docs_to_qdrant(docs, embeddings, "pmid", client, "collection")
