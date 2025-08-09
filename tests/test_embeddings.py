from unittest.mock import patch
from data_handling.embedding_functions import embed_chunks

@patch("data_handling.embeddings.embedding_function")
def test_embed_chunk_returns_correct_length(mock_embedding_function):
    mock_embedding_function.embed_query.return_value = [0.1] * 384
    result = embed_chunks("test text")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)
