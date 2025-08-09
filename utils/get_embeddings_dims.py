"""
Obtains the dimensions of the embeddings for the 
embedding model selected in the config.json file 
"""

import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_dims() -> int:
    """
    Used to obtain the dimensions of the embeddings, when 
    using the embedding model selected in the config.json file

    Args:
        None
    
    Returns:
        embedding_dim (int): dimensions of the embeddings
    """

    config_path = Path(__file__).resolve().parent.parent \
        / "data_handling" / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("hf_embedding_model")
    if not model_name:
        raise ValueError("No 'hf_embedding_model' found in config.json")

    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    test_vector = embedding_function.embed_query("dimension check")
    embedding_dim = len(test_vector)

    config["embedding_dim"] = embedding_dim

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    return embedding_dim
