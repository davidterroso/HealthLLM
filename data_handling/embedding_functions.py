"""
This file is used in the embedding of strings using
the preferences selected in the config.json file
"""

import os
import json
from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from data_handling.upload_to_vectordb import upload_docs_to_qdrant
from utils.logging_config import TqdmLogger as log

config_path = os.path.join(os.path.dirname(__file__),
                           '..', 'data_handling', 'config.json')

with open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
    config = json.load(f)

embedding_function = HuggingFaceEmbeddings(
    model_name=config["hf_embedding_model"]
)

def embed_docs(docs: List[Document],
               client: QdrantClient,
               collection_name: str) -> None:
    """
    Function used to embed a List of chunks, handling them 
    individually, and exporting the results to the VectorDB

    Args:
        docs (List[Document]): list of chunks, and their 
            metadata, to embed
        client (QdrantClient): iniated QdrantClient object
        collection_name (str): name of the Qdrant collection

    Returns:
        None
    """
    if not docs:
        log.warning("No documents provided for embedding.")
        return

    chunks = [doc.page_content.strip() for doc in docs if doc.page_content.strip()]

    if not chunks:
        log.warning("All provided documents have empty content.")


    embeddings = embed_chunks(chunks=chunks)

    try:
        base_id = docs[0].metadata["pmid"]
    except KeyError as e:
        raise KeyError("First document is missing 'pmid' in its metadata.") from e

    upload_docs_to_qdrant(docs=docs,
                          embeddings=embeddings,
                          base_id=base_id,
                          client=client,
                          collection_name=collection_name)

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Function used to embed any set of chunks of text, using 
    the given HuggingFace model

    Args:
        chunks (List[str]): chunks desired to embedd
    
    Returns:
        vectors (List[List[float]]): embedding of the given chunks
    """
    try:
        vectors = embedding_function.embed_documents(chunks)
    except Exception as e:
        log.error(str("Batch embedding failed: %s", e))
        raise

    for i, vector in enumerate(vectors):
        if len(vector) != config["embedding_dim"]:
            raise ValueError(
                f"Invalid vector size for doc #{i}: expected {config['embedding_dim']},\
                got {len(vector)}."
            )

    return vectors
