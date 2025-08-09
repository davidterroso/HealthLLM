"""
This file is used in the uploading each entry into the VectorDB that 
is hosted in the cloud
"""

import os
import json
import logging
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

config_path = os.path.join(os.path.dirname(__file__),
                           '..', 'data_handling', 'config.json')

with open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
    config = json.load(f)

def initiate_qdrant_session(collection_name: str) -> QdrantClient:
    """
    Function used to create the Qdrant collection for the data 
    being processed

    Args:
        collection_name (str): name of the Qdrant collection

    Returns:
        (QdrantClient Object): returns the initiated QdrantClient 
            Object
    """

    url = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_HOST is not set in environment " \
        "variables or repository secrets.")
    if not api_key:
        raise ValueError("QDRANT_API_KEY is not set in environment " \
        "variables or repository secrets.")

    if "embedding_dim" not in config:
        raise KeyError("Missing 'embedding_dim' in configuration.")

    try:
        client = QdrantClient(url = url, api_key = api_key)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e

    try:
        if collection_name not in [col.name for col \
                                   in client.get_collections().collections]:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config["embedding_dim"],
                    distance=Distance.COSINE
                )
            )
    except Exception as e:
        raise RuntimeError(f"Failed to create/recreate collection"
                           f"'{collection_name}': {e}") from e

    return client

def upload_docs_to_qdrant(docs: List[Document],
                          embeddings: List[List[float]],
                          base_id: str,
                          client: QdrantClient,
                          collection_name: str) -> None:
    """
    Uploads the document's chunks embeddings to the VectorDB 
    that is hosted in the cloud 

    Args:
        docs (List[Document]): list of the chunks to upload
        embeddings (List[List[float]]): list of the embeddings 
            of each chunk 
        base_id (str): article unique identifier
        client (QdrantClient Object): initiated Qdrant Object
        collection_name (str): name of the Qdrant collection

    Returns:
        None
    """

    if "embedding_dim" not in config:
        raise KeyError("Missing 'embedding_dim' in configuration.")

    points = []

    for i, doc in enumerate(docs):
        try:
            embedding = embeddings[i]
            if len(embedding) != config['embedding_dim']:
                raise ValueError(f"Invalid vector size: expected \
                                 {config['embedding_dim']}, got {len(embedding)}")

            payload={
                **doc.metadata,
                "chunk_index": i,
                "text_preview": doc.page_content[:300]
            }

            if not payload.get('title'):
                raise KeyError(f"Missing 'title' in metadata for doc #{i}")

            points.append(
                PointStruct(
                    id=f"{base_id}_chunk_{i}",
                    vector=embedding,
                    payload=payload
                )
            )

        except (IndexError, KeyError, ValueError, TypeError) as e:
            logging.warning("[%s] Problem with doc #%d: %s", type(e).__name__, i, e)
        except (AttributeError, RuntimeError) as e:
            logging.error("[UnexpectedError] Failed to build point"
                          "for doc #%d: %s", i, e, exc_info=True)

    if points:
        total_batches = (len(points) + config['batch_size'] - 1) // config['batch_size']
        for i in tqdm(range(0, len(points), config['batch_size']),
                      total=total_batches, unit="batch", desc="Uploading batches"):
            batch = points[i:i + config['batch_size']]
            try:
                client.upsert(collection_name=collection_name, points=batch)
            except UnexpectedResponse as e:
                raise RuntimeError(f"Qdrant upsert failed: {e}") from e
            except Exception as e:
                raise ConnectionError(f"Failed to upload points to Qdrant: {e}") from e
    else:
        logging.info("No valid points to upload.")
