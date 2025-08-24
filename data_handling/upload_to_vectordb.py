"""
This file is used in the uploading each entry into the VectorDB that 
is hosted in the cloud
"""

import os
import uuid
import logging
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.load_config import load_config

load_dotenv()
config = load_config()

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

def build_point(doc: Document, embedding: List[float], base_id: str, i: int) -> PointStruct:
    """
    Builds a Qdrant Point for a given chunk

    Args:
        doc (Document): chunk that is going to be 
            uploaded
        embedding (List[float]): embeddings of the 
            given chunk
        base_id (str): file identifier
        i (int): index of the chunk

    Returns:
        (PointStruct) Qdrant point ready to be uploaded
    """
    if len(embedding) != config['embedding_dim']:
        raise ValueError(f"Invalid vector size: expected {config['embedding_dim']},\
                         got {len(embedding)}")

    payload = {**doc.metadata, "chunk_index": i, "text": doc.page_content}

    if not payload.get("title"):
        raise KeyError(f"Missing 'title' in metadata for chunk #{i}")

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{base_id}_chunk_{i}"))

    return PointStruct(id=point_id, vector=embedding, payload=payload)

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
        payload = None
        try:
            point = build_point(doc, embeddings[i], base_id, i)
            points.append(point)

        except (IndexError, KeyError, ValueError, TypeError) as e:
            file_val = None
            if payload is not None:
                file_val = payload.get("file")
            elif hasattr(doc, "metadata"):
                file_val = doc.metadata.get("file")
            logging.warning("[%s] Problem with chunk #%d from document %s: %s",
                            type(e).__name__, i, file_val, e)
        except (AttributeError, RuntimeError) as e:
            logging.error("[UnexpectedError] Failed to build point"
                          "for doc #%d: %s", i, e)

    if points:
        for i in range(0, len(points), config['batch_size']):
            batch = points[i:i + config['batch_size']]
            try:
                client.upsert(collection_name=collection_name, points=batch)
            except UnexpectedResponse as e:
                raise RuntimeError(f"Qdrant upsert failed: {e}") from e
            except Exception as e:
                raise ConnectionError(f"Failed to upload points to Qdrant: {e}") from e
    else:
        logging.info("No valid points to upload.")
