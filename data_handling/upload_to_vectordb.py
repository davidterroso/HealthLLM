"""
This file is used in the uploading each 
entry into the VectorDB that is hosted 
in the cloud
"""

import os
import json
import logging
from typing import List
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def initiate_qdrant_session(collection_name: str) -> QdrantClient:
    """
    Function used to create the Qdrant 
    collection for the data being processed

    Args:
        None

    Returns:
        (QdrantClient Object): returns the 
            initiated QdrantClient Object
    """

    url = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_HOST is not set in environment " \
        "variables or repository secrets.")
    if not api_key:
        raise ValueError("QDRANT_API_KEY is not set in environment " \
        "variables or repository secrets.")

    try:
        client = QdrantClient(
            url = url,
            api_key = api_key,
        )

        if collection_name not in [col.name for col in client.get_collections().collections]:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=config["embedding_dim"], distance=Distance.COSINE)
            )
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e

    return client

def upload_docs_to_qdrant(docs: List[Document],
                          embeddings: List[List[float]],
                          base_id: str,
                          client: QdrantClient,
                          collection_name: str) -> None:
    """
    Uploads the document's chunks embeddings 
    to the VectorDB that is hosted in the cloud 

    Args:
        docs (List[Document]): list of the chunks
            to upload
        embeddings (List[List[float]]): list of 
            the embeddings of each chunk 
        base_id (str): article unique identifier
        client (QdrantClient Object): initiated 
            Qdrant Object
        collection_name (str): name of the 
            Qdrant collection

    Returns:
        None
    """
    points = []

    for i, doc in enumerate(docs):
        try:
            embedding = embeddings[i]
            if len(embedding) != config["embedding_dim"]:
                raise ValueError(f"Invalid vector size: expected \
                                 {config["embedding_dim"]}, got {len(embedding)}")

            point_id = f"{base_id}_chunk_{i}"

            payload={
                **doc.metadata,
                "chunk_index": i,
                "text_preview": doc.page_content[:300]
            }

            if "title" not in payload or payload["title"] is None:
                raise KeyError(f"Missing 'title' in metadata for doc #{i}")

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload
                )
            )

        except IndexError as e:
            logging.warning("[IndexError] Embedding missing for doc #%d: %s", i, e)
            continue
        except KeyError as e:
            logging.warning("[KeyError] Metadata problem for doc #%d: %s", i, e)
        except ValueError as e:
            logging.warning("[ValueError] Invalid data for doc #%d: %s", i, e)
        except TypeError as e:
            logging.warning("[TypeError] Bad data types for doc #%d: %s", i, e)
        except (AttributeError, RuntimeError) as e:
            logging.error("[UnexpectedError] Failed to construct PointStruct for \
                          doc #%d: %s", i, e, exc_info=True)
    if points:
        try:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
        except UnexpectedResponse as e:
            raise RuntimeError(f"Qdrant upsert failed due to bad response: {e}") from e
        except Exception as e:
            raise ConnectionError(f"Failed to upload points to Qdrant: {e}") from e

    else:
        print("[INFO]: No valid points to upload.")
