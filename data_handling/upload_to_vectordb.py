"""
This file is used in the uploading each 
entry into the VectorDB that is hosted 
in the cloud
"""

import os
from typing import List
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

def initiate_qdrant_session():
    """
    Function used to create the Qdrant 
    collection for the data being processed
    """
    client = QdrantClient(
        url = os.getenv("QDRANT_HOST"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )
    collection_name = "pmc-embeddings"

    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

def upload_docs_to_qdrant(docs: List[Document],
                          embeddings: List[List[float]],
                          base_id: str) -> None:
    """
    Uploads the document's chunks embeddings 
    to the VectorDB that is hosted in the cloud 

    Args:
        docs (List[Document]): list of the chunks
            to upload
        embeddings (List[List[float]]): list of 
            the embeddings of each chunk 
        base_id (str): article unique identifier

    Returns:
        None
    """
    points = []

    for i, doc in enumerate(docs):
        point_id = f"{base_id}_chunk_{i}"

        points.append(
            PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload={
                    **doc.metadata,
                    "chunk_index": i,
                    "text_preview": doc.page_content[:300]
                }
            )
        )

    client = QdrantClient(
        url = os.getenv("QDRANT_HOST"),
        api_key = os.getenv("QDRANT_API_KEY"),
    )

    collection_name = "pmc-embeddings"

    client.upsert(
        collection_name=collection_name,
        points=points
    )
