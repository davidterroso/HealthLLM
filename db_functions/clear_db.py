"""
Clears all the entries in the vector database
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def wipe_collection(collection_name: str) -> None:
    """
    Deletes all points (embeddings) from a Qdrant collection after
    user confirmation

    Args:
        collection_name (str): name of the collection to wipe

    Returns:
        None
    """
    url = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key)

    confirm = input(f"WARNING: This will DELETE ALL embeddings from collection \
                     '{collection_name}'. Continue? (y/n): ").strip().lower()

    if confirm == "y":
        client.delete(
            collection_name=collection_name,
            points_selector={"filter": {}},
            wait=True
        )
        print(f"All embeddings deleted from collection '{collection_name}'.")
    else:
        print("Operation cancelled.")
