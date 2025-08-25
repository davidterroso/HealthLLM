"""
Function used to check how the metadata is stored in Qdrant
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def debug_stored_data(collection_name: str):
    """Debug function to see what's actually stored in Qdrant"""
    client = QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    points = client.scroll(
        collection_name=collection_name,
        limit=3,
        with_payload=True,
        with_vectors=False
    )

    for point in points[0]:
        payload = point.payload or {}
        print(f"Point ID: {point.id}")
        print(f"Payload keys: {list(payload.keys())}")
        print(f"Payload: {point.payload}")
        print("---")

if __name__ == "__main__":
    debug_stored_data(collection_name="pmc_embeddings")
