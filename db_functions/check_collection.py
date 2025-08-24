"""
Quick test script to connect to your HealthLLM Qdrant Cloud cluster
and check the pmc_embeddings collection
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def test_pmc_connection() -> None:
    """
    Tests connection to your pmc_embeddings collection
    by retrieving some documents from the vector database

    Args:
        None
    
    Returns:
        None
    """

    url = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")

    try:
        print("Connecting to HealthLLM Qdrant Cloud...")
        client = QdrantClient(url=url, api_key=api_key)

        print("Getting collections...")
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]

        print("Connection successful!")
        print(f"Found {len(collection_names)} collections: {collection_names}")

        pmc_collection = "pmc_embeddings"
        if pmc_collection in collection_names:
            print(f"\nFound '{pmc_collection}' collection!")

            info = client.get_collection(pmc_collection)
            print(f"  ├ Total documents: {info.points_count}")

            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                for name, params in vectors.items():
                    print(f"  ├ Vector '{name}' dimension: {getattr(params, 'size', 'Unknown')}")
                    print(f"  ├ Vector '{name}' distance metric: \
                          {getattr(params, 'distance', 'Unknown')}")
            elif vectors is not None:
                print(f"  ├ Vector dimension: {getattr(vectors, 'size', 'Unknown')}")
                print(f"  └ Distance metric: {getattr(vectors, 'distance', 'Unknown')}")
            else:
                print("  ├ Vector config not found.")

            print("\nSample documents:")
            points, _ = client.scroll(
                collection_name=pmc_collection,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            for i, point in enumerate(points):
                payload = point.payload or {}
                print(f"  {i+1}. ID: {str(point.id)}")
                print(f"     Title: {payload.get('title', 'No title')}")
                print(f"     PMID: {payload.get('pmid', 'No PMID')}")
                print(f"     Chunk: {payload.get('chunk_index', 'Unknown')}")
                if 'text' in payload:
                    print(f"     Preview: {payload['text']}")
                print()
        else:
            print(f"Collection '{pmc_collection}' not found!")
            print("Available collections:", collection_names)
            print("\nThis might mean the data hasn't been uploaded yet.")

    except ConnectionError as e:
        print("Connection failed: %s", e)
        print("\n Troubleshooting:")
        print("1. Check your QDRANT_HOST URL format")
        print("2. Verify your QDRANT_API_KEY is correct")
        print("3. Make sure your cluster is running (not paused)")

if __name__ == "__main__":
    test_pmc_connection()
