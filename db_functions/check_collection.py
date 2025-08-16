"""
Quick test script to connect to your HealthLLM Qdrant Cloud cluster
and check the pmc_embeddings collection
"""

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def test_pmc_connection():
    """Test connection to your pmc_embeddings collection"""

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
            print(f"  ├ Vector dimension: {info.config.params.vectors.size}")
            print(f"  └ Distance metric: {info.config.params.vectors.distance}")

            print("\nSample documents:")
            points, _ = client.scroll(
                collection_name=pmc_collection,
                limit=3,
                with_payload=True,
                with_vectors=False
            )

            for i, point in enumerate(points):
                payload = point.payload or {}
                print(f"  {i+1}. ID: {str(point.id)[:20]}...")
                print(f"     Title: {payload.get('title', 'No title')[:60]}...")
                print(f"     Chunk: {payload.get('chunk_index', 'Unknown')}")
                if 'text_preview' in payload:
                    print(f"     Preview: {payload['text_preview'][:80]}...")
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
