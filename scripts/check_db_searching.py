"""
Script called to process the test the search in the vector 
database with proper metadata access
"""

from requests import RequestException
from langchain.schema import Document
from data_handling.db_searching import search_docs
from utils.load_config import load_config

config = load_config()

def print_document_details(doc: Document, index: int) -> None:
    """
    Helper function to print document details nicely
    
    Args:
        doc (Document): document desired to print
        index (int): index of the retrieved chunk
        score (float): similarity score between 
            documents

    Returns:
        None
    """
    print(f"\n=== Document {index + 1} ===")
    print(f"Similarity Score: {doc.metadata['similarity_score']:.4f}")

    print(f"Title: {doc.metadata.get('title', 'N/A')}")
    print(f"Journal: {doc.metadata.get('journal', 'N/A')}")
    print(f"Year: {doc.metadata.get('year', 'N/A')}")
    print(f"PMID: {doc.metadata.get('pmid', 'N/A')}")
    print(f"DOI: {doc.metadata.get('doi', 'N/A')}")
    print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
    print(f"File: {doc.metadata.get('file', 'N/A')}")
    print(f"\nContent Preview: {doc.page_content[:200]}...")
    print("-" * 80)


if __name__ == "__main__":
    QUERY = (
        "In all sexual animals and plants, production of an egg cell involves meiosis, "
        "the complex cellular process (involving DNA replication, recombination, and "
        "two nuclear divisions) whereby one diploid nucleus (with two copies of each "
        "chromosome) becomes four genetically different haploid nuclei"
    )

    print("=" * 80)
    print("TESTING RAG PIPELINE WITH METADATA ACCESS")
    print("=" * 80)
    print(f"Query: {QUERY}")
    print("=" * 80)

    search_exceptions = (ValueError, RuntimeError, ConnectionError, RequestException, KeyError)

    print("\n\nResults:")
    try:
        direct_docs = search_docs(query=QUERY, k=3)
        for i, document in enumerate(direct_docs):
            print_document_details(document, index=i)
    except search_exceptions as e:
        print(f"Direct search failed: {e}")
