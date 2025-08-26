"""
Used in the comparison between the embedded queries and the documents 
in the VectorDB
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv
from spellchecker import SpellChecker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
from requests import RequestException
from utils.load_config import load_config

load_dotenv()
config = load_config()

def create_embedding_function() -> Tuple[HuggingFaceEmbeddings, str]:
    """
    Factory function to create the appropriate embedding function and vector store name
    
    Args:
        None

    Returns:
        embedding_function (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): function used 
            in the embedding
        collection_name (str): name of the collection where the embeddings are stored
    """
    try:
        embedding_func = HuggingFaceEmbeddings(
            model_name=config["hf_embedding_model"]
        )
        collection_name = config["collection_name"]
        return embedding_func, collection_name
    except KeyError as e:
        raise KeyError(f"Missing config key in load_config(): {e}") from e
    except OSError as e:
        raise OSError(f"Error loading embedding model \
                      '{config.get('hf_embedding_model')}': {e}") from e

def get_qdrant_store(embedding_function: HuggingFaceEmbeddings,
                     collection_name: str) -> QdrantVectorStore:
    """
    Initiates the Qdrant store and using the secret keys 

    Args:
        embedding_function (HuggingFaceEmbeddings): initiated
            HuggingFaceEmbeddings function
        collection_name (str): name of the Qdrant collection 
            to search

    Returns:
        (Qdrant): initiated Qdrant LangChain vector store
    """
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_function,
            content_payload_key="text"
        )
    except (UnexpectedResponse, ResponseHandlingException) as e:
        raise RuntimeError(f"Qdrant API error: {e}") from e
    except RequestException as e:
        raise ConnectionError(f"Network error while connecting to Qdrant: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating Qdrant store:\
                           {type(e).__name__}: {e}") from e

def correct_query(query: str) -> str:
    """
    Corrects typos in the received query

    Args:
        query (str): query desired to correct

    Returns:
        corrected_query (str): corrected query
    """
    spell = SpellChecker()
    corrected_words = []

    for word in query.split():
        corrected = spell.correction(word)
        if corrected is None:
            corrected_words.append(word)
        else:
            corrected_words.append(corrected)

    corrected_query = " ".join(corrected_words)
    return corrected_query

def search_docs(query: str, k: int = 5) -> List[Document]:
    """
    Search documents method using Qdrant client directly for more control
    
    Args:
        query (str): query to search for
        k (int): number of results to return
        
    Returns:
        List[Document]: list of documents with properly extracted metadata
        or [] if an error occurs
    """
    try:
        embedding_function, collection_name = create_embedding_function()

        query = correct_query(query)

        query_embedding = embedding_function.embed_query(query)

        client = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )

        documents = []
        for hit in search_result:
            payload = hit.payload or {}
            content = payload.get("text", "")

            metadata = {k: v for k, v in payload.items() if k != "text"}
            metadata["similarity_score"] = hit.score

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    except (
        UnexpectedResponse,
        ResponseHandlingException,
        RequestException,
        KeyError,
        OSError,
        ValueError,
        RuntimeError,
    ) as e:
        print(f"Handled error in search_docs: {type(e).__name__}: {e}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Unexpected error in search_docs: {type(e).__name__}: {e}")

    return []
