"""
This file is used in the embedding of strings using
the preferences selected in the config.json file
"""

import json
import os
from typing import List
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def embed_docs(chunk: List[str]) -> None:
    """
    Function used to embed any chunk of text, 
    using the given HuggingFace model

    Args:
        docs (List[str]):
            documents desired to embedd
    """
    embedding_function = HuggingFaceEmbeddings(model_name=config["hf_embedding_model"])
    vector = embedding_function.embed_query(chunk)

def embed_docs_to_faiss(docs: List) -> None:
    """
    Embeds the documents that it is fed using the 
    BioBERT model. This model is being ran locally 
    and is optimized for biomedical articles

    Args:
        docs (List[str]): list of documents that 
            are going to be embedded. The information 
            is saved in a FAISS datastore and saved 
            locally after embedding
    
    Returns:
        None
    """
    embedding_function = HuggingFaceEmbeddings(model_name=config["hf_embedding_model"])
    vs_name = "../local_embeddings/hf_faiss_pmc"

    vector_store = FAISS.from_documents(docs, embedding=embedding_function)
    vector_store.save_local(vs_name)
