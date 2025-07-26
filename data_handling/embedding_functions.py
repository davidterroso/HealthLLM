import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

with open("config.json", "r") as f:
        config = json.load(f)

def embed_docs(docs: List):
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
        if config["local_or_api_embedding"] == "local":
                embedding_function = HuggingFaceEmbeddings(model_name=config["local_embedding_model"])
                vs_name = "../local_embeddings/hf_faiss_pmc"
        elif config["local_or_api_embedding"] == "api":
                # In this case, the embeddings are performed using the OpenAI API
                # The selected model is lightest embedding model from this API
                # Other examples:
                # text-embedding-3-large
                # text-embedding-ada-002
                embedding_function = OpenAIEmbeddings(model=["openai_embedding_model"], 
                                                      openai_api_key=openai_api_key)
                vs_name = "../local_embeddings/openai_faiss_pmc"
        vector_store = FAISS.from_documents(docs, embedding=embedding_function)
        vector_store.save_local(vs_name)
