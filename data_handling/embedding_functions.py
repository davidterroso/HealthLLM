import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

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
        embedding_function = HuggingFaceEmbeddings(model_name=config["local_embedding_model"])
        vector_store = FAISS.from_documents(docs, embedding=embedding_function)
        vector_store.save_local("../local_embeddings/faiss_pmc")
    