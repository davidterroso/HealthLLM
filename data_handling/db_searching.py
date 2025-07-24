import json
from langchain import hub
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List

with open("config.json", "r") as f:
        config = json.load(f)

def search_by_query(query: str) -> List[Document]:
        """
        Searches the given query in the VectorDB, 
        embedding it previously

        Args:
                query (str): query used to search 
                        the document 
        
        Returns:
                results (List[Document]): list of 
                        documents retrieved
        """
        embedding_function = HuggingFaceEmbeddings(model_name=config["local_embedding_model"])
        # Dangerous deserialization is on because the data is local
        vector_store = FAISS.load_local("../local_embeddings/faiss_pmc", 
                                        embeddings=embedding_function, 
                                        allow_dangerous_deserialization=True)

        results = vector_store.similarity_search(query=query, k=3)

        return results

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def answer_questions(query: str):
        """
        """
        tokenizer = AutoTokenizer.from_pretrained(config["local_llm"])
        model = AutoModelForSeq2SeqLM.from_pretrained(config["local_llm"])

        embedding_function = HuggingFaceEmbeddings(model_name=config["local_embedding_model"])
        # Dangerous deserialization is on because the data is local
        vector_store = FAISS.load_local("../local_embeddings/faiss_pmc", 
                                        embeddings=embedding_function, 
                                        allow_dangerous_deserialization=True)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(pipeline=pipe)

        prompt = hub.pull("rlm/rag-prompt")

        qa_chain = (
                {       
                        "context": vector_store.as_retriever() | format_docs,
                        "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
        )
        result = qa_chain.invoke(query)
        print("\n💬 Answer:", result)
        return result
        
query = "foot shape differences in diabetic patients"
answer_questions(query=query)