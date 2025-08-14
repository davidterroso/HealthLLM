"""
Used in the comparison between the embedded queries and the documents 
in the VectorDB

THIS FILE IS OUTDATED AND CORRESPONDS TO AN OLDER VERSION OF THE CODE
"""

import json
import os
from typing import List, Union, Tuple
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

def create_embedding_function() -> Tuple[Union[HuggingFaceEmbeddings, OpenAIEmbeddings], str]:
    """
    Factory function to create the appropriate embedding function and vector store name
    
    Args:
        None

    Returns:
        embedding_function (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): function used 
            in the embedding
        vector_store_name (str): name of the vector store
    """
    if config["local_or_api_embedding"] == "local":
        embedding_func: Union[HuggingFaceEmbeddings, OpenAIEmbeddings] = HuggingFaceEmbeddings(
            model_name=config["local_embedding_model"]
        )
        vs_name = "../local_embeddings/hf_faiss_pmc"
    elif config["local_or_api_embedding"] == "api":
        embedding_func = OpenAIEmbeddings(model=config["openai_embedding_model"])
        vs_name = "../local_embeddings/openai_faiss_pmc"
    else:
        raise ValueError(f"Invalid embedding source: {config['local_or_api_embedding']}")
    return embedding_func, vs_name

def create_llm_and_embedding() -> Tuple[Union[HuggingFacePipeline, HuggingFaceEndpoint],
                                       Union[HuggingFaceEmbeddings, OpenAIEmbeddings], str]:
    """
    Factory function to create the appropriate LLM and embedding function
    
    Args:
        None

    Returns:
        llm (Union[HuggingFacePipeline, HuggingFaceEndpoint]): function used when calling 
            the LLM
        embedding_function (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): function used 
            in the embedding
        vector_store_name (str): name of the vector store
    """
    if config["local_or_api_llm"] == "local":
        tokenizer = AutoTokenizer.from_pretrained(config["local_llm"])
        model = AutoModelForSeq2SeqLM.from_pretrained(config["local_llm"])
        embedding_func: Union[HuggingFaceEmbeddings, OpenAIEmbeddings] = HuggingFaceEmbeddings(
            model_name=config["local_embedding_model"]
        )
        vs_name = "../local_embeddings/hf_faiss_pmc"

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        llm_instance: Union[HuggingFacePipeline, HuggingFaceEndpoint] = HuggingFacePipeline(pipeline=pipe)

    elif config["local_or_api_llm"] == "api":
        embedding_func = OpenAIEmbeddings(model=config["openai_embedding_model"])
        vs_name = "../local_embeddings/openai_faiss_pmc"
        llm_instance = HuggingFaceEndpoint(
            repo_id=config["mistral_llm"],
            model=config["mistral_llm"],
            model_kwargs={"temperature": 0.7, "max_new_tokens": 512},
            huggingfacehub_api_token=hf_api_key,
        )

    else:
        raise ValueError(f"Invalid LLM source: {config['local_or_api_llm']}")

    return llm_instance, embedding_func, vs_name

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
    embedding_function, vs_name = create_embedding_function()

    # Dangerous deserialization is on because the data is local
    vector_store = FAISS.load_local(vs_name,
                                    embeddings=embedding_function,
                                    allow_dangerous_deserialization=True)

    results = vector_store.similarity_search(query=query, k=3)

    return results


def format_docs(docs: List[Document]) -> str:
    """
    Function used to format the documents it receives, by 
    joining consecutive documents through paragraphs.

    Args:
        docs (List[str]): documents desired to join
    
    Returns:
        str: documents formated into a single string
    """
    return "\n\n".join(doc.page_content for doc in docs)


def answer_questions(query: str) -> str:
    """
    Receives a query, searches the most similar documents 
    in the VectorDB (FAISS) and summarizes its content. 
    All this limited by the given prompt, which can 
    obviously be tuned 

    Args:
            query (str): search query or question input by the 
                    user to retrieve the most similar documents
    
    Returns:
            answer (str): answer given by the selected LLM

    """
    llm, embedding_function, vs_name = create_llm_and_embedding()

    # Dangerous deserialization is on because the data is local
    vector_store = FAISS.load_local(vs_name,
                                    embeddings=embedding_function,
                                    allow_dangerous_deserialization=True)

    prompt = PromptTemplate.from_template(
        """
        You are a biomedical research assistant. \
        Use the context below to answer the user's\
        question in a clear, evidence-based manner.\

        Only use the provided context. Do not rely\
        on prior knowledge. If the answer cannot be\
        determined from the context, say so honestly.\

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

    qa_chain: Runnable = (
        {
            "context": vector_store.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    result = qa_chain.invoke(query)
    print("\n Answer:", result)
    return result

if __name__ == "__main__":
    QUERY = "foot shape differences in diabetic patients"
    answer_questions(query=QUERY)
