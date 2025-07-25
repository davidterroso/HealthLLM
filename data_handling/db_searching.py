import json
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import anthropic
from langchain_core.prompts import PromptTemplate
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

        if config["local_or_api_embedding"] == "local":
                embedding_function = HuggingFaceEmbeddings(model_name=config["local_embedding_model"])
                vs_name = "../local_embeddings/hf_faiss_pmc"
        elif config["local_or_api_embedding"] == "api":
                # In this case, the embeddings are performed using the OpenAI API
                # The selected model is lightest embedding model from this API
                # Other examples:
                # text-embedding-3-large
                # text-embedding-ada-002
                embedding_function = OpenAIEmbeddings(model=["openai_embedding_model"])
                vs_name = "../local_embeddings/openai_faiss_pmc"

        # Dangerous deserialization is on because the data is local
        vector_store = FAISS.load_local(vs_name, 
                                        embeddings=embedding_function, 
                                        allow_dangerous_deserialization=True)

        results = vector_store.similarity_search(query=query, k=3)

        return results

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def answer_questions(query: str):
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
        # Gets the model depending on the configurations 
        # between local or through an API
        if config["local_or_api_llm"] == "local":
                tokenizer = AutoTokenizer.from_pretrained(config["local_llm"])
                model = AutoModelForSeq2SeqLM.from_pretrained(config["local_llm"])
                model_name = config["local_embedding_model"]
                embedding_function = HuggingFaceEmbeddings(model_name=config["local_embedding_model"])
                vs_name = "../local_embeddings/hf_faiss_pmc"

                pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
                llm = HuggingFacePipeline(pipeline=pipe)
        elif config["local_or_api_llm"] == "api":
                embedding_function = OpenAIEmbeddings(model=["openai_embedding_model"])
                vs_name = "../local_embeddings/openai_faiss_pmc"
                llm = anthropic(model=config["anthropic_llm"], temperature=0, max_tokens=512)

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
        print("\n Answer:", result)
        return result
        
query = "foot shape differences in diabetic patients"
answer_questions(query=query)