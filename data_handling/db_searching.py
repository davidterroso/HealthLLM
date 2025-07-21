from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def search_by_query(query: str):
        embedding_function = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        vector_store = FAISS.load_local("../embedding_results/faiss_pmc", embeddings=embedding_function, allow_dangerous_deserialization=True)

        results = vector_store.similarity_search(query=query, k=3)

        for i, doc in enumerate(results):
                print(f"\n--- Result {i+1} ---\n")
                print(doc.page_content)

query = "foot shape differences in diabetic patients"
search_by_query(query=query)