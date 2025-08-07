"""
This file is used to fetch the articles from the PMC website, 
through the website, or to handle the bulk data downloaded
"""

from pathlib import Path
from typing import Dict, List
import os
import tarfile
import requests
from tqdm import tqdm
from lxml import etree
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from data_handling.embedding_functions import embed_docs, embed_docs_to_faiss
from data_handling.upload_to_vectordb import initiate_qdrant_session
from utils.logging_config import setup_logging

def data_pipeline(collection_name: str) -> None:
    """
    This initiates the data extraction pipeline 
    and initiates the Qdrant Session so that it 
    can be called prior to the data uploading

    Args:
        collection_name (str): name of the 
            Qdrant collection

    Returns:
        None
    """
    setup_logging()
    client = initiate_qdrant_session(collection_name=collection_name)
    extract_tar(extract_dir="extracted", tar_file_dir="data.tar.gz")
    iterate_xml_files(xml_dir="extracted", client=client, collection_name=collection_name)

def extract_tar(extract_dir: str, tar_file_dir: str) -> None:
    """
    Extracts the information from the .tar.gz file
    and stores it in a given directory

    Args:
        extract_dir (str): directory in which the 
            extracted information will be stored
        tar_file_dir (str): directory of the file 
            desired to extract

    Returns:
        None 
    """
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_file_dir, "r:gz") as tar:
        tar.extractall(extract_dir)

def iterate_xml_files(xml_dir: str,
                      client: QdrantClient,
                      collection_name: str) -> None:
    """
    Iterates through the XML files extracted 
    from the .tar.gz file, getting their metadata, 
    their content, embedding the content, and 
    uploading it into the VectorDB 

    Args:
        xml_dir (str): directory where the 
            extracted XML files are located
        client (QdrantClient Object): initiated 
            QdrantClient object
        collection_name (str): name of the 
            Qdrant collection

    Returns:
        None
    """
    files = list(Path(xml_dir).rglob("*.xml"))

    for xml_file in tqdm(files, desc="Processing XML Files"):
        text, metadata = extract_from_xml(str(xml_file))

        chunks = text_chunker(text, metadata)
        embed_docs(chunks, client, collection_name)

def extract_from_xml(xml_dir: str) -> tuple[str|None, Dict[str, str]|Dict[None]]:
    """
    Used in the extraction of relevant information 
    from the selected XML file

    Args:
        text (str): directory for the XML file

    Returns:
        text (str): the text in the XML file
        metadata (Dict[str, str]): the metadata
            from the article
    """
    try:
        tree = etree.parse(xml_dir)
        text = ' '.join(tree.xpath('//body//text()')).strip()

        title = tree.findtext('.//article-title')
        journal = tree.findtext('.//journal-title')
        year = tree.findtext('.//pub-date/year')
        doi = tree.findtext('.//article-id[@pub-id-type="doi"]')
        pmid = tree.findtext('.//article-id[@pub-id-type="pmid"]')

        metadata = {
            "file": os.path.basename(xml_dir),
            "title": title,
            "journal": journal,
            "year": year,
            "doi": doi,
            "pmid": pmid
        }

        return text, metadata

    except (etree.XMLSyntaxError, OSError, IOError, PermissionError) as e:
        print(f"Error parsing {xml_dir}: {type(e).__name__} - {e}")
        return None, {}

def fetch_pmc_articles_by_query(query: str, page_size: int=25) -> List[Dict[str, str]]:
    """
    Given a query, searches the open-access
    articles in the PMC website. This function
    is more oriented for testing, since it only
    allows the retrival of data using a query

    Args:
        query (str): search query in the PMC
            website
        page_size (int): number of articles
            per page

    Returns:
        results (List[dict{str}]): results of 
            the search query. Returns a 
            dictionary that stores the title, 
            abstract, link, and PMID
    """

    # URL where we can access the articles published in PMC
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    # Parameters regarding which text will be retrieved
    params = {
            "query": f"{query} AND OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": page_size,
            "resultType": "core"
    }

    response = requests.get(url=url, params=params, timeout=5)
    data = response.json()
    articles = data.get("resultList", {}).get("result", [])

    results = []

    # Iterate through the articles in the page
    for article in articles:
        entry = {
            "title": article.get("title"),
            "abstract": article.get("abstractText"),
            "source": article.get("fullTextUrlList", {}).get("fullTextUrl", []),
            "pmid": article.get("id")
        }

        results.append(entry)

    text_chunker_faiss(results=results)

def text_chunker(full_text: str, metadata: None):
    """
    Splits a large string in multiple chunks, preparing them for the 
    embedding

    Args:
        full_text (str): large string that contains all the text in 
            a file, to be split in multiple chunks

    Returns:
        docs (List[Documents]): list of LangChain Documents from the
            given XML file, which contain both the text and their 
            metadata
    """
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split_text(full_text)

    docs = []
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata=metadata or {}))

    return docs

def text_chunker_faiss(results: dict):
    """
    Function used to chunk the text
    present in the results that it
    is fed

    Args:
        results (List[dict{str}]):
        results of the search query.
        Represented in a dictionary
        that stores the title,
        abstract, link, and PMID
    """

    # Chunks the text so that it can be ingested into a vector database
    chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = chunker.create_documents([result["abstract"] for result in results \
        if result.get("abstract")])
    embed_docs_to_faiss(docs)
