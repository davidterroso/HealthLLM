"""
This file is used to fetch the articles from the PMC website, 
through the website, or to handle the bulk data downloaded
"""

from typing import BinaryIO, Dict, List, Optional, Tuple, Union
from io import BytesIO
import os
import logging
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

def data_pipeline(collection_name: str, extract_dir: str,
                  tar_file_dir: str) -> None:
    """
    This initiates the data extraction pipeline and initiates the 
    Qdrant Session so that it can be called prior to the data 
    uploading

    Args:
        collection_name (str): name of the Qdrant collection
        extract_dir (str): directory where the XML files are 
            going to be extracted to
        tar_file_dir (str): directory where the .tar.gz file 
            is located

    Returns:
        None
    """
    setup_logging()
    client = initiate_qdrant_session(collection_name=collection_name)
    iterate_tar(client=client,
                extract_dir=extract_dir,
                collection_name=collection_name,
                tar_file_dir=tar_file_dir)

def ensure_directory_creation(dir_name: str) -> None:
    """
    Used to safely create folders where data is going to be located

    Args:
        dir_name (str): name of the directory desired to create

    Returns:
        None
    """
    try:
        os.makedirs(dir_name, exist_ok=True)
    except PermissionError:
        logging.error("Permission denied creating repository: %s", dir_name)
        raise
    except OSError as e:
        logging.error("OS error creating directory %s: %s", dir_name, e)
        raise

def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo,
                        base_dir: str) -> BinaryIO:
    """
    Handles the extraction of the tar file safely, handling its 
    errors gracefully

    Args:
        tar (TarFile Object): compressed tar file that is being handled
        member (TarInfo Object): one of the files that is in the 
            compressed tar folder 

    Returns:
        (Binary IO): XML file in binary
    """
    if not member.name.lower().endswith(".xml"):
        return None

    if member.isdir():
        return None

    if member.issym() or member.islnk():
        logging.warning("Skipping symbolic link in tar: %s", member.name)
        return None

    member_path = os.path.abspath(os.path.join(base_dir, member.name))
    if not member_path.startswith(os.path.abspath(base_dir) + os.sep) \
        and member.isfile():
        raise ValueError(f"Unsafe file path detected: {member.name}")

    return tar.extractfile(member)

def process_xml_member(fileobj: BinaryIO,
                       member_name: str,
                       client: QdrantClient,
                       collection_name: str) -> None:
    """
    Receives the binary information of a XML file, reads it and extracts 
    the relevant informations from it, calling the chunking and embedding 
    functions subsequently

    Args:
        fileobj (BinaryIO): XML file in binary
        member_name (str): name of the XML file
        client (QdrantClient): initialized QdrantClient Object
        collection_name (str): name of the collection

    Returns:
        None
    """
    try:
        file_content = fileobj.read()
        text, metadata = extract_from_xml(BytesIO(file_content), member_name)
        chunks = text_chunker(text, metadata)
        embed_docs(chunks, client, collection_name)
    except etree.XMLSyntaxError as e:
        logging.error("XML parsing error in %s: %s", member_name, e)
    except UnicodeDecodeError as e:
        logging.error("Encoding error in %s: %s", member_name, e)
    except ValueError as e:
        logging.error("Data error in %s: %s", member_name, e)
    finally:
        fileobj.close()

def iterate_tar(client: QdrantClient,
                collection_name: str,
                extract_dir: str,
                tar_file_dir: str) -> None:
    """
    Iterates through the XML files in the downloaded .tar.gz file

    Args:
        client (QdrantClient Object): initiated QdrantClient object
        collection_name (str): name of the Qdrant collection
        extract_dir (str): directory where the XML file is extracted to
        tar_file_dir (str): directory where the .tar.gz is located

    Returns:
        None
    """
    ensure_directory_creation(dir_name=extract_dir)

    try:
        with tarfile.open(tar_file_dir, "r:gz") as tar:
            for member in tqdm(tar, desc="Processing files", unit="file"):
                try:
                    fileobj = safe_extract_member(tar=tar,
                                                  member=member,
                                                  base_dir=extract_dir)
                except ValueError as e:
                    logging.error(e)
                    continue

                if fileobj:
                    process_xml_member(fileobj,
                                       member.name,
                                       client,
                                       collection_name)

    except (FileNotFoundError, PermissionError, EOFError,
            tarfile.ReadError, tarfile.TarError, OSError) as e:
        logging.error("Extraction failed: %s", e)

def extract_from_xml(xml_source: Union[str, BinaryIO],
                     file_name: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Used in the extraction of relevant information from the selected XML file

    Args:
        xml_source (str | BinaryIO): path to the XML file or a binary stream
        file_name (str): original file name

    Returns:
        text (str | None): the text in the XML file
        metadata (Dict[str, str]): the metadata from the article
    """
    try:
        tree = etree.parse(xml_source)
        text = ' '.join(tree.xpath('//body//text()')).strip()

        title = tree.findtext('.//article-title')
        journal = tree.findtext('.//journal-title')
        year = tree.findtext('.//pub-date/year')
        doi = tree.findtext('.//article-id[@pub-id-type="doi"]')
        pmid = tree.findtext('.//article-id[@pub-id-type="pmid"]')

        metadata = {
            "file": file_name,
            "title": title,
            "journal": journal,
            "year": year,
            "doi": doi,
            "pmid": pmid
        }

        return text, metadata

    except (etree.XMLSyntaxError, OSError, IOError, PermissionError) as e:
        logging.error("XML parsing error in %s: %s - %s", xml_source, type(e).__name__, e)
        return None, {}

def fetch_pmc_articles_by_query(query: str,
                                page_size: int=25) -> List[Dict[str, str]]:
    """
    Given a query, searches the open-access articles in the PMC website. 
    This function is more oriented for testing, since it only allows the 
    retrival of data using a query

    Args:
        query (str): search query in the PMC website
        page_size (int): number of articles per page

    Returns:
        results (List[dict{str}]): results of the search query. Returns a 
            dictionary that stores the title, abstract, link, and PMID
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
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000,
                                             chunk_overlap=100)
    chunks = chunker.split_text(full_text)

    docs = []
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata=metadata or {}))

    return docs

def text_chunker_faiss(results: dict) -> None:
    """
    Function used to chunk the text present in the results that it is fed

    Args:
        results (List[dict{str}]): results of the search query. Represented 
        in a dictionary that stores the title, abstract, link, and PMID

    Returns:
        None
    """

    chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = chunker.create_documents([result["abstract"] for result in results\
        if result.get("abstract")])
    embed_docs_to_faiss(docs)
