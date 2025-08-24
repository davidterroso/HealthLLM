"""
This file is used to fetch the articles from the PMC website, 
through the website, or to handle the bulk data downloaded
"""

from typing import BinaryIO, Dict, IO, List, Optional, Tuple, Union
from io import BytesIO
import os
import json
import tarfile
import uuid
import logging
from lxml import etree
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from data_handling.embedding_functions import embed_docs
from data_handling.upload_to_vectordb import initiate_qdrant_session
from utils.get_embeddings_dims import get_embeddings_dims
from utils.logging_config import setup_logging
from utils.load_config import load_config

config = load_config()

def data_pipeline(collection_name: str, tar_file_dir: str) -> None:
    """
    This initiates the data extraction pipeline and initiates the 
    Qdrant Session so that it can be called prior to the data 
    uploading

    Args:
        collection_name (str): name of the Qdrant collection
        tar_file_dir (str): directory where the .tar.gz file 
            is located

    Returns:
        None
    """
    setup_logging(level_str=config.get("logging_level", "info"))
    logging.info("Getting embeddings dimensions.")
    get_embeddings_dims()
    logging.info("Initiating Qdrant session.")
    client = initiate_qdrant_session(collection_name=collection_name)
    logging.info("Initiating extraction.")
    iterate_tar(client=client,
                collection_name=collection_name,
                tar_file_dir=tar_file_dir)

def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo,
                        processed_files: set) -> Optional[IO]:
    """
    Handles the extraction of the tar file safely, handling its 
    errors gracefully

    Args:
        tar (TarFile Object): compressed tar file that is being handled
        member (TarInfo Object): one of the files that is in the 
            compressed tar folder
        processed_files (set): set of files that have already been 
            processed

    Returns:
        (Optional[IO]): XML file in binary or None
    """
    if not member.name.lower().endswith(".xml"):
        return None

    if member.isdir():
        return None

    if member.issym() or member.islnk():
        logging.warning("Skipping symbolic link in tar: %s", member.name)
        return None

    if member.name in processed_files:
        return None

    return tar.extractfile(member)

def process_xml_member(fileobj: IO,
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

        doc_id = metadata.get("pmid") if metadata else None

        if not doc_id:
            logging.warning("Skipping %s: no PMID found", member_name)
            return

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_0"))

        existing = client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )

        if not existing:
            if text is not None:
                chunks = text_chunker(text, metadata)
                embed_docs(chunks, client, collection_name)
            else:
                logging.warning("No text extracted from %s", member_name)
        else:
            logging.info("Skipping embedded document.")

    except etree.XMLSyntaxError as e: # pylint: disable=c-extension-no-member
        logging.error("XML parsing error in %s: %s", member_name, e)
    except UnicodeDecodeError as e:
        logging.error("Encoding error in %s: %s", member_name, e)
    except ValueError as e:
        logging.error("Data error in %s: %s", member_name, e)
    finally:
        fileobj.close()

def load_checkpoint() -> set:
    """
    Returns a set of the XML files that have already been iterated

    Args:
        None

    Returns:
        processed_files (set): set of the files that have already 
            been processed
    """
    if os.path.exists(config["checkpoints_path"]):
        with open(config["checkpoints_path"], "r", encoding="utf-8") as checkpoint_file:
            return set(json.load(checkpoint_file))
    return set()

def save_checkpoint(processed_files: set) -> None:
    """
    Saves the set of XML files that have already been iterated

    Args:
        processed_files (set): set of the files that have already 
            been processed

    Returns:
        None
    """
    with open(config["checkpoints_path"], "w", encoding="utf-8") as checkpoint_file:
        json.dump(list(processed_files), checkpoint_file)

def iterate_tar(client: QdrantClient,
                collection_name: str,
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
    processed_files = load_checkpoint()
    os.makedirs(os.path.dirname(config["checkpoints_path"]), exist_ok=True)

    try:
        with tarfile.open(tar_file_dir, "r:gz") as tar:
            members = tar.getmembers()

            for i, member in enumerate(members):
                try:
                    fileobj = safe_extract_member(tar=tar,
                                                  member=member,
                                                  processed_files=processed_files)
                except ValueError as e:
                    logging.error(e)
                    continue

                if fileobj:
                    process_xml_member(fileobj=fileobj,
                                       member_name=member.name,
                                       client=client,
                                       collection_name=collection_name)
                    processed_files.add(member.name)
                    save_checkpoint(processed_files=processed_files)
                    logging.info("Uploaded %s: %d / %d", member.name, i, len(members))
                else:
                    logging.info("Skipping embedded document: %s", member.name)

    except (FileNotFoundError, PermissionError, EOFError,
            tarfile.ReadError, tarfile.TarError, OSError) as e:
        logging.error("Extraction failed: %s", e)

def extract_from_xml(xml_source: Union[str, BinaryIO],
                     file_name: str) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
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
        tree = etree.parse(xml_source) # pylint: disable=c-extension-no-member

        results = tree.xpath("//body//text()")
        if not isinstance(results, list):
            results = [results]
        text = " ".join(map(str, results)).strip()

        title_el = tree.find('.//article-title')
        title = ''.join(title_el.itertext()) if title_el is not None else None
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

    except (etree.XMLSyntaxError, OSError, IOError, PermissionError) as e: # pylint: disable=c-extension-no-member
        logging.error("XML parsing error in %s: %s - %s", xml_source, type(e).__name__, e)
        return None, {}

def text_chunker(full_text: str, metadata: Optional[dict] = None) -> List[Document]:
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
    if not isinstance(full_text, str):
        raise TypeError("full_text must be a string")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None")

    if not full_text.strip():
        return []

    chunker = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"],
                                             chunk_overlap=config["chunk_overlap"])
    chunks = chunker.split_text(full_text)

    docs = [Document(page_content=chunk, metadata=metadata or {}) for chunk in chunks]

    return docs
