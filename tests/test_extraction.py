"""
File used in the testing of the functions responsible for 
extracting the data from the tar file
"""

import sys
import logging
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from lxml import etree
from pytest import LogCaptureFixture

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_handling.get_data import safe_extract_member, extract_from_xml, process_xml_member

def create_test_tar(tmp_path) -> None:
    """
    Creates a mock tar file to test the functions

    Args:
        tmp_path (str): path to the temporary tar 
            file which is going to be created
    
    Returns:
        None
    """
    tar_path = tmp_path / "test.tar.gz"
    xml_content = b"<root><title>Test</title></root>"

    with tarfile.open(tar_path, "w:gz") as tar:
        # Valid XML
        xml_file = tmp_path / "test.xml"
        xml_file.write_bytes(xml_content)
        tar.add(xml_file, arcname="test.xml")

        # Invalid non-XML
        txt_file = tmp_path / "ignore.txt"
        txt_file.write_text("Ignore me")
        tar.add(txt_file, arcname="ignore.txt")

    return tar_path

def test_safe_extract_member_xml_only(tmp_path: str) -> None:
    """
    Tests the safe extraction with the files created in 
    the previous function, passing the .xml file and 
    ignoring the .txt file

    Args:
        tmp_path (str): path to the created temporary 
            tar file

    Returns:
        None
    """
    tar_path = create_test_tar(tmp_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            fobj = safe_extract_member(tar, member, set())
            if member.name.endswith(".xml"):
                assert isinstance(fobj, io.BufferedReader)
            else:
                assert fobj is None

def test_safe_extract_member_directory(tmp_path: str) -> None:
    """
    Tests the handling other directories inside the 
    .tar file

    Args:
        tmp_path (str): path to the created temporary 
            .tar file

    Returns:
        None
    """
    tar_path = tmp_path / "dir_test.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        dir_info = tarfile.TarInfo(name="somedir/")
        dir_info.type = tarfile.DIRTYPE
        tar.addfile(dir_info)

    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember("somedir/")
        assert safe_extract_member(tar, member, set()) is None


def test_safe_extract_member_symlink(tmp_path: str,
                                     caplog: LogCaptureFixture) -> None:
    """
    Tests what happens when a symlink file is found 
    inside the .tar file

    Args:
        tmp_path (str): path to the created temporary 
            tar file
        caplog (LogCaptureFixture): object that 
            captures the logged information

    Returns:
        None
    """
    tar_path = tmp_path / "link_test.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        link_info = tarfile.TarInfo(name="link.xml")
        link_info.type = tarfile.SYMTYPE
        tar.addfile(link_info)

    caplog.set_level(logging.WARNING)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember("link.xml")
        result = safe_extract_member(tar, member, set())
        assert result is None
        assert "Skipping symbolic link" in caplog.text

def test_safe_extract_member_already_processed(tmp_path: str) -> None:
    """
    Tests what happens when a file that has already 
    been extracted appears in the list of files to 
    be extracted

    Args:
        tmp_path (str): path to the created temporary 
            tar file

    Returns:
        None
    """
    tar_path = create_test_tar(tmp_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember("test.xml")
        processed_files = {"test.xml"}
        assert safe_extract_member(tar, member, processed_files) is None

def create_test_xml(tmp_path: str):
    """
    Creates a mock XML file to test the functions

    Args:
        tmp_path (str): path to the temporary XML 
            file which is going to be created
    
    Returns:
        None
    """
    xml_content = b"""
    <article>
        <front>
            <journal-meta>
                <journal-title>Test Journal</journal-title>
            </journal-meta>
            <article-meta>
                <title-group>
                    <article-title>Test Article</article-title>
                </title-group>
                <pub-date>
                    <year>2024</year>
                </pub-date>
                <article-id pub-id-type="doi">10.1234/test</article-id>
                <article-id pub-id-type="pmid">987654</article-id>
            </article-meta>
        </front>
        <body>
            <p>This is some test content.</p>
        </body>
    </article>
    """
    xml_file = tmp_path / "test.xml"
    xml_file.write_bytes(xml_content)
    return xml_file

def test_extract_from_valid_xml(tmp_path):
    """
    Tests if the information is correctly extracted 
    from the mock XML file

    Args:
        tmp_path (str): path to the created temporary 
            XML file

    Returns:
        None
    """
    xml_file = create_test_xml(tmp_path=tmp_path)
    text, metadata = extract_from_xml(str(xml_file), "test.xml")

    assert "This is some test content." in text
    assert metadata["title"] == "Test Article"
    assert metadata["journal"] == "Test Journal"
    assert metadata["year"] == "2024"
    assert metadata["doi"] == "10.1234/test"
    assert metadata["pmid"] == "987654"

def test_extract_from_binary_stream() -> None:
    """
    Tests if the information is correctly extracted 
    when passing it as a binary stream

    Args:
        None

    Returns:
        None
    """
    xml_stream = io.BytesIO(b"<root><body><p>Stream content</p></body></root>")

    text, metadata = extract_from_xml(xml_stream, "stream.xml")

    assert "Stream content" in text
    assert metadata["file"] == "stream.xml"


def test_extract_from_malformed_xml(caplog):
    """
    Tests if poorly formated information is gracefully 
    handled

    Args:
        caplog (LogCaptureFixture): object that captures 
            the logged information

    Returns:
        None
    """

    bad_xml_stream = io.BytesIO(b"<root><body><p>Missing closing tags")

    with caplog.at_level(logging.ERROR):
        text, metadata = extract_from_xml(bad_xml_stream, "bad.xml")

    assert text is None
    assert not metadata
    assert any("XML parsing error" in message for message in caplog.messages)


def test_extract_from_nonexistent_file(caplog):
    """
    Tests if empty files are handled gracefully

    Args:
        caplog (LogCaptureFixture): object that captures 
            the logged information

    Returns:
        None
    """
    with caplog.at_level(logging.ERROR):
        text, metadata = extract_from_xml("nonexistent.xml", "nonexistent.xml")

    assert text is None
    assert not metadata
    assert any("XML parsing error" in message for message in caplog.messages)

@patch("data_handling.get_data.embed_docs")
@patch("data_handling.get_data.text_chunker")
@patch("data_handling.get_data.extract_from_xml")
def test_process_xml_member_correct_path(mock_extract: MagicMock,
                                         mock_chunker: MagicMock,
                                         mock_embed: MagicMock) -> None:
    """
    Tests the processing of a correct XML file

    Args:
        mock_extract (MagicMock): mock function 
            that mimics the extraction function
        mock_chunker (MagicMock): mock function 
            that mimics the chunking function
        mock_embed (MagicMock): mock function 
            that mimics the embedding function

    Returns:
        None
    """
    mock_qdrant_client = MagicMock(name="QdrantClient")
    test_fileobj = io.BytesIO(b"<root><body><p>content</p></body></root>")

    mock_extract.return_value = ("Some text", {"meta": "data"})
    mock_chunker.return_value = ["chunk1", "chunk2"]

    process_xml_member(test_fileobj,
                       "test.xml",
                       mock_qdrant_client,
                       "test_collection")

    mock_extract.assert_called_once()
    mock_chunker.assert_called_once_with("Some text", {"meta": "data"})
    mock_embed.assert_called_once_with(["chunk1", "chunk2"],
                                       mock_qdrant_client,
                                       "test_collection")
    assert test_fileobj.closed

def test_process_xml_member_xml_error(caplog: LogCaptureFixture):
    """
    Tests the processing of an incorrect XML file

    Args:
        caplog (LogCaptureFixture): object that 
            captures the logged information

    Returns:
        None
    """
    mock_qdrant_client = MagicMock(name="QdrantClient")
    test_fileobj = io.BytesIO(b"<root><body><p>content</p></body></root>")

    with patch("data_handling.get_data.extract_from_xml",
               side_effect=etree.XMLSyntaxError("msg", 0, 0, 0, "bad.xml")):
        with caplog.at_level(logging.ERROR):
            process_xml_member(test_fileobj,
                               "bad.xml",
                               mock_qdrant_client,
                               "collection")
    assert "XML parsing error" in caplog.text
    assert test_fileobj.closed


def test_process_xml_member_unicode_error(caplog: LogCaptureFixture):
    """
    Tests the processing of a XML file with an 
    unicode error

    Args:
        caplog (LogCaptureFixture): object that 
            captures the logged information

    Returns:
        None
    """
    mock_qdrant_client = MagicMock(name="QdrantClient")
    test_fileobj = io.BytesIO(b"<root><body><p>content</p></body></root>")

    with patch("data_handling.get_data.extract_from_xml",
               side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "reason")):
        with caplog.at_level(logging.ERROR):
            process_xml_member(test_fileobj,
                               "bad_encoding.xml",
                               mock_qdrant_client,
                               "collection")
    assert "Encoding error" in caplog.text
    assert test_fileobj.closed

def test_process_xml_member_value_error(caplog: LogCaptureFixture):
    """
    Tests the processing of a XML file with a 
    value error

    Args:
        caplog (LogCaptureFixture): object that 
            captures the logged information

    Returns:
        None
    """
    mock_qdrant_client = MagicMock(name="QdrantClient")
    test_fileobj = io.BytesIO(b"<root><body><p>content</p></body></root>")

    with patch("data_handling.get_data.extract_from_xml",
               side_effect=ValueError("Bad data")):
        with caplog.at_level(logging.ERROR):
            process_xml_member(test_fileobj,
                               "bad_data.xml",
                               mock_qdrant_client,
                               "collection")
    assert "Data error" in caplog.text
    assert test_fileobj.closed
