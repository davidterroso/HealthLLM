"""
File used in the testing of the functions responsible for 
extracting the data from the tar file
"""

import sys
import logging
import io
import tarfile
from pathlib import Path
from pytest import LogCaptureFixture

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_handling.get_data import safe_extract_member

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
    Verifies the data extracted from an XML file

    Args:
        tmp_path (str): path to the created temporary 
            tar file

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
