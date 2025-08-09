import io
import tarfile
from data_handling.get_data import safe_extract_member

def create_test_tar(tmp_path):
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

def test_safe_extract_member_xml_only(tmp_path):
    tar_path = create_test_tar(tmp_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            fobj = safe_extract_member(tar, member, str(tmp_path))
            if member.name.endswith(".xml"):
                assert isinstance(fobj, io.BufferedReader)
            else:
                assert fobj is None
