from backend.karl_docs import KARL_VERSION, get_karl_document, get_karl_info


def test_karl_info_lists_documents() -> None:
    info = get_karl_info()
    assert info["version"] == KARL_VERSION
    assert isinstance(info["documents"], list)
    assert any(doc["id"] == "update_log" for doc in info["documents"])
    assert any(doc["id"] == "governance_charter" for doc in info["documents"])


def test_karl_document_reads_allowlisted_markdown() -> None:
    doc = get_karl_document("update_log")
    assert doc["id"] == "update_log"
    assert doc["path"] == "UPDATE_LOG.md"
    assert "v0.8.109" in doc["content"] or "v0.8.110" in doc["content"]


def test_karl_document_rejects_unknown_doc() -> None:
    try:
        get_karl_document("not_a_real_doc")
    except KeyError:
        return
    raise AssertionError("Expected unknown KARL doc to raise KeyError")


if __name__ == "__main__":
    test_karl_info_lists_documents()
    test_karl_document_reads_allowlisted_markdown()
    test_karl_document_rejects_unknown_doc()
    print("backend/test_karl_docs.py: PASS")
