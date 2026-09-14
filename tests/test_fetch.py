import hashlib

import pytest

from evals.datasets import fetch


def _downloader(content: bytes):
    calls = []

    def download(url, dest, *, label=None):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return dest

    return download, calls


def _tiny(monkeypatch, expected: bytes):
    remote = fetch.RemoteFile("a.jsonl", "https://example.org/a.jsonl", hashlib.sha256(expected).hexdigest())
    monkeypatch.setitem(fetch.SETS, "tiny", fetch.GoldSet("tiny", "a test set", "MIT", "https://example.org", (remote,)))


def test_ensure_downloads_once_and_verifies_the_hash(tmp_path, monkeypatch, no_network):
    body = b'{"x": 1}\n'
    _tiny(monkeypatch, body)
    download, calls = _downloader(body)
    folder = fetch.ensure("tiny", root=tmp_path, downloader=download)
    assert (folder / "a.jsonl").read_bytes() == body and calls == ["https://example.org/a.jsonl"]
    fetch.ensure("tiny", root=tmp_path, downloader=download)
    assert len(calls) == 1  # present and verified: not downloaded again


def test_a_file_that_changed_upstream_is_deleted_and_reported(tmp_path, monkeypatch):
    _tiny(monkeypatch, b"what was pinned")
    download, _ = _downloader(b"something else")
    with pytest.raises(fetch.ChecksumMismatch, match="tiny/a.jsonl"):
        fetch.ensure("tiny", root=tmp_path, downloader=download)
    assert not (tmp_path / "tiny" / "a.jsonl").exists()


def test_unknown_set_names_the_ones_that_exist(tmp_path):
    with pytest.raises(KeyError, match="ragtruth"):
        fetch.ensure("nope", root=tmp_path)


def test_every_pinned_set_has_a_licence_a_source_and_full_hashes():
    for gold in fetch.SETS.values():
        assert gold.licence and gold.source.startswith("https://") and gold.files
        assert all(len(f.sha256) == 64 and f.url.startswith("https://") for f in gold.files)
