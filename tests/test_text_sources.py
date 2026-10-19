"""Plain-text evidence library: format gate, storage, search, locate."""

import pytest

from clap.core.text_sources import TextLibrary


def test_accepts_markdown_and_gb18030_text(tmp_path):
    lib = TextLibrary([str(tmp_path)])
    a = lib.add_bytes("notes.md", "量子力学研究波函数。".encode())
    assert a["name"] == "notes.md" and a["duplicate"] is False
    b = lib.add_bytes("gbk.txt", "波函数的模方是概率密度。".encode("gb18030"))
    assert b["name"] == "gbk.txt"
    names = [f["name"] for f in lib.list_files()]
    assert names == ["gbk.txt", "notes.md"]


@pytest.mark.parametrize("name", ["paper.pdf", "doc.docx", "sheet.xlsx", "image.png", "archive.zip"])
def test_rejects_non_text_formats(tmp_path, name):
    lib = TextLibrary([str(tmp_path)])
    with pytest.raises(ValueError):
        lib.add_bytes(name, b"whatever")


def test_rejects_binary_content_even_with_text_extension(tmp_path):
    lib = TextLibrary([str(tmp_path)])
    with pytest.raises(ValueError):
        lib.add_bytes("fake.txt", b"\x00\x01\x02binary")


def test_identical_upload_is_deduplicated(tmp_path):
    lib = TextLibrary([str(tmp_path)])
    data = "重复内容。".encode()
    first = lib.add_bytes("a.txt", data)
    second = lib.add_bytes("a.txt", data)
    assert second["duplicate"] is True
    assert first["name"] == second["name"]
    assert len(lib.list_files()) == 1


def test_name_collision_gets_suffix(tmp_path):
    lib = TextLibrary([str(tmp_path)])
    lib.add_bytes("a.txt", "第一份".encode())
    rec = lib.add_bytes("a.txt", "第二份".encode())
    assert rec["name"] != "a.txt" and rec["name"].endswith(".txt")
    assert len(lib.list_files()) == 2


def test_search_finds_paragraph_and_article_locates_it(tmp_path):
    lib = TextLibrary([str(tmp_path)])
    body = (
        "前言\n\n"
        "第一章：量子力学研究波函数，波函数的模方是概率密度。\n\n"
        "第二章：与主题无关的内容。\n"
    )
    lib.add_bytes("lesson.md", body.encode())
    hits = lib.search("波函数的模方是什么")
    assert hits and hits[0].name == "lesson.md"
    assert "量子力学" in hits[0].snippet

    art = lib.article("lesson.md", "波函数的模方")
    assert art["kind"] == "text"
    match = art["match"]
    assert match and "概率密度" in art["text"][match["start"]:match["end"]]


def test_add_path_directory_skips_other_formats(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("文本一。", encoding="utf-8")
    (src / "b.md").write_text("文本二。", encoding="utf-8")
    (src / "c.pdf").write_bytes(b"%PDF-1.4")
    lib = TextLibrary([str(tmp_path / "lib")])
    rec = lib.add_path(str(src))
    assert rec["count"] == 2
    assert rec["skipped"] == ["c.pdf"]
    assert sorted(f["name"] for f in lib.list_files()) == ["a.txt", "b.md"]


def test_delete_removes_file(tmp_path):
    lib = TextLibrary([str(tmp_path)])
    lib.add_bytes("gone.txt", b"bye")
    assert lib.delete("gone.txt") is True
    assert lib.list_files() == []
    assert lib.delete("gone.txt") is False
