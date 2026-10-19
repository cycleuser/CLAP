"""ZIM retrieval, grounded audit, export and renewal (unit + real archives).

The real-archive tests use the user's `~/Documents/重点文献/` bundle when it
exists and are skipped otherwise, so the suite stays portable.
"""

import glob
import json
from pathlib import Path

import pytest

from clap.core import export as EXP
from clap.core import grounded as GR
from clap.core import renew as RN
from clap.core import session as SM
from clap.core import zim_kb

ZIM_ROOT = Path.home() / "Documents" / "重点文献"
HAS_ZIMS = len(glob.glob(str(ZIM_ROOT / "*.zim"))) >= 1


# ---------------------------------------------------------------- real ZIM


@pytest.mark.skipif(not HAS_ZIMS, reason="需要 ~/Documents/重点文献/*.zim")
def test_real_library_lists_archives():
    lib = zim_kb.ZimLibrary([str(ZIM_ROOT)])
    zims = lib.list_zims()
    assert len(zims) >= 5
    ok = [z for z in zims if not z.error]
    assert len(ok) >= 5
    assert all(z.entries > 0 for z in ok)
    assert all(z.title for z in ok)


@pytest.mark.skipif(not HAS_ZIMS, reason="需要 ~/Documents/重点文献/*.zim")
def test_real_chinese_search_returns_relevant_article():
    lib = zim_kb.ZimLibrary([str(ZIM_ROOT)])
    hits = lib.search("量子", top_k=3)
    assert hits, "中文检索应有命中"
    assert any("量子" in (h.title + h.snippet) for h in hits)
    assert hits[0].snippet


@pytest.mark.skipif(not HAS_ZIMS, reason="需要 ~/Documents/重点文献/*.zim")
def test_real_english_search_and_article_read():
    lib = zim_kb.ZimLibrary([str(ZIM_ROOT)])
    hits = lib.search("transformer", zim_names=["AAAI2023-2025.zim"], top_k=2)
    assert hits, "英文检索应有命中"
    art = lib.article(hits[0].zim, hits[0].path)
    assert art["chars"] > 200 and art["text"].strip()
    assert "<" not in art["text"][:200]  # HTML stripped


@pytest.mark.skipif(not HAS_ZIMS, reason="需要 ~/Documents/重点文献/*.zim")
def test_real_article_shorthand_does_not_crash():
    lib = zim_kb.ZimLibrary([str(ZIM_ROOT)])
    with pytest.raises(KeyError):
        lib.article("物理资料MD.zim", "no/such/article.html")


def test_html_to_text_keeps_blocks_and_drops_tags():
    raw = "<html><head><style>p{}</style></head><body><h1>标题</h1><p>第一段 &amp; 实体</p><p>第二段</p></body></html>"
    text = zim_kb.html_to_text(raw)
    assert "标题" in text and "第一段 & 实体" in text and "第二段" in text
    assert "<" not in text and "p{}" not in text


# ---------------------------------------------------------------- grounded


class Hit:
    def __init__(self, n=1, text="量子力学研究波函数。", title="讲稿", path="a/b.html", zim="物理资料MD.zim"):
        self.zim = zim
        self.path = path
        self.title = title
        self.text = text
        self.snippet = text[:20]
        self.score = 10.0


def test_build_context_numbers_sources():
    sources, block = GR.build_context([Hit(1), Hit(2, text="薛定谔方程描述演化。")])
    assert [s.n for s in sources] == [1, 2]
    assert "[1]" in block and "[2]" in block and "物理资料MD.zim" in block


def test_render_system_prompt_restricts_to_material():
    sources, _ = GR.build_context([Hit()])
    prompt = GR.render_system_prompt(sources)
    assert "只使用下方编号材料回答" in prompt
    assert "来源未提及" in prompt
    assert "用与用户提问相同的语言回答" in prompt
    assert "[1] 讲稿" in prompt


def test_audit_grounded_sentence():
    sources, _ = GR.build_context([Hit()])
    result = GR.audit("量子力学研究波函数[1]。", sources)
    assert result["sentences_total"] == 1
    assert result["sentences"][0]["grounded"] is True
    assert result["score"] == 1.0


def test_audit_catches_uncited_and_unsupported():
    sources, _ = GR.build_context([Hit()])
    answer = "量子力学研究波函数[1]。紫罗兰在火星上开花结果。爱因斯坦是钢琴家[1]。"
    result = GR.audit(answer, sources)
    rows = result["sentences"]
    assert len(rows) == 3
    assert rows[0]["grounded"] is True
    assert rows[1]["grounded"] is False and "未标注来源" in rows[1]["reason"]
    assert rows[2]["grounded"] is False
    assert result["score"] == pytest.approx(1 / 3, abs=0.01)


def test_audit_flags_unknown_citation_number():
    sources, _ = GR.build_context([Hit()])
    result = GR.audit("量子力学研究波函数[9]。", sources)
    row = result["sentences"][0]
    assert row["grounded"] is False and "不存在的编号" in row["reason"]


def test_audit_accepts_honest_abstention():
    sources, _ = GR.build_context([Hit()])
    result = GR.audit("来源未提及。", sources)
    assert result["score"] == 1.0


def test_audit_marks_cross_lingual_unverified():
    """Chinese paraphrase of an English paper with no shared entity: unverifiable."""
    src = GR.Source(
        n=1,
        zim="AAAI2023-2025.zim",
        path="p.html",
        title="CAR-Transformer",
        text=(
            "Most existing methods adopt multi-task training and knowledge distillation, "
            "which increases training cost. CAR-Transformer uses a Cross-Attention "
            "Reinforcement module without extra parameters."
        ),
    )
    result = GR.audit("这些方法采用多任务训练和知识蒸馏，增加了训练成本[1]。", [src])
    row = result["sentences"][0]
    assert row["verified"] is False and row["grounded"] is False
    assert "跨语言" in row["reason"]
    assert result["sentences_unverified"] == 1
    assert result["sentences_total"] == 0  # excluded from the denominator


def test_audit_cross_lingual_entity_anchor_is_grounded():
    src = GR.Source(
        n=1,
        zim="AAAI2023-2025.zim",
        path="p.html",
        title="CAR-Transformer",
        text="CAR-Transformer uses a Cross-Attention Reinforcement module without extra parameters.",
    )
    result = GR.audit("CAR-Transformer 使用交叉注意力强化模块[1]。", [src])
    assert result["sentences"][0]["grounded"] is True


def test_cited_sources_only_lists_used():
    sources, _ = GR.build_context([Hit(1), Hit(2, text="薛定谔方程描述演化。")])
    used = GR.cited_sources("量子力学研究波函数[1]。", sources)
    assert [s["n"] for s in used] == [1]


# ---------------------------------------------------------------- export


def _demo_session():
    s = SM.new_session(chat_model="m", language="zh")
    SM.ensure_main_branch(s)
    m1 = SM.append_message(s, "user", "量子力学研究什么？", parent=None, branch_id="b1")
    m2 = SM.append_message(
        s,
        "assistant",
        "量子力学研究波函数[1]。",
        parent=m1["id"],
        branch_id="b1",
        model="m",
        provider="fake",
        latency_ms=12,
        tokens={"in": 5, "out": 6},
        sources=[{"n": 1, "zim": "物理资料MD.zim", "path": "a/b.html", "title": "讲稿"}],
        audit={"score": 1.0, "sentences_total": 1, "sentences_grounded": 1},
    )
    SM.add_annotation(s, m2["id"], "keypoint", tags=["波函数"], note="核心", author="teacher")
    return s


def test_export_markdown_includes_provenance_annotations_citations():
    md = EXP.export_markdown(_demo_session())
    assert "# CLAP 会话" in md
    assert "量子力学研究什么" in md
    assert "[1] 讲稿" in md and "物理资料MD.zim" in md
    assert "标注［keypoint］" in md and "波函数" in md
    assert "依据度" in md


def test_export_html_is_self_contained():
    html = EXP.export_html(_demo_session())
    assert html.startswith("<!DOCTYPE html>")
    assert "prefers-color-scheme: dark" in html
    assert "http://" not in html and "https://" not in html
    assert 'class="cite"' in html


def test_export_json_roundtrip():
    s = _demo_session()
    payload = json.loads(EXP.export_json(s))
    assert payload["branch_id"] == "b1"
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]
    assert payload["annotations"][0]["kind"] == "keypoint"


def test_export_unknown_format_rejected():
    with pytest.raises(ValueError):
        EXP.export(_demo_session(), fmt="pdf")


def test_export_branch_selection():
    s = _demo_session()
    m = s["messages"][-1]
    SM.fork_branch(s, m["id"], name="追问")
    b2 = s["branches"][-1]["id"]
    extra = SM.append_message(s, "user", "分支里的问题", parent=m["id"], branch_id=b2)
    SM.append_message(s, "assistant", "分支里的回答", parent=extra["id"], branch_id=b2)
    md2 = EXP.export_markdown(s, branch_id=b2)
    assert "分支里的问题" in md2 and "分支里的回答" in md2
    # the branch export keeps the shared spine
    assert "量子力学研究什么" in md2


# ---------------------------------------------------------------- renewal


def test_renew_digest_and_history():
    s = _demo_session()
    d = RN.renewal(s, "b1")
    assert "会话续期摘要" in d["summary"]
    assert d["messages"] == 2 and d["turns"] == 1
    assert d["est_tokens_after"] > 0
    assert d["chars"] == len(d["summary"])
    assert any(h.get("event") == "renewal" for h in s["manifest"]["history"])


def test_renew_system_message_shape():
    msg = RN.system_message_for("# 摘要\n- 之前聊过量子力学")
    assert msg["role"] == "system" and "量子力学" in msg["content"]


# ---------------------------------------------------------------- session extras


def test_message_extras_persist_and_validate(tmp_path):
    s = _demo_session()
    f = tmp_path / "x.clap"
    SM.save_session(s, f)  # assert_valid inside
    back = SM.load_session(f)
    assert back["messages"][1]["sources"][0]["zim"] == "物理资料MD.zim"
    assert back["messages"][1]["audit"]["score"] == 1.0


# ---------------------------------------------------------------- markdown


def test_markdown_headings_lists_and_tables():
    from clap.core import markdown_render as MD

    html = MD.render_markdown("# 标题\n\n- 一\n- 二\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<h1>" in html and "<li>" in html and "<table>" in html and "<th>" in html


def test_markdown_inline_and_display_math_becomes_mathml():
    from clap.core import markdown_render as MD

    html = MD.render_markdown("质能关系 $E=mc^2$，以及\n\n$$\\frac{a}{b}$$\n")
    assert 'class="math math-inline"' in html
    assert 'class="math math-block"' in html
    assert "<math" in html


def test_markdown_code_block_is_highlighted_and_escaped():
    from clap.core import markdown_render as MD

    html = MD.render_markdown("```python\ndef f(x):\n    return x < 3\n```")
    assert 'class="code-block"' in html and "python" in html
    assert "&lt;" in html and "return" in html  # escaped, never executed
    assert "<script>" not in html


def test_markdown_raw_html_is_escaped():
    from clap.core import markdown_render as MD

    html = MD.render_markdown("<script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_markdown_citation_chip_carries_sentence():
    from clap.core import markdown_render as MD

    html = MD.render_markdown("量子力学研究波函数[1]。另一个事实[2,3]。")
    assert 'class="cite"' in html and 'data-n="1"' in html
    assert "量子力学研究波函数" in html  # sentence kept for locating the passage
    assert 'data-n="2,3"' in html


def test_markdown_dangerous_links_are_neutralised():
    from clap.core import markdown_render as MD

    html = MD.render_markdown("[点我](javascript:alert(1))")
    assert "javascript:" not in html.lower()


def test_locate_passage_finds_the_matching_paragraph():
    text = "前言\n\n第一章：量子力学研究波函数与概率密度。\n\n第二章：无关内容。\n\n"
    match = zim_kb.locate_passage(text, "量子力学研究波函数")
    assert match is not None
    assert "量子力学" in text[match["start"]:match["end"]]
