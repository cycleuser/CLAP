"""Web workspace tests: SSE chat, grounded mode, ZIM endpoints, lifecycle.

Rules kept from the W1 tests: no network beyond a loopback fake provider; the
knowledge-base singleton is redirected to a temp directory; every SSE body is
consumed explicitly (a Flask test client runs streaming generators lazily, so
a test that never reads the body never exercises the server side).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("flask")

from clap.core import providers as PR  # noqa: E402
from clap.core import session as SM  # noqa: E402
from clap.core.providers import OpenAICompatProvider  # noqa: E402
from clap.web import app as webapp  # noqa: E402


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    """SSE server: /v1/models -> one model; /v1/chat/completions -> 2 deltas + usage."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "fake-chat-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.close_connection = True

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # A grounded request carries the narrow prompt; echo a cited sentence
        # so the audit path is exercised for real.
        grounded = "编号材料" in payload.decode("utf-8", "ignore")
        text = "量子力学研究波函数[1]。" if grounded else "答"
        packets = [
            {"choices": [{"delta": {"content": text}}]},
            {"choices": [{"delta": {"content": "" if grounded else "案"}}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            "[DONE]",
        ]
        for p in packets:
            line = p if isinstance(p, str) else json.dumps(p)
            self.wfile.write(f"data: {line}\n\n".encode())
            self.wfile.flush()
        self.close_connection = True


class FakeZimLibrary:
    """Deterministic stand-in for the ZIM library (real archives in test_zim)."""

    roots = ["/fake/roots"]

    class Hit:
        def __init__(self, n=1):
            self.zim = "物理资料MD.zim"
            self.path = "5. Quantum Mechanics/第五章-量子力学-讲稿.html"
            self.title = "第五章-量子力学-讲稿"
            self.snippet = "这份讲稿整理自剑桥大学 David Tong 的量子力学讲义"
            self.text = (
                "量子力学：一份讲稿。这份讲稿整理自剑桥大学 David Tong 的量子力学讲义。"
                "内容覆盖波函数、薛定谔方程、一维问题。量子力学研究波函数。"
            )
            self.score = 21.0

        def as_dict(self, with_text=False):
            d = {"zim": self.zim, "path": self.path, "title": self.title, "snippet": self.snippet, "score": self.score}
            if with_text:
                d["text"] = self.text
            return d

    def list_zims(self):
        return []

    def search(self, query, zim_names=None, top_k=10):
        if "无材料" in query:
            return []
        return [self.Hit()]

    def article(self, zim_name, entry_path, max_chars=60000):
        return {
            "zim": zim_name,
            "path": entry_path,
            "title": "第五章-量子力学-讲稿",
            "text": self.Hit().text,
            "chars": len(self.Hit().text),
            "is_html": False,
        }

    def suggestions(self, prefix, zim_name="", limit=8):
        return ["第五章-量子力学-讲稿"]


@pytest.fixture(scope="module")
def fake_base():
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture()
def client(fake_base, tmp_path, monkeypatch):
    fake = OpenAICompatProvider("fake-web", fake_base, api_key="sk-fake-web-0000", name="fake")
    monkeypatch.setattr(webapp, "get_registry", lambda: {"fake-web": fake})
    monkeypatch.setattr(webapp, "DEFAULT_PROVIDER", "fake-web")
    # never touch the user's real settings file
    monkeypatch.setenv("CLAP_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("CLAP_TEXT_ROOTS", str(tmp_path / "texts"))
    from clap.core import text_sources as TX

    TX.reset_library()
    from clap.core.knowledge_base import KnowledgeBase

    monkeypatch.setattr(webapp, "kb", KnowledgeBase(persist_directory=str(tmp_path / "kb")))
    monkeypatch.setattr(webapp.zim_kb, "library", lambda: FakeZimLibrary())
    webapp.set_session(None)
    yield webapp.app.test_client()
    webapp.set_session(None)


def _sse(body: str) -> list[dict]:
    out = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data:"):
            out.append(json.loads(chunk[5:].strip()))
    return out


def _chat(client, body) -> list[dict]:
    """POST /api/chat and CONSUME the stream (required for generators to run)."""
    r = client.post("/api/chat", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)[:200]
    return _sse(r.get_data(as_text=True))


# ---------------------------------------------------------------- pages


@pytest.mark.parametrize("path", ["/", "/kb", "/api-docs"])
def test_pages_render_200_with_design_tokens(client, path):
    r = client.get(path)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "<style>" in html and "--blue-600" in html  # design tokens inlined
    assert 'html[data-theme="dark"]' in html  # dark palette shipped
    assert "CLAPtheme" in html and 'data-theme' in html  # theme switcher wired
    assert "有依据" in html or "ZIM" in html


@pytest.mark.parametrize("path", ["/", "/kb", "/api-docs", "/timeline"])
def test_pages_offer_language_and_theme_switches(client, path):
    """Ten interface languages and three theme modes on every page."""
    html = client.get(path).get_data(as_text=True)
    assert 'id="lang"' in html and 'id="theme"' in html
    assert "CLAPi18n" in html and "CLAPtheme" in html
    for tag in ("zh", "en", "ja", "fr", "ru", "de", "es", "pt", "it", "ko"):
        assert 'value="%s"' % tag in html, (path, tag)
    for mode in ("auto", "light", "dark"):
        assert 'value="%s"' % mode in html, (path, mode)


def test_i18n_dictionaries_cover_all_ten_languages():
    """Every language block carries the complete key set (199 entries)."""
    import pathlib
    import re

    js = (pathlib.Path(webapp.__file__).parent / "static" / "i18n.js").read_text(encoding="utf-8")
    marks = list(re.finditer(r"^    (\w+): \{\n", js, re.M))
    blocks: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else js.find("\n  };", m.end())
        blocks[m.group(1)] = js[m.end():end]
    assert set(blocks) >= {"zh", "en", "ja", "fr", "ru", "de", "es", "pt", "it", "ko"}
    base = set(re.findall(r'"([A-Za-z0-9_.]+)"\s*:', blocks["zh"]))
    assert len(base) >= 190
    for lang, body in blocks.items():
        keys = set(re.findall(r'"([A-Za-z0-9_.]+)"\s*:', body))
        assert keys == base, (lang, sorted(base - keys)[:5], sorted(keys - base)[:5])


def test_pages_have_no_external_resources(client):
    """W1+ rule: zero CDN / external font / external script."""
    for path in ("/", "/kb", "/api-docs", "/timeline"):
        html = client.get(path).get_data(as_text=True)
        for bad in ("http://", "https://", "//cdn", "<link rel=\"stylesheet\"", "src=\"/static"):
            assert bad not in html, (path, bad)


@pytest.mark.parametrize("path", ["/", "/kb", "/api-docs", "/timeline"])
def test_pages_offer_language_switch(client, path):
    """The whole interface is bilingual: a switcher + both dictionaries inline."""
    html = client.get(path).get_data(as_text=True)
    assert 'id="lang"' in html
    assert "CLAPi18n" in html
    assert "English" in html and "中文" in html
    assert 'data-i18n="nav.chat"' in html


def test_timeline_page_without_param_is_friendly(client):
    r = client.get("/timeline")
    assert r.status_code == 400
    assert "缺少 file 参数" in r.get_data(as_text=True)


def test_timeline_page_renders_a_clap_session(client, tmp_path):
    s = SM.new_session(chat_model="m", language="zh")
    SM.ensure_main_branch(s)
    m1 = SM.append_message(s, "user", "什么是兼语句？", parent=None, branch_id="b1")
    SM.append_message(s, "assistant", "兼语句是……", parent=m1["id"], branch_id="b1", model="m", latency_ms=900)
    f = tmp_path / "s.clap"
    SM.save_session(s, f)
    r = client.get("/timeline", query_string={"file": str(f)})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "什么是兼语句" in html and "兼语句是" in html


def test_timeline_page_bad_path_is_friendly(client):
    r = client.get("/timeline", query_string={"file": "/no/such/file.clap"})
    assert r.status_code == 404
    assert "无法渲染" in r.get_data(as_text=True)


# ---------------------------------------------------------------- models


def test_models_grouped_and_masked(client):
    j = client.get("/api/models").get_json()
    assert j["success"] is True
    fake = [g for g in j["groups"] if g["provider"] == "fake-web"][0]
    assert fake["available"] is True and fake["models"] == ["fake-chat-model"]
    assert "sk-fake-web-0000" not in json.dumps(j)
    assert j["default"] == {"provider": "fake-web", "model": "fake-chat-model"}
    assert isinstance(j["env_assembly"], list)


# ---------------------------------------------------------------- chat


def test_chat_streams_through_provider_layer(client):
    events = _chat(client, {"message": "测试流式"})
    assert "".join(e["delta"] for e in events if "delta" in e) == "答案"
    done = [e for e in events if e.get("done")]
    assert len(done) == 1
    meta = done[0]["meta"]
    assert meta["provider"] == "fake-web" and meta["model"] == "fake-chat-model"
    assert meta["tokens"] == {"in": 9, "out": 2}
    assert meta["error"] is None and meta["latency_ms"] >= 0


def test_chat_records_provenance_in_session(client):
    _chat(client, {"message": "第一问"})
    msgs = client.get("/api/session/get").get_json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["provider"] == "fake-web"
    assert msgs[1]["tokens"]["out"] == 2 and msgs[1]["latency_ms"] is not None


def test_chat_tools_lane_does_not_touch_main_line(client):
    _chat(client, {"message": "机器自检"})
    _chat(client, {"message": "工具探针", "lane": "tools"})
    sess = client.get("/api/session/get").get_json()
    assert [m["content"] for m in sess["messages"]] == ["机器自检", "答案"]
    names = {b["name"] for b in sess["branches"]}
    assert "tools" in names


def test_chat_rejects_empty_and_unknown_provider(client):
    assert client.post("/api/chat", json={"message": " "}).status_code == 400
    r = client.post("/api/chat", json={"message": "x", "provider": "nope"})
    assert r.status_code == 400 and "还没有配置" in r.get_json()["error"]
    r2 = client.post("/api/chat", json={"message": "x", "provider": "nope", "lang": "en"})
    assert r2.status_code == 400 and "not configured" in r2.get_json()["error"]


def test_chat_error_is_friendly_and_scrubbed(client, monkeypatch):
    class Boom(OpenAICompatProvider):
        def chat(self, messages, model, stream=True):
            raise PR.ProviderError(f"{self.id}: HTTP 401 key sk-fake-web-0000 bad")

    boom = Boom("fake-web", "http://127.0.0.1:9/v1", api_key="sk-fake-web-0000")
    monkeypatch.setattr(webapp, "get_registry", lambda: {"fake-web": boom})
    events = _chat(client, {"message": "hi", "model": "fake-chat-model"})
    meta = [e for e in events if e.get("done")][0]["meta"]
    assert "HTTP 401" in meta["error"]
    assert "sk-fake-web-0000" not in meta["error"]


def test_stop_endpoint_graceful(client):
    j = client.post("/api/chat/stop", json={"stream_id": "nope"}).get_json()
    assert j["success"] is False


# ---------------------------------------------------------------- grounded


def test_grounded_chat_carries_sources_and_audit(client):
    events = _chat(client, {"message": "量子力学研究什么", "grounded": True})
    assert any("sources" in e for e in events)
    meta = [e for e in events if e.get("done")][0]["meta"]
    assert meta["grounded"] is True
    assert meta["sources"] and meta["sources"][0]["zim"] == "物理资料MD.zim"
    assert meta["audit"]["sentences_total"] >= 1
    assert meta["audit"]["score"] >= 0.5
    # persisted with provenance for exports
    msgs = client.get("/api/session/get").get_json()["messages"]
    assert msgs[-1]["sources"] and msgs[-1]["audit"]


def test_grounded_chat_abstains_when_no_material(client):
    events = _chat(client, {"message": "无材料的问题", "grounded": True})
    text = "".join(e["delta"] for e in events if "delta" in e)
    assert "来源未提及" in text
    meta = [e for e in events if e.get("done")][0]["meta"]
    assert meta["audit"]["score"] == 1.0  # honest abstention


def test_grounded_abstention_follows_ui_language(client):
    events = _chat(client, {"message": "无材料的问题", "grounded": True, "lang": "en"})
    text = "".join(e["delta"] for e in events if "delta" in e)
    assert "does not mention" in text
    meta = [e for e in events if e.get("done")][0]["meta"]
    assert meta["audit"]["score"] == 1.0


# ---------------------------------------------------------------- lifecycle


def test_export_three_formats(client):
    _chat(client, {"message": "导出这轮"})
    for fmt in ("md", "html", "json"):
        j = client.post("/api/session/export", json={"format": fmt}).get_json()
        assert j["success"] is True and j["content"]
        assert "导出这轮" in j["content"]
    md = client.post("/api/session/export", json={"format": "md"}).get_json()["content"]
    assert "## 1. 教师" in md and "## 2. 助手" in md
    html = client.post("/api/session/export", json={"format": "html"}).get_json()["content"]
    assert "<!DOCTYPE html>" in html and "http://" not in html and "https://" not in html


def test_export_writes_file_when_path_given(client, tmp_path):
    _chat(client, {"message": "落盘"})
    out = tmp_path / "out.md"
    j = client.post("/api/session/export", json={"format": "md", "path": str(out)}).get_json()
    assert j["success"] and j["path"] == str(out) and out.exists()


def test_renew_returns_digest_and_records_history(client):
    _chat(client, {"message": "第一问"})
    _chat(client, {"message": "第二问"})
    j = client.post("/api/session/renew", json={}).get_json()
    assert j["success"] is True
    assert "会话续期摘要" in j["summary"]
    assert j["turns"] == 2 and j["messages"] == 4
    hist = client.get("/api/session/get").get_json()["history"]
    assert any(h.get("event") == "renewal" for h in hist)


def test_share_package_is_created_and_key_free(client, tmp_path):
    _chat(client, {"message": "分享这轮"})
    j = client.post("/api/session/share", json={"out_dir": str(tmp_path)}).get_json()
    assert j["success"] is True
    pkg = j["package"]
    assert pkg.endswith(".clapshare")
    data = open(pkg, "rb").read()
    assert b"sk-" not in data


def test_adopt_endpoint_delegates(client, monkeypatch):
    seen = {}

    def fake_adopt(package, policy="rename", library=None):
        seen["package"] = package
        seen["policy"] = policy
        return {"session_id": "ses_test", "path": "/tmp/x.clap"}

    monkeypatch.setattr(webapp.SH, "adopt", fake_adopt)
    j = client.post("/api/session/adopt", json={"package": "/tmp/x.clapshare", "policy": "fork"}).get_json()
    assert j["success"] is True and j["record"]["session_id"] == "ses_test"
    assert seen == {"package": "/tmp/x.clapshare", "policy": "fork"}


# ---------------------------------------------------------------- zim


def test_zim_list_endpoint(client):
    j = client.get("/api/zim/list").get_json()
    assert j["success"] is True
    assert j["roots"] == ["/fake/roots"]


def test_zim_search_endpoint(client):
    j = client.post("/api/zim/search", json={"query": "量子", "top_k": 3}).get_json()
    assert j["success"] is True and j["results"]
    hit = j["results"][0]
    assert set(hit) >= {"zim", "path", "title", "snippet", "score"}


def test_zim_article_endpoint(client):
    j = client.get(
        "/api/zim/article",
        query_string={"zim": "物理资料MD.zim", "path": "5. Quantum Mechanics/第五章-量子力学-讲稿.html"},
    ).get_json()
    assert j["success"] is True and "量子力学" in j["text"]


def test_zim_search_rejects_empty_query(client):
    assert client.post("/api/zim/search", json={"query": " "}).status_code == 400


# ---------------------------------------------------------------- session


def test_annotate_roundtrip_and_export(client, tmp_path):
    _chat(client, {"message": "标注我"})
    target = client.get("/api/session/get").get_json()["messages"][0]["id"]
    j = client.post(
        "/api/annotate",
        json={"target": target, "kind": "keypoint", "tags": ["测试"], "note": "n", "author": "teacher"},
    ).get_json()
    assert j["success"] is True and j["annotation"]["kind"] == "keypoint"

    bad = client.post("/api/annotate", json={"target": target, "kind": "alien"}).get_json()
    assert bad["success"] is False
    missing = client.post("/api/annotate", json={"kind": "keypoint"}).get_json()
    assert missing["success"] is False

    out = tmp_path / "ann.csv"
    jj = client.post("/api/annotate/export", json={"path": str(out), "format": "csv"}).get_json()
    assert jj["success"] is True and jj["rows"] == 1
    assert "keypoint" in out.read_text(encoding="utf-8")


def test_fork_save_load_roundtrip(client, tmp_path):
    _chat(client, {"message": "主线一问"})
    msgs = client.get("/api/session/get").get_json()["messages"]
    b = client.post("/api/session/fork", json={"at": msgs[-1]["id"], "name": "追问"}).get_json()
    assert b["success"] is True and b["branch"]["fork_at"] == msgs[-1]["id"]

    p = tmp_path / "web.clap"
    assert client.post("/api/session/save", json={"path": str(p)}).get_json()["success"] is True
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == "2.0"

    client.post("/api/session/new", json={})
    assert client.get("/api/session/get").get_json()["messages"] == []
    assert client.post("/api/session/load", json={"path": str(p)}).get_json()["success"] is True
    assert client.get("/api/session/get").get_json()["messages"][0]["content"] == "主线一问"


def test_load_missing_file_is_friendly(client):
    r = client.post("/api/session/load", json={"path": "/no/such.clap"})
    assert r.status_code == 400 and "无法加载" in r.get_json()["error"]


# ---------------------------------------------------------------- kb


def test_kb_contract_unchanged(client):
    assert "success" in client.post("/api/kb/index", json={}).get_json()
    assert "success" in client.post("/api/kb/search", json={"query": "函数"}).get_json()
    assert "chunks" in client.get("/api/kb/stats").get_json()
    assert client.post("/api/kb/clear").get_json().get("success") is True


# ---------------------------------------------------------------- env assembly


def test_env_assembler_maps_names_to_inventory_ids():
    probe = {
        "providers": {
            "deepseek": "https://api.deepseek.com/v1",
            "ollama-cloud": "https://ollama.com/v1",
            "minimax-cn": "https://api.minimax.cn/anthropic/v1",
        },
        "models": {"deepseek": ["deepseek-flash"]},
    }
    env = {
        "DEEPSEEK_API_KEY": "sk-test-0000000000",
        "OLLAMA_CLOUD_API_KEY": "ollama-test-00000",
        "HF_TOKEN": "hf-not-in-inventory",
        "DOC2X_API_KEY": "doc2x-000000000",
    }
    asm = PR.EnvAssembler(probe, environ=env)
    reg = asm.assemble()
    assert set(reg) == {"deepseek", "ollama-cloud"}  # HF not in inventory → ignored
    assert "DOC2X_API_KEY" not in json.dumps(asm.sources)
    payload = json.dumps(asm.sources)
    assert "sk-test-0000000000" not in payload and "ollama-test-00000" not in payload
    assert asm.sources["deepseek"]["masked"].startswith("***len")


def test_env_assembler_skips_anthropic_style_provider():
    probe = {"providers": {"minimax-cn": "https://api.minimax.cn/anthropic/v1"}, "models": {}}
    asm = PR.EnvAssembler(probe, environ={"MINIMAX_API_KEY": "mm-0000000000"})
    assert asm.assemble() == {}
    row = asm.sources["minimax-cn"]
    assert row["configured"] is True and row["unsupported"]


def test_bundled_inventory_is_present_and_keyless():
    probe = PR.load_probe_list()
    assert probe["path"] and probe["path"].endswith("opencode_models.json")
    assert len(probe["providers"]) == 13
    assert sum(len(v) for v in probe["models"].values()) == 188
    assert "sk-" not in json.dumps(probe)


def test_local_ollama_needs_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("CLAP_OLLAMA_LOCAL", raising=False)
    reg = PR.discover_providers()
    assert "ollama-local" not in reg
    assert all(not isinstance(p, PR.OllamaProvider) for p in reg.values())


# ---------------------------------------------------------------- redo


def test_regenerate_replaces_the_previous_answer(client):
    _chat(client, {"message": "第一问"})
    msgs = client.get("/api/session/get").get_json()["messages"]
    user, old_answer = msgs[0], msgs[1]
    events = _chat(client, {"mode": "regenerate", "message_id": user["id"], "model": "fake-chat-model"})
    assert any(e.get("done") for e in events)
    after = client.get("/api/session/get").get_json()
    assert [m["role"] for m in after["messages"]] == ["user", "assistant"]
    assert after["messages"][0]["content"] == "第一问"
    assert after["messages"][1]["content"] == "答案"
    assert any(h.get("event") == "regenerate" for h in after["history"])


def test_edit_question_rewrites_text_and_drops_old_answer(client):
    _chat(client, {"message": "原来的问题"})
    user = client.get("/api/session/get").get_json()["messages"][0]
    _chat(client, {"mode": "edit", "message_id": user["id"], "message": "改后的问题", "model": "fake-chat-model"})
    after = client.get("/api/session/get").get_json()
    assert [m["role"] for m in after["messages"]] == ["user", "assistant"]
    assert after["messages"][0]["content"] == "改后的问题"
    assert after["messages"][0]["edited_from"] == "原来的问题"
    assert any(h.get("event") == "edit" for h in after["history"])


def test_regenerate_rejects_assistant_message_id(client):
    _chat(client, {"message": "问题"})
    answer = client.get("/api/session/get").get_json()["messages"][1]
    r = client.post("/api/chat", json={"mode": "regenerate", "message_id": answer["id"], "model": "fake-chat-model"})
    assert r.status_code == 400


def test_grounded_abstention_translates_to_japanese(client):
    events = _chat(client, {"message": "无材料的问题", "grounded": True, "lang": "ja"})
    text = "".join(e["delta"] for e in events if "delta" in e)
    assert "資料には言及されていません" in text
    meta = [e for e in events if e.get("done")][0]["meta"]
    assert meta["audit"]["score"] == 1.0


# ---------------------------------------------------------------- settings


def test_settings_endpoint_lists_providers_with_masked_keys(client):
    j = client.get("/api/settings").get_json()
    assert j["success"] is True
    fake = [p for p in j["providers"] if p["id"] == "fake-web"][0]
    assert fake["configured"] is True
    assert "sk-fake-web-0000" not in json.dumps(j)
    assert fake["masked"].startswith("***len")
    assert j["default"]["provider"] == "fake-web"  # registry fallback in tests


def test_settings_save_provider_masks_and_persists(client, tmp_path):
    j = client.post(
        "/api/settings/provider",
        json={"id": "deepseek", "base": "https://api.deepseek.com/v1", "key": "sk-test-abcdef123456"},
    ).get_json()
    assert j["success"] is True
    assert "sk-test-abcdef123456" not in json.dumps(j)
    assert j["provider"]["masked"].startswith("***len")

    raw = (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert "sk-test-abcdef123456" in raw  # stored locally, not in API responses
    mode = oct((tmp_path / "settings.json").stat().st_mode & 0o777)
    assert mode == "0o600"

    listed = client.get("/api/settings").get_json()
    row = [p for p in listed["providers"] if p["id"] == "deepseek"][0]
    assert row["source"] == "settings" and row["base"].endswith("/v1")
    assert "sk-test-abcdef123456" not in json.dumps(listed)


def test_settings_default_and_remove(client, tmp_path):
    client.post("/api/settings/provider", json={"id": "deepseek", "base": "https://api.deepseek.com/v1", "key": "sk-xxxxxxxxxxxx"})
    j = client.post("/api/settings/default", json={"provider": "deepseek", "model": "deepseek-flash"}).get_json()
    assert j["success"] is True and j["default"]["model"] == "deepseek-flash"
    assert client.delete("/api/settings/provider/deepseek").get_json()["success"] is True
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["providers"] == {}


def test_default_selection_prefers_deepseek_flash(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAP_SETTINGS_PATH", str(tmp_path / "s.json"))

    class FakeDeepSeek:
        base = "https://api.deepseek.com/v1"

        def list_models(self):
            return ["deepseek-v4-pro", "deepseek-flash"]

    assert webapp.default_selection({"deepseek": FakeDeepSeek()}) == {
        "provider": "deepseek",
        "model": "deepseek-flash",
    }


def test_settings_import_opencode_json_masks_and_keeps_models(client, tmp_path):
    doc = {
        "provider": {
            "deepseek": {
                "options": {"baseURL": "https://api.deepseek.com/v1", "apiKey": "sk-import-1234567890"},
                "models": {"deepseek-flash": {}, "deepseek-v4-pro": {}},
            },
            "nvidia": {"options": {"baseURL": "https://integrate.api.nvidia.com/v1", "apiKey": "nvapi-abcdefghij"}},
        }
    }
    j = client.post("/api/settings/import", json={"json": json.dumps(doc)}).get_json()
    assert j["success"] is True and j["count"] == 2
    assert "sk-import" not in json.dumps(j) and "nvapi-" not in json.dumps(j)

    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["providers"]["deepseek"]["models"] == ["deepseek-flash", "deepseek-v4-pro"]
    assert saved["providers"]["deepseek"]["apiKey"] == "sk-import-1234567890"  # local file only

    listed = client.get("/api/settings").get_json()
    assert "sk-import" not in json.dumps(listed) and "nvapi-" not in json.dumps(listed)
    row = [p for p in listed["providers"] if p["id"] == "deepseek"][0]
    assert row["source"] == "settings"
    assert "deepseek-flash" in row["models"]


def test_manual_models_are_used_when_endpoint_lists_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAP_SETTINGS_PATH", str(tmp_path / "s.json"))
    from clap.core import providers as PR
    from clap.core import settings_store as SS

    SS.set_provider(
        "manual", "http://127.0.0.1:9/v1", "sk-manual-12345678", models=["manual-a", "manual-b"]
    )
    reg = PR.discover_providers()
    assert reg["manual"].list_models() == ["manual-a", "manual-b"]


def test_settings_import_rejects_broken_json(client):
    r = client.post("/api/settings/import", json={"json": "{not json"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


# ---------------------------------------------------------------- text evidence


def test_text_upload_accepts_plain_text_and_rejects_other_formats(client):
    import io

    ok = client.post(
        "/api/text/add",
        data={"file": (io.BytesIO("量子力学研究波函数与概率密度。\n".encode("utf-8")), "notes.md")},
        content_type="multipart/form-data",
    ).get_json()
    assert ok["success"] is True and ok["count"] == 1

    bad = client.post(
        "/api/text/add",
        data={"file": (io.BytesIO(b"%PDF-1.4 fake"), "paper.pdf")},
        content_type="multipart/form-data",
    )
    assert bad.status_code == 415

    listed = client.get("/api/text/list").get_json()
    assert [f["name"] for f in listed["files"]] == ["notes.md"]


def test_text_search_and_article_match(client):
    import io

    client.post(
        "/api/text/add",
        data={"file": (io.BytesIO("前言\n\n第一章：量子力学研究波函数。\n\n第二章：无关。\n".encode()), "lesson.txt")},
        content_type="multipart/form-data",
    )
    hits = client.post("/api/text/search", json={"query": "量子力学研究波函数"}).get_json()
    assert hits["success"] is True and hits["results"]
    hit = hits["results"][0]
    art = client.get("/api/text/article", query_string={"name": hit["path"], "q": "量子力学研究波函数"}).get_json()
    assert art["success"] is True
    assert art["match"] and "量子力学" in art["text"][art["match"]["start"]:art["match"]["end"]]


def test_text_add_path_skips_non_text_files(client, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("纯文本资料。", encoding="utf-8")
    (d / "b.pdf").write_bytes(b"%PDF-1.4")
    j = client.post("/api/text/add_path", json={"path": str(d)}).get_json()
    assert j["success"] is True and j["count"] == 1
    assert j["skipped"] == ["b.pdf"]


def test_text_delete(client):
    import io

    client.post(
        "/api/text/add",
        data={"file": (io.BytesIO(b"temp"), "temp.txt")},
        content_type="multipart/form-data",
    )
    j = client.post("/api/text/delete", json={"name": "temp.txt"}).get_json()
    assert j["success"] is True
    assert client.get("/api/text/list").get_json()["files"] == []


def test_grounded_chat_includes_text_sources(client, monkeypatch):
    from clap.core import text_sources as TX

    class FakeTextLib:
        roots = ["/fake"]

        def list_files(self):
            return []

        def search(self, query, top_k=5):
            return [TX.TextHit(name="notes.md", path="/fake/notes.md", title="notes",
                               text="量子力学研究波函数，波函数的模方是概率密度。", snippet="…", score=9.0)]

        def article(self, name, query=""):
            return {"kind": "text", "name": "notes.md", "path": "/fake/notes.md", "title": "notes",
                    "text": "量子力学研究波函数，波函数的模方是概率密度。", "chars": 24, "match": None}

    monkeypatch.setattr(webapp.TX, "library", lambda: FakeTextLib())
    events = _chat(client, {"message": "波函数的模方是什么", "grounded": True, "model": "fake-chat-model"})
    meta = [e for e in events if e.get("done")][0]["meta"]
    kinds = [s.get("kind") for s in meta["sources"]]
    assert "text" in kinds
    assert any(s["title"] == "notes" for s in meta["sources"])
