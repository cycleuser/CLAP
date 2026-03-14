"""Web interface for CLAP."""

import os

import ollama
from flask import Flask, jsonify, request

from clap.core.knowledge_base import KnowledgeBase

app = Flask(__name__)
kb = KnowledgeBase()


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head><title>CLAP</title></head>
<body>
<h1>CLAP API</h1>
<ul>
<li>GET /api/models</li>
<li>POST /api/chat {"message": "...", "model": "..."}</li>
<li>POST /api/kb/index (multipart file)</li>
<li>POST /api/kb/search {"query": "...", "k": 5}</li>
<li>GET /api/kb/stats</li>
<li>POST /api/kb/clear</li>
</ul>
</body>
</html>
"""


@app.route("/api/models")
def get_models():
    try:
        models = [m["model"] for m in ollama.list().get("models", [])]
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        msg = data.get("message", "")
        model = data.get("model", "gemma3:1b")

        if not msg:
            return jsonify({"success": False, "error": "No message"})

        full = ""
        for chunk in ollama.chat(model=model, messages=[{"role": "user", "content": msg}], stream=True):
            full += chunk["message"]["content"]

        return jsonify({"success": True, "response": full})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/kb/index", methods=["POST"])
def kb_index():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file"})

        f = request.files["file"]
        path = os.path.join("/tmp", f.filename)
        f.save(path)
        result = kb.index_document(path)
        os.remove(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/kb/search", methods=["POST"])
def kb_search():
    try:
        data = request.get_json()
        results = kb.search(data.get("query", ""), data.get("k", 5))
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/kb/stats")
def kb_stats():
    return jsonify(kb.get_stats())


@app.route("/api/kb/clear", methods=["POST"])
def kb_clear():
    return jsonify(kb.clear())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLAP Web Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"CLAP web server: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
