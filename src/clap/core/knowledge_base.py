"""Knowledge base management for CLAP."""

import os
import shutil
import stat
from pathlib import Path

import ollama

from clap.core.document import chunk_text, load_document


class KnowledgeBase:
    """Knowledge base for document indexing and retrieval."""

    def __init__(self, persist_directory: str = "", collection_name: str = "clap", embedding_model: str = "nomic-embed-text:latest"):
        self.persist_directory = persist_directory or str(Path.home() / ".clap" / "kb")
        self.embedding_model = embedding_model
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    def index_document(self, file_path: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> dict:
        """Index a document into the knowledge base."""
        try:
            docs = load_document(file_path)
            if not docs:
                return {"success": False, "error": f"Could not load: {file_path}"}

            all_text = "\n\n".join(d.content for d in docs)
            self.chunks = chunk_text(all_text, chunk_size, chunk_overlap)

            if not self.chunks:
                return {"success": False, "error": "No text extracted"}

            result = ollama.embed(model=self.embedding_model, input=self.chunks)
            self.embeddings = result.get("embeddings", [])

            self._save()
            return {"success": True, "chunks": len(self.chunks), "file": os.path.basename(file_path)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search the knowledge base."""
        if not self.chunks or not self.embeddings:
            self._load()
            if not self.chunks:
                return []

        try:
            result = ollama.embed(model=self.embedding_model, input=[query])
            query_emb = result["embeddings"][0]

            scores = []
            for i, emb in enumerate(self.embeddings):
                score = sum(a * b for a, b in zip(emb, query_emb, strict=False))
                scores.append((score, i))

            results = []
            for score, i in sorted(scores, reverse=True)[:k]:
                results.append({"content": self.chunks[i], "score": score})
            return results

        except Exception:
            return []

    def clear(self) -> dict:
        """Clear the knowledge base."""
        self.chunks = []
        self.embeddings = []

        if os.path.exists(self.persist_directory):
            def remove_readonly(func, path, _):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(self.persist_directory, onerror=remove_readonly)

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        return {"success": True}

    def get_stats(self) -> dict:
        """Get statistics."""
        self._load()
        return {"chunks": len(self.chunks), "persist_directory": self.persist_directory}

    def _save(self):
        """Save to disk."""
        import json
        data_path = Path(self.persist_directory) / "data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def _load(self):
        """Load from disk."""
        import json
        data_path = Path(self.persist_directory) / "data.json"
        if data_path.exists():
            try:
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = data.get("chunks", [])
                self.embeddings = data.get("embeddings", [])
            except Exception:
                pass
