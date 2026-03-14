"""Chat thread for handling asynchronous LLM communication."""

import ollama
from PySide6.QtCore import QThread, Signal

from clap.core.document import chunk_text, is_image, load_document


class ChatThread(QThread):
    """Thread for handling chat communication with Ollama."""

    new_text = Signal(str)

    def __init__(
        self,
        messages: list[dict],
        model: str,
        path: str = "",
        embed_model: str = "nomic-embed-text:latest",
    ):
        super().__init__()
        self.messages = messages
        self.model = model
        self.path = path
        self.embed_model = embed_model

    def run(self):
        """Execute the chat thread, streaming responses from the model."""
        try:
            if self.path and is_image(self.path):
                self._process_image()
            elif self.path:
                self._process_document()
            else:
                self._process_chat()
        except Exception as e:
            self.new_text.emit(f"Error: {e}")

    def _process_image(self):
        """Process image with vision model."""
        with open(self.path, "rb") as f:
            image_data = f.read()

        for chunk in ollama.generate(
            model=self.model, prompt=self.messages[-1]["content"], images=[image_data], stream=True
        ):
            self.new_text.emit(chunk.get("response", ""))

    def _process_document(self):
        """Process document with RAG."""
        docs = load_document(self.path)
        if not docs:
            self._process_chat()
            return

        all_text = "\n\n".join(d.content for d in docs)
        chunks = chunk_text(all_text)

        if not chunks:
            self._process_chat()
            return

        try:
            embeddings = ollama.embed(model=self.embed_model, input=chunks)
            query_emb = ollama.embed(model=self.embed_model, input=[self.messages[-1]["content"]])[
                "embeddings"
            ][0]

            scores = []
            for i, emb in enumerate(embeddings["embeddings"]):
                score = sum(a * b for a, b in zip(emb, query_emb, strict=False))
                scores.append((score, i))

            top_chunks = [chunks[i] for _, i in sorted(scores, reverse=True)[:4]]
            context = "\n\n---\n\n".join(top_chunks)

            prompt = f"Context:\n{context}\n\nQuestion: {self.messages[-1]['content']}\n\nAnswer based on the context:"
            messages = [{"role": "user", "content": prompt}]

            for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
                self.new_text.emit(chunk["message"]["content"])
        except Exception:
            self._process_chat()

    def _process_chat(self):
        """Process regular chat without file."""
        for chunk in ollama.chat(model=self.model, messages=self.messages, stream=True):
            self.new_text.emit(chunk["message"]["content"])
