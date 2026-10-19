"""Chat thread for handling asynchronous LLM communication."""

import io
import time

import ollama
from PySide6.QtCore import QThread, Signal

from clap.core.document import chunk_text, is_image, is_svg, load_document


def wall_ms(start: float) -> int:
    """Elapsed wall-clock milliseconds since `start`."""
    return int((time.time() - start) * 1000)


class ChatThread(QThread):
    """Thread for handling chat communication with Ollama.

    By default the module-level ollama functions are used (local server,
    unchanged behaviour). An optional pre-built `ollama.Client` can be
    injected (e.g. `Client(host=..., headers=...)`); the call shapes and
    response protocol are identical. While streaming, the final chunk
    (done=True, carrying `eval_count` / `prompt_eval_count`) is watched and
    real provenance recorded into `last_meta`; fields the server did not
    provide stay None (never fabricated).
    """

    new_text = Signal(str)

    def __init__(
        self,
        messages: list[dict],
        model: str,
        path: str = "",
        embed_model: str = "nomic-embed-text:latest",
        client=None,
        provider_name: str = "ollama-local",
    ):
        super().__init__()
        self.messages = messages
        self.model = model
        self.path = path
        self.embed_model = embed_model
        self.client = client
        self.provider_name = provider_name
        self.last_meta: dict = {}

    # ------------------------------------------------------------------
    # unified call layer (client injection point, default = local ollama)
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict], model: str | None = None):
        if self.client is not None:
            return self.client.chat(model=model or self.model, messages=messages, stream=True)
        return ollama.chat(model=model or self.model, messages=messages, stream=True)

    def _generate(self, prompt: str, images: list[bytes] | None = None):
        if self.client is not None:
            return self.client.generate(model=self.model, prompt=prompt, images=images or [], stream=True)
        return ollama.generate(model=self.model, prompt=prompt, images=images or [], stream=True)

    def _embed(self, model: str, input_texts: list[str]) -> dict:
        if self.client is not None:
            return self.client.embed(model=model, input=input_texts)
        return ollama.embed(model=model, input=input_texts)

    def _watch(self, stream, t0: float):
        """Yield chunks unchanged; record provenance when the final chunk passes."""
        try:
            for chunk in stream:
                try:
                    if chunk.get("done"):
                        self.last_meta = {
                            "provider": self.provider_name,
                            "model": self.model,
                            "latency_ms": wall_ms(t0),
                            "tokens": {
                                "in": chunk.get("prompt_eval_count"),
                                "out": chunk.get("eval_count"),
                            },
                            "error": None,
                        }
                except (AttributeError, TypeError):
                    pass
                yield chunk
        except Exception:
            raise

    def run(self):
        """Execute the chat thread, streaming responses from the model."""
        try:
            if self.path and is_image(self.path):
                self._process_image()
            elif self.path and is_svg(self.path):
                self._process_svg()
            elif self.path:
                self._process_document()
            else:
                self._process_chat()
        except Exception as e:
            self.new_text.emit(f"Error: {e}")
            self.last_meta = {
                "provider": self.provider_name,
                "model": self.model,
                "latency_ms": None,
                "tokens": {"in": None, "out": None},
                "error": str(e)[:200],
            }

    def _process_image(self):
        """Process image with vision model."""
        with open(self.path, "rb") as f:
            image_data = f.read()

        t0 = time.time()
        stream = self._generate(prompt=self.messages[-1]["content"], images=[image_data])
        for chunk in self._watch(stream, t0):
            self.new_text.emit(chunk.get("response", ""))

    def _process_svg(self):
        """Process SVG with vision model by rendering to PNG."""
        docs = load_document(self.path)
        if not docs:
            self._process_chat()
            return

        doc = docs[0]
        svg_content = doc.metadata.get("raw_svg", "")
        extracted_text = doc.content

        try:
            image_data = self._render_svg_to_png(svg_content)
            if image_data:
                prompt = self.messages[-1]["content"]
                if extracted_text:
                    prompt = f"[SVG contains text: {extracted_text}]\n\n{prompt}"
                t0 = time.time()
                for chunk in self._watch(self._generate(prompt=prompt, images=[image_data]), t0):
                    self.new_text.emit(chunk.get("response", ""))
            else:
                if extracted_text:
                    prompt = f"[SVG file content]\nText extracted: {extracted_text}\n\nQuestion: {self.messages[-1]['content']}"
                    t0 = time.time()
                    for chunk in self._watch(
                        self._chat([{"role": "user", "content": prompt}]), t0
                    ):
                        self.new_text.emit(chunk["message"]["content"])
                else:
                    self._process_chat()
        except Exception:
            self._process_chat()

    def _render_svg_to_png(self, svg_content: str) -> bytes | None:
        """Render SVG content to PNG bytes using CairoSVG or Qt."""
        try:
            import cairosvg

            png_data = cairosvg.svg2png(bytestring=svg_content.encode("utf-8"))
            return png_data
        except ImportError:
            pass
        except Exception:
            pass

        try:
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(svg_content.encode("utf-8"))
            if not renderer.isValid():
                return None

            size = renderer.defaultSize()
            if size.width() > 2048 or size.height() > 2048:
                size = size.scaled(2048, 2048, 1)

            image = QImage(size, QImage.Format_ARGB32)
            image.fill(0xFFFFFFFF)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()

            buffer = io.BytesIO()
            image.save(buffer, "PNG")
            return buffer.getvalue()
        except Exception:
            return None

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
            embeddings = self._embed(self.embed_model, chunks)
            query_emb = self._embed(self.embed_model, [self.messages[-1]["content"]])[
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

            t0 = time.time()
            for chunk in self._watch(self._chat(messages), t0):
                self.new_text.emit(chunk["message"]["content"])
        except Exception:
            # embedding unavailable -> plain chat fallback (original behaviour)
            self._process_chat()

    def _process_chat(self):
        """Process regular chat without file."""
        t0 = time.time()
        for chunk in self._watch(self._chat(self.messages), t0):
            self.new_text.emit(chunk["message"]["content"])
