"""
2. RAG-BASED QUESTION ANSWERING SYSTEM
Three stages: indexing -> retrieval -> response generation.

    documents -> chunk -> build index
    question  -> retrieve top-k relevant chunks
    question + chunks -> LLM generates a grounded, cited answer

Requires: pip install anthropic
Set ANTHROPIC_API_KEY in your environment before running.

Retrieval below uses dependency-free TF-IDF + cosine similarity so the
demo runs anywhere. Swap `_tfidf_vector` / `_cosine` for a real
embedding model (OpenAI, Voyage, sentence-transformers) plus a vector
store (FAISS, pgvector, Chroma) for production-scale corpora — the
`index.retrieve(query, top_k)` interface stays the same.
"""
import os
import re
import math
from collections import Counter
from dataclasses import dataclass
from anthropic import Anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
client = Anthropic()


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    id: str
    source: str
    text: str


# -------------------------------------------------------------------
# Stage 1: Indexing
# -------------------------------------------------------------------
def chunk_document(source: str, text: str, chunk_size: int = 500, overlap: int = 100):
    """Split a document into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(Chunk(id=f"{source}::chunk{idx}", source=source, text=chunk_text))
        idx += 1
        start += chunk_size - overlap
    return chunks


class VectorIndex:
    """TF-IDF index with cosine-similarity retrieval."""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self._token_lists = [_tokenize(c.text) for c in chunks]
        self._df = Counter()
        for tokens in self._token_lists:
            for term in set(tokens):
                self._df[term] += 1
        self._n = len(chunks)

    def _tfidf(self, tokens):
        tf = Counter(tokens)
        return {t: c * (math.log((1 + self._n) / (1 + self._df.get(t, 0))) + 1)
                for t, c in tf.items()}

    @staticmethod
    def _cosine(a, b):
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        nb = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (na * nb)

    def retrieve(self, query: str, top_k: int = 4):
        q_vec = self._tfidf(_tokenize(query))
        scored = []
        for chunk, tokens in zip(self.chunks, self._token_lists):
            score = self._cosine(q_vec, self._tfidf(tokens))
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scored[:top_k] if score > 0]


def build_index(documents: dict) -> VectorIndex:
    """documents: {source_name: raw_text}"""
    all_chunks = []
    for source, text in documents.items():
        all_chunks.extend(chunk_document(source, text))
    return VectorIndex(all_chunks)


# -------------------------------------------------------------------
# Stage 2: Retrieval (VectorIndex.retrieve, used directly below)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Stage 3: Response generation, grounded + cited
# -------------------------------------------------------------------
def generate_answer(question: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        f"[{i+1}] (source: {c.source})\n{c.text}" for i, c in enumerate(retrieved_chunks)
    )
    prompt = (
        "Answer the question using ONLY the numbered context passages below. "
        "Cite passages inline like [1], [2]. If the context doesn't contain "
        "the answer, say so explicitly rather than guessing.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = client.messages.create(
        model=MODEL, max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------
def rag_answer(question: str, index: VectorIndex, top_k: int = 4) -> dict:
    retrieved = index.retrieve(question, top_k=top_k)
    if not retrieved:
        return {"answer": "No relevant context found in the index.", "sources": []}
    answer = generate_answer(question, retrieved)
    return {"answer": answer, "sources": [c.id for c in retrieved]}


if __name__ == "__main__":
    documents = {
        "ai_dev_notes.txt": (
            "AI coding assistants can generate code, explain programming concepts, "
            "detect bugs and help developers understand large codebases. Large "
            "language models such as Qwen, Llama and GPT can be integrated into "
            "development environments. Local models are useful when privacy matters "
            "because data stays on the user's computer. AI-generated code still "
            "requires human review for security, performance, and correctness."
        ),
        "roles_notes.txt": (
            "Organizations are combining AI with traditional software engineering "
            "practices, creating new roles such as AI engineer, LLM engineer, and "
            "AI platform engineer. Knowledge of Python, APIs, databases, cloud "
            "platforms, and DevOps is increasingly useful for these roles."
        ),
    }

    index = build_index(documents)
    result = rag_answer("What skills do AI engineering roles need?", index)
    print(result["answer"])
    print("Sources:", result["sources"])
