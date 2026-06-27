from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?|\\[A-Za-z]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def clean_tex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^{}]+)\}", r"\n\1\n", text)
    text = re.sub(r"\\(?:textbf|textit|emph|mathrm|text)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def chunk_text(text: str, *, chunk_chars: int = 1800, overlap: int = 250) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\\\\", text) if len(p.strip()) > 80]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_chars:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = (current[-overlap:] + "\n" + para).strip() if current else para
    if current:
        chunks.append(current)
    return chunks


class TextbookIndex:
    def __init__(self, chunks: list[dict[str, str]]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(item["text"]) for item in chunks]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(1, len(self.doc_tokens))

    @classmethod
    def build(cls, tex_dir: str | Path) -> "TextbookIndex":
        tex_dir = Path(tex_dir)
        chunks: list[dict[str, str]] = []
        for path in sorted(tex_dir.glob("*.tex")):
            text = clean_tex(path.read_text(encoding="utf-8", errors="ignore"))
            for i, chunk in enumerate(chunk_text(text)):
                chunks.append({"source": f"{path.name}#{i}", "text": chunk})
        return cls(chunks)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "TextbookIndex":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f)["chunks"])

    def search(self, query: str, *, field: str | None = None, top_k: int = 4) -> list[RetrievedChunk]:
        query_terms = Counter(tokenize(query))
        if not query_terms:
            return []
        n_docs = len(self.chunks)
        scores: defaultdict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75

        for i, tokens in enumerate(self.doc_tokens):
            source = self.chunks[i]["source"]
            if field == "physics" and source.startswith("Atkins"):
                continue
            if field == "chemistry" and source.startswith("UniversityPhysics"):
                continue
            tf = Counter(tokens)
            doc_len = len(tokens)
            for term, qtf in query_terms.items():
                if term not in tf:
                    continue
                idf = math.log(1 + (n_docs - self.doc_freq[term] + 0.5) / (self.doc_freq[term] + 0.5))
                denom = tf[term] + k1 * (1 - b + b * doc_len / max(1, self.avg_len))
                scores[i] += idf * (tf[term] * (k1 + 1) / denom) * qtf

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                source=self.chunks[i]["source"],
                text=self.chunks[i]["text"][:1600],
                score=score,
            )
            for i, score in ranked
        ]


def load_or_build_index(tex_dir: str | Path, cache_path: str | Path) -> TextbookIndex:
    cache = Path(cache_path)
    if cache.exists():
        return TextbookIndex.load(cache)
    index = TextbookIndex.build(tex_dir)
    index.save(cache)
    return index

