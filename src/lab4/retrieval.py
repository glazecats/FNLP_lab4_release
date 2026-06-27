from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import Question


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*|\d+(?:\.\d+)?")


CHINESE_HINTS: dict[str, str] = {
    "黑洞": "black hole event horizon Schwarzschild radius gravitation",
    "事件视界": "event horizon Schwarzschild radius",
    "引力": "gravity gravitation gravitational force potential",
    "轨道": "orbit orbital angular momentum",
    "开普勒": "Kepler law orbit period",
    "波": "wave wavelength frequency speed interference diffraction",
    "光": "light optics photon wavelength frequency",
    "射线": "x ray diffraction Bragg wavelength crystal",
    "岩盐": "rock salt NaCl sodium chloride crystal spacing Bragg diffraction nearest neighbor equilibrium separation",
    "食盐": "rock salt NaCl sodium chloride crystal spacing Bragg diffraction nearest neighbor equilibrium separation",
    "衍射": "diffraction interference grating Bragg",
    "干涉": "interference diffraction phase",
    "电": "electric charge current voltage field potential capacitance",
    "磁": "magnetic field flux induction",
    "热": "thermodynamics heat temperature entropy enthalpy",
    "温度": "temperature Kelvin thermal",
    "压强": "pressure gas ideal",
    "量子": "quantum photon electron energy level wavefunction",
    "原子": "atomic atom electron orbital spectrum",
    "分子": "molecule molecular vibration rotation spectroscopy",
    "半导体": "semiconductor band gap carrier Fermi",
    "相对论": "relativity Lorentz energy momentum",
    "动量": "momentum impulse collision",
    "能量": "energy work kinetic potential",
    "概率": "probability distribution statistical",
    "熵": "entropy thermodynamics statistical",
    "焓": "enthalpy heat reaction",
    "自由能": "Gibbs Helmholtz free energy",
    "平衡": "equilibrium constant chemical potential",
    "反应速率": "reaction rate kinetics Arrhenius",
    "活化能": "activation energy Arrhenius",
    "电池": "cell electrode Nernst electrochemistry",
    "酸": "acid base pH equilibrium",
    "碱": "acid base pH equilibrium",
    "积分": "integral distribution probability average",
    "截面": "cross section probability number density target thickness absorption capture scattering",
    "俘获": "capture absorption cross section probability number density target thickness",
    "金箔": "gold foil Rutherford alpha scattering nucleus impact parameter fraction target thickness",
    "偏转": "deflection scattering angle impact parameter Rutherford",
    "alpha": "alpha particle Rutherford scattering gold foil nucleus",
    "阿尔法": "alpha particle Rutherford scattering gold foil nucleus",
    "频率": "frequency wavelength angular frequency",
    "波长": "wavelength frequency photon",
    "折射": "refraction Snell lens optics",
    "透镜": "lens focal length optics",
}


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    text: str
    score: float


@dataclass
class ChunkRecord:
    source: str
    text: str
    tokens: dict[str, int]
    length: int


class TextbookIndex:
    def __init__(self, records: list[ChunkRecord], doc_freq: dict[str, int]) -> None:
        self.records = records
        self.doc_freq = doc_freq
        self.avg_len = sum(record.length for record in records) / max(len(records), 1)

    @classmethod
    def load_or_build(
        cls,
        textbook_dir: str | Path = "textbooks-tex",
        cache_path: str | Path = "cache/textbook_index_v2.json",
    ) -> "TextbookIndex":
        cache = Path(cache_path)
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            records = [
                ChunkRecord(
                    source=item["source"],
                    text=item["text"],
                    tokens=item["tokens"],
                    length=item["length"],
                )
                for item in data["records"]
            ]
            return cls(records, data["doc_freq"])

        records = []
        for path in sorted(Path(textbook_dir).glob("*.tex")):
            records.extend(_chunk_tex(path))
        doc_freq: Counter[str] = Counter()
        for record in records:
            doc_freq.update(record.tokens.keys())
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [
                {
                    "source": record.source,
                    "text": record.text,
                    "tokens": record.tokens,
                    "length": record.length,
                }
                for record in records
            ],
            "doc_freq": dict(doc_freq),
        }
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return cls(records, dict(doc_freq))

    def search(self, question: Question, top_k: int = 6) -> list[RetrievedChunk]:
        query_tokens = _query_tokens(question)
        if not query_tokens:
            return []
        scores = []
        total_docs = len(self.records)
        k1 = 1.5
        b = 0.75
        for record in self.records:
            score = 0.0
            for token, qtf in query_tokens.items():
                tf = record.tokens.get(token)
                if not tf:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1 - b + b * record.length / max(self.avg_len, 1))
                score += qtf * idf * (tf * (k1 + 1) / denom)
            if score > 0:
                scores.append((score, record))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(source=record.source, text=record.text, score=score)
            for score, record in scores[:top_k]
        ]


def _chunk_tex(path: Path, *, chunk_words: int = 220, overlap: int = 50) -> list[ChunkRecord]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _clean_tex(raw)
    words = raw.split()
    records: list[ChunkRecord] = []
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_words]
        if len(window) < 40:
            break
        text = " ".join(window)
        tokens = Counter(_tokenize(text))
        if tokens:
            records.append(
                ChunkRecord(
                    source=f"{path.name}#{index}",
                    text=text,
                    tokens=dict(tokens),
                    length=sum(tokens.values()),
                )
            )
        index += 1
        start += max(chunk_words - overlap, 1)
    return records


def _clean_tex(text: str) -> str:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\(chapter|section|subsection|subsubsection)\*?\{([^{}]*)\}", r" \2 ", text)
    text = re.sub(r"\\(begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _query_tokens(question: Question) -> Counter[str]:
    parts = [
        question.field or "",
        question.subfield or "",
        question.theorem or "",
        question.unit or "",
        question.question or "",
    ]
    hints = []
    for zh, en in CHINESE_HINTS.items():
        if zh in question.question:
            hints.append(en)
    raw = " ".join(parts + hints)
    tokens = Counter(_tokenize(raw))
    for token in list(tokens):
        if token.isdigit() and len(token) < 3:
            del tokens[token]
    return tokens


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
