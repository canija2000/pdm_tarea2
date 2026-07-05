import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path



TOKEN_RE = re.compile(r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return TOKEN_RE.findall(without_accents)

# estructuras de datos para representar docs/ y rresultados ttras busqueda en indice.
@dataclass
class SparseDocument:
    doc_id: str # id del chunk
    intervention_id: str # id de la intervencion 
    content: str # contenido del chunk


@dataclass
class RetrievalResult:
    doc_id: str # id del chunk que se recupero en el match [por score]
    intervention_id: str 
    content: str
    score: float # score obtenido en la busqueda
    rank: int # rank del resultado en la lista de resultados
    source: str 


class BM25Index:
    def __init__(self, documents: list[SparseDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(doc.content) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_docs]
        self.idf = self._compute_idf()

    # construimos el indice de rareza para todos los docs 
    def _compute_idf(self) -> dict[str, float]:
        doc_freq = defaultdict(int)
        total_docs = len(self.tokenized_docs)

        for tokens in self.tokenized_docs:
            for token in set(tokens):
                doc_freq[token] += 1 #construimos la freq de cada tok x documento.

        return {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores = []
        for idx, doc in enumerate(self.documents):
            score = self._score_document(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda item: item[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:k], start=1):
            doc = self.documents[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    intervention_id=doc.intervention_id,
                    content=doc.content,
                    score=score,
                    rank=rank,
                    source="sparse_bm25",
                )
            )
        return results

    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = self.term_frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        score = 0.0

        for token in query_tokens:
            freq = tf.get(token, 0)
            if freq == 0:
                continue

            idf = self.idf.get(token, 0.0)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += idf * (freq * (self.k1 + 1)) / denominator

        return score

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as file:
            return pickle.load(file)


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    by_doc_id: dict[str, RetrievalResult] = {}
    scores = defaultdict(float)

    for result in dense_results + sparse_results:
        by_doc_id[result.doc_id] = result
        scores[result.doc_id] += 1.0 / (rrf_k + result.rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    fused_results = []

    for rank, (doc_id, score) in enumerate(ranked[:k], start=1):
        base = by_doc_id[doc_id]
        fused_results.append(
            RetrievalResult(
                doc_id=base.doc_id,
                intervention_id=base.intervention_id,
                content=base.content,
                score=score,
                rank=rank,
                source="hybrid_rrf",
            )
        )

    return fused_results
