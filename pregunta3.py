import argparse
import csv
import hashlib
import os
import pickle
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from color_print import Colors, print_error, print_header, print_info, print_step, print_success
from config import K, MODEL, SYSTEM_PROMPT, URL
from dotenv import load_dotenv

import queries as q
from build_sparse_index import DEFAULT_INDEX_PATH
from sparse_retriever import BM25Index, RetrievalResult, reciprocal_rank_fusion
from utils import get_embedding, q_run
from openai import OpenAI

from millenniumdb_driver import driver
from sentence_transformers import SentenceTransformer



SPARSE_MODE = "sparse"
DENSE_MODE = "dense"
HYBRID_MODE = "hybrid"
CSV_LOG_FILE = "output/evaluacion_resultados.csv"
SPARSE_CACHE_DIR = Path("output/sparse_cache")

load_dotenv()


def retrieve_sparse(index: BM25Index, query: str, k: int) -> list[RetrievalResult]:
    return index.search(query, k=k)


def retrieve_sparse_all(index: BM25Index, query: str) -> list[RetrievalResult]:
    return index.search_all(query)


def _sparse_cache_path(index_path: str, query: str) -> Path:
    resolved_index = Path(index_path).resolve()
    index_mtime = os.path.getmtime(resolved_index)
    cache_key = hashlib.sha256(
        f"{resolved_index}|{index_mtime:.6f}|{query}".encode("utf-8")
    ).hexdigest()
    return SPARSE_CACHE_DIR / f"{cache_key}.pkl"


def load_cached_sparse_results(index_path: str, query: str) -> list[RetrievalResult] | None:
    cache_path = _sparse_cache_path(index_path, query)
    if not cache_path.exists():
        return None

    with cache_path.open("rb") as file:
        payload = pickle.load(file)

    return payload.get("results")


def save_cached_sparse_results(index_path: str, query: str, results: list[RetrievalResult]) -> None:
    SPARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _sparse_cache_path(index_path, query)
    payload = {
        "results": results,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "index_path": str(Path(index_path).resolve()),
    }

    with cache_path.open("wb") as file:
        pickle.dump(payload, file)


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    print_step(3, "Cargando modelo de embeddings para recuperacion densa...")
    return SentenceTransformer(MODEL)


def retrieve_dense(session, query_vector: list[float], k: int) -> list[RetrievalResult]:
    result = q_run(
        session,
        q.HNSW_TOP_K_WITH_INTERVENTION_QUERY,
        parameters={"query_embedding": query_vector},
        replaces={"k": k},
    )

    dense_results = []
    for rank, record in enumerate(result, start=1):
        chunk = record.get("c")
        intervention = record.get("i")
        content = record.get("content")
        distance = record.get("distance")

        dense_results.append(
            RetrievalResult(
                doc_id=str(chunk),
                intervention_id=str(intervention),
                content=content,
                score=float(distance),
                rank=rank,
                source="dense_hnsw",
            )
        )

    return dense_results


def retrieve_context(
    query: str,
    mode: str,
    k: int,
    dense_candidates: int,
    sparse_candidates: int,
    index_path: str,
) -> list[RetrievalResult]:
    cached_sparse_results = load_cached_sparse_results(index_path, query)

    if cached_sparse_results is None:
        index = BM25Index.load(index_path)
        cached_sparse_results = retrieve_sparse_all(index, query)
        save_cached_sparse_results(index_path, query, cached_sparse_results)

    if mode == SPARSE_MODE:
        return cached_sparse_results[:k]

   
    embedding_model = load_embedding_model()
    query_vector = get_embedding(embedding_model, query)

    db = driver(URL)
    try:
        with db.session() as session:
            if mode == DENSE_MODE:
                return retrieve_dense(session, query_vector, k=k)

            dense_results = retrieve_dense(session, query_vector, k=dense_candidates)
    finally:
        db.close()

    sparse_results = cached_sparse_results[:sparse_candidates]
    return reciprocal_rank_fusion(dense_results, sparse_results, k=k)


def generate_answer(query: str, context_documents: list[RetrievalResult]) -> str:
   

    context_parts = []
    for result in context_documents:
        context_parts.append(
            f"ID Documento [{result.intervention_id}]\n"
            f"Fuente recuperacion: {result.source}\n"
            f"{result.content}"
        )

    context_text = "\n\n---\n\n".join(context_parts)
    prompt = (
        "Contexto:\n"
        f"{context_text}\n\n"
        "Pregunta:\n"
        f"{query}"
    )

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content


def save_log(
    results: list[RetrievalResult],
    query: str,
    k: int,
    answer: str,
    strategy: str,
    csv_file: str = CSV_LOG_FILE,
) -> None:
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "id_ejecucion",
                    "fecha",
                    "estrategia",
                    "query",
                    "k",
                    "distancia_promedio",
                    "contexto_recuperado",
                    "respuesta_llm",
                ]
            )

        avg_score = sum(result.score for result in results) / len(results) if results else 0.0
        context_str = "\n".join(
            [
                f"ID[{result.intervention_id}] SCORE[{result.score:.4f}] "
                f"SOURCE[{result.source}]\n{result.content}"
                for result in results
            ]
        )

        writer.writerow(
            [
                str(uuid.uuid4())[:8],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                strategy,
                query,
                k,
                f"{avg_score:.4f}",
                context_str,
                answer,
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG sparse e hibrido para la Parte 3")
    parser.add_argument("query", type=str, help="Pregunta a responder")
    parser.add_argument(
        "--mode",
        choices=[SPARSE_MODE, DENSE_MODE, HYBRID_MODE],
        default=HYBRID_MODE,
        help="Estrategia de recuperacion",
    )
    parser.add_argument("--k", type=int, default=K, help="Cantidad final de documentos")
    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=20,
        help="Candidatos densos antes de fusionar",
    )
    parser.add_argument(
        "--sparse-candidates",
        type=int,
        default=20,
        help="Candidatos sparse antes de fusionar",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help="Ruta del indice BM25 construido con build_sparse_index.py",
    )
    args = parser.parse_args()

    print_header("RAG PARTE 3: SPARSE / HIBRIDO")
    print_info(f"Query: '{args.query}'")
    print_info(f"Estrategia: {args.mode}")
    print_info(f"K final: {args.k}")

    try:
        print_step(1, f"Cargando indice sparse desde {args.index}...")
        if not os.path.exists(args.index):
            print_error("No existe el indice sparse.")
            print_info("Primero ejecuta: python build_sparse_index.py")
            return 1

        print_step(2, "Recuperando contexto...")
        results = retrieve_context(
            query=args.query,
            mode=args.mode,
            k=args.k,
            dense_candidates=args.dense_candidates,
            sparse_candidates=args.sparse_candidates,
            index_path=args.index,
        )
    except Exception as exc:
        print_error(f"Error recuperando contexto: {exc}")
        return 1

    if not results:
        print_error("No se recuperaron documentos.")
        return 1

    print_success("Contexto recuperado")
    for result in results:
        print_success(
            f"Rank {result.rank} | ID[{result.intervention_id}] | "
            f"{result.source} | Score: {round(result.score, 5)}"
        )
        print_success(f"{Colors.BOLD}{result.content[:90]} . . .{Colors.END}")

    print_step(4, "Generando respuesta con GPT-4o-mini...")
    try:
        answer = generate_answer(args.query, results)
    except Exception as exc:
        print_error(f"Error generando respuesta: {exc}")
        return 1

    print_success("Respuesta generada")
    print("=" * 10)
    print_success(f"Respuesta GPT: {Colors.BOLD}{answer}{Colors.END}")
    print("=" * 10)

    print_step(5, f"Guardando resultado en {CSV_LOG_FILE}...")
    try:
        strategy = {
            SPARSE_MODE: "RAG_SPARSE_BM25",
            DENSE_MODE: "RAG_DENSO_HNSW",
            HYBRID_MODE: "RAG_HIBRIDO_RRF",
        }[args.mode]
        save_log(results, args.query, args.k, answer, strategy)
        print_success("Resultado guardado")
    except Exception as exc:
        print_error(f"Error guardando resultado: {exc}")
        return 1

    print_header("PROCESO COMPLETADO EXITOSAMENTE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
