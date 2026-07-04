import argparse
from millenniumdb_driver import driver
from color_print import print_error, print_header, print_info, print_step, print_success
from config import URL
from sparse_retriever import BM25Index, SparseDocument
from utils import q_run


DEFAULT_INDEX_PATH = "output/sparse_bm25_index.pkl"


def fetch_sparse_documents(session, limit: int | None = None) -> list[SparseDocument]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    MATCH (?i :Intervention)-[:HasEmbedding]->(?c :Embedding)
    RETURN ?c, ?i, ?c.content AS ?content
    {limit_clause}
    """

    documents = []
    result = q_run(session, query)

    for record in result:
        chunk = record.get("c")
        intervention = record.get("i")
        content = record.get("content")

        if not content:
            continue

        documents.append(
            SparseDocument(
                doc_id=str(chunk),
                intervention_id=str(intervention),
                content=content,
            )
        )

    return documents


def build_index(output_path: str = DEFAULT_INDEX_PATH, limit: int | None = None) -> BM25Index:


    print_header("CONSTRUYENDO INDICE SPARSE BM25")

    print_step(1, f"Conectando a MillenniumDB en {URL}...")
    db = driver(URL)
    print_success("Conexion establecida")

    try:
        print_step(2, "Leyendo chunks e identificadores desde MillenniumDB...")
        with db.session() as session:
            documents = fetch_sparse_documents(session, limit=limit)
        print_success(f"Documentos recuperados: {len(documents)}")
    finally:
        db.close()

    if not documents:
        raise RuntimeError("No se encontraron documentos para construir el indice sparse.")

    print_step(3, "Calculando pesos BM25...")
    index = BM25Index(documents)
    print_success("Indice BM25 construido")

    print_step(4, f"Guardando indice en {output_path}...")
    index.save(output_path)
    print_success("Indice sparse guardado correctamente")

    print_info("Este archivo local permite buscar por terminos exactos sin recalcular todo.")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye el indice sparse BM25 para la Parte 3")
    parser.add_argument(
        "--output",
        default=DEFAULT_INDEX_PATH,
        help="Ruta donde guardar el indice BM25",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad maxima de documentos a indexar. Util para pruebas chicas.",
    )
    args = parser.parse_args()

    try:
        build_index(output_path=args.output, limit=args.limit)
    except Exception as exc:
        print_error(f"Error construyendo indice sparse: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
