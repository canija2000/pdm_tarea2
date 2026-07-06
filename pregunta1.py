from config import URL, K, MODEL, SYSTEM_PROMPT
from utils import q_run, get_embedding
from color_print import Colors, print_header, print_step, print_success, print_error, print_info
import queries as q

import os
import argparse
from dotenv import load_dotenv
import uuid
from datetime import datetime
import csv

from millenniumdb_driver import driver
from sentence_transformers import SentenceTransformer
from openai import OpenAI



### Parametros

# tipo de busqueda (hnsw y fuerza bruta)
HNSW_SEARCH = 0
DENSE_SEARCH = 1

# path para registrar ejecuciones
CSV_LOG_FILE = "output/evaluacion_resultados.csv"



### Cliente OpenAI

# cargar variables de entorno desde .env
load_dotenv()

# cliente gpt (leer api key en .env)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



### Funciones auxiliares

# Recuperar texto de todos chunks de una intervencion
def get_chunks_content(session, intervention) -> str:
    chunk_list = []
    result = q_run(session, q.ALL_CHUNKS, parameters={"node": intervention})

    for record in result:
        chunk_content = record.get('content')
        chunk_list.append(chunk_content)

    return "".join(chunk_list)


# Recuperar top-k intervenciones
def retrieve_k_interventions(
        session, 
        query_vector: list[float], 
        k: int = K, 
        t: int = HNSW_SEARCH, 
        all_chunks = False
) -> list[tuple[str, float, str]]:
    
    query = q.HNSW_TOP_K_QUERY if t == HNSW_SEARCH else q.DENSE_TOP_K_QUERY
    result = q_run(session, query, parameters={"query_embedding": query_vector}, replaces={"k": k})

    interventions = []
    for record in result:
        chunk = record.get('c')
        content = record.get('content')
        distance = record.get('distance')

        tmp_result = q_run(session, q.GET_INTERVENTION, parameters={"chunk": chunk})
        tmp_record = tmp_result.records()[0]

        intervention = tmp_record.get('i')
        intervention_id = str(intervention)

        if all_chunks:
            content = get_chunks_content(session, intervention)

        interventions.append((intervention_id, distance, content)) 

    return interventions


# Generar respuesta con GPT
def generate_answer(
        query: str, 
        context_documents: list[tuple[str, float, str]]
) -> tuple[str, str, str]:
    # Crear el prompt con el contexto
    textos  = [f"ID Documento [{doc_id}]\n" + doc_text for doc_id, _, doc_text in context_documents]
    context_text = "\n\n---\n\n".join(textos)
    prompt = (
        "Contexto:"
        f"{context_text}"
        " "
        "Pregunta:"
        f"{query}"
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content, SYSTEM_PROMPT, prompt


# Guardar log con resultados de ejecucion
def save_log(k_interventions, query, k, answer, strategy, csv_file = CSV_LOG_FILE):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id_ejecucion", "fecha", "estrategia", "query", "k", "distancia_promedio", "contexto_recuperado", "respuesta_llm"])
        
        avg_dist = sum(dist for _, dist, _ in k_interventions) / len(k_interventions) if k_interventions else 0.0
        context_str = "\n".join([f"ID[{doc_id}] DIST[{dist:.4f}] \n{doc_text}" for doc_id, dist, doc_text in k_interventions])

        writer.writerow([
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            strategy,
            query,
            k,
            f"{avg_dist:.4f}",
            context_str,
            answer
        ])



def main():
    # formato esperado de ejecucion: 
    # python pregunta1.py "pregunta a responder" -k --t --all
    # -k: numero de documentos a recuperar
    # --t: tipo de busqueda (0: HNSW, 1: DENSE o fuerza bruta)
    # --all: recuperar todos los chunks de cada intervencion (0: NO, 1: SI)
    parser = argparse.ArgumentParser(description="Flujo de RAG Denso (Parte 1)")
    parser.add_argument("query", type=str, help="Pregunta a responder")
    parser.add_argument("-k", type=int, default=K, help="Número de documentos a recuperar")
    parser.add_argument("--t", type=int, default=HNSW_SEARCH, help=f"Tipo de mecanismo de búsqueda (HNSW: {HNSW_SEARCH}, DENSA: {DENSE_SEARCH})")
    parser.add_argument("--all", type=int, default=0, help=f"Recuperar todos los chunks de las intervenciones (NO: 0, SI: 1)")
    args = parser.parse_args()

    print_header("BUSQUEDA SEMÁNTICA TOP-K INTERVENCIONES")
    print_info(f"Query: '{args.query}'")
    print_info(f"Tipo de búsqueda: {"HNSW" if args.t == HNSW_SEARCH else "Fuerza Bruta"}")
    print_info(f"K: {args.k}")
    print_info(f"Reconstruir intervenciones recuperando chunks: {"SI" if args.all else "NO"}")


    ### modelo de embedding
    print_step(1, "Cargando modelo de embedding...")
    try:
        embedding_model = SentenceTransformer(MODEL)
        print_success("Modelo cargado correctamente")
    except Exception as e:
        print_error("Error al cargar el modelo de embedding: {e}")


    ### vectorizar query usuario
    print_step(2, f"Vectorizando la pregunta: '{args.query}'")
    try:
        query_vector = get_embedding(embedding_model, args.query)
        print_success("Pregunta vectorizada correctamente")
    except Exception as e:
        print_error(f"Error al vectorizar la pregunta: {e}")


    ### conectarse a MDB
    print_step(3, f"Conectando a MillenniumDB en {URL}...")
    try:
        db = driver(URL)
        print_success(f"Conexión establecida")
    except Exception as e:
        print_error(f"No se pudo conectar: {e}")
        return


    ### recuperar top-k intervenciones
    print_step(4, f"Recuperando las top-{args.k} intervenciones de MillenniumDB...")
    try:
        session = db.session()
        k_interventions = retrieve_k_interventions(session, query_vector, args.k, args.t, args.all)
        session.close()
        print_success("Intervenciones recuperadas con éxito")
        for node_id, distance, context in k_interventions:
            print_success(f"ID[{node_id}]  ~  Distancia: {round(distance, 5)}")
            print_success(f"{Colors.BOLD}{context[:50]} . . .{Colors.END}")
    except Exception as e:
        print_error(f"Error al recuperar intervenciones: {e}")
        return
    

    ### cerrar conexion
    print_step(5, "Cerrando conexión...")
    db.close()
    print_success("Conexión cerrada")


    ### generar respuesta con GPT
    print_step(6, "Generando respuesta con GPT-4o-mini...")
    try:
        answer, system_prompt, user_prompt = generate_answer(args.query, k_interventions)
        print_success("Respuesta generada con éxito")
        print("="*10)
        print_success(f"Prompt sistema: {Colors.BOLD}{system_prompt}{Colors.END}")
        print("="*10)
        print_success(f"Prompt usuario: {Colors.BOLD}{user_prompt}{Colors.END}")
        print("="*10)
        print_success(f"Respuesta GPT: {Colors.BOLD}{answer}{Colors.END}")
        print("="*10)
    except Exception as e:
        print_error(f"Error al generar respuesta: {e}")
        return

    # guardar en CSV para evaluación posterior
    print_step(7, f"Guardando resultados de la ejecución en {CSV_LOG_FILE}")
    try:
        strategy = "RAG_HNSW" if args.t == HNSW_SEARCH else "RAG_DENSO"
        save_log(
            k_interventions=k_interventions,
            query=args.query,
            k=args.k,
            answer=answer,
            strategy=strategy,
            csv_file=CSV_LOG_FILE
        )
        print_success("Resultados guardados exitosamente")
    except Exception as e:
        print_error(f"Error al escribir resultados: {e}")
        return


    print_header("PROCESO COMPLETADO EXITOSAMENTE")

    return 0


if __name__ == "__main__":
    main()
