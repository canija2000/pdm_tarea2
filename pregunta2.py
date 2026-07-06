from config import URL, INDEX_NAME, K, MODEL, SYSTEM_PROMPT
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

# Frente 1: Restricción tipada (ej: Solo intervenciones de un partido específico)
RESTR_TIPADA = 1
 # Frente 2: Agregación / Contraste (ej: Recuperamos el género de la persona para contrastar)
AGREGACION = 2
# Frente 3: Filtro numérico o temporal (ej: Intervenciones de sesiones posteriores a una fecha)
# HNSW trae candidatos por similitud; luego el MATCH conserva solo los que cumplen el patron temporal.
FILTRO_TEMP = 3

# Candidatos por intervencion recuperada
N_PER_INTERVENTION = 200
# Limite estricto de total de candidatos
HARD_LIMIT = 10000

# path para registrar ejecuciones
CSV_LOG_FILE = "output/evaluacion_resultados.csv"



### Cliente OpenAI

# Cargar variables de entorno desde .env
load_dotenv()

# gtp-client (leer api key en .env)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



### Funciones Auxiliares

# Recuperar top-k intervenciones
def retrieve_k_interventions_graphrag(
        session,
        query_vector: list[float], 
        patron: int, 
        filtro: str, 
        k: int = K
) -> list[tuple[str, float, str, str]]:
    
    query = q.FROM_A_POLITICAL_PARTY if patron == RESTR_TIPADA else \
            q.GENDER_AGG if patron == AGREGACION else \
            q.FILTER_MIN_DATE if patron == FILTRO_TEMP else \
            None
    
    metadata_name = "party_name" if patron == RESTR_TIPADA else \
                    "gender" if patron == AGREGACION else \
                    "date" if patron == FILTRO_TEMP else \
                    None
    
    hnsw_candidates = max(k * N_PER_INTERVENTION, HARD_LIMIT)
    result = q_run(
        session, 
        query,
        parameters={"query_embedding": query_vector, "filtro": filtro},
        replaces={"n": hnsw_candidates, "k": k}
    )

    interventions = []
    for record in result:
        chunk = record.get('c')
        content = record.get('content')
        distance = record.get('distance')
        metadata = None
        if metadata_name:
            metadata = str(record.get(metadata_name))

        tmp_result = q_run(session, q.GET_INTERVENTION, parameters={"chunk": chunk})
        tmp_record = tmp_result.records()[0]

        intervention = tmp_record.get('i')
        intervention_id = str(intervention)

        interventions.append((intervention_id, distance, metadata, content)) 
        
    return interventions


# Generar respuesta con GPT
def generate_answer(
        query: str, 
        patron: int, 
        context_documents: list[tuple[str, float, str, str]]
) -> tuple[str, str, str]:
    # Formatear el contexto para que incluya la metadata (género, partido, fecha) dependiendo del patrón
    textos = []
    for doc_id, dist, metadata, doc in context_documents:
        if patron == RESTR_TIPADA:
            textos.append(f"ID Documento [{doc_id}]\n" + f"[Partido Politico: {metadata}]\n" + doc)
        elif patron == AGREGACION:
            textos.append(f"ID Documento [{doc_id}]\n" + f"[Género Persona: {metadata}]\n" + doc)
        elif patron == FILTRO_TEMP:
            textos.append(f"ID Documento [{doc_id}]\n" + f"[Fecha: {metadata}]\n" + doc)
    context_text = "\n\n---\n\n".join(textos)

    # Personalizar el prompt según el objetivo
    if patron == RESTR_TIPADA:
        instruccion_extra = "Tu respuesta debe resumir lo que opina el partido político señalado al inicio de cada fragmento de texto."
    elif patron == AGREGACION:
        instruccion_extra = "Tu respuesta debe CONTRASTAR claramente las distintas posturas encontradas en el contexto basándote en el género del interlocutor señalado al inicio de cada fragmento de texto."
    elif patron == FILTRO_TEMP:
        instruccion_extra = "Tu respuesta debe tomar en consideración la fecha asociada a cada intervención para ver cómo ha evolucionado el tema."

    system_prompt = (
        f"{SYSTEM_PROMPT}"
        " "
        f"{instruccion_extra}"
    )
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content, system_prompt, prompt


# Guardar log con resultados de ejecucion
def save_log(k_interventions, query, k, answer, strategy, csv_file = CSV_LOG_FILE):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id_ejecucion", "fecha", "estrategia", "query", "k", "distancia_promedio", "contexto_recuperado", "respuesta_llm"])
        
        avg_dist = sum(dist for _, dist, _, _ in k_interventions) / len(k_interventions) if k_interventions else 0.0
        context_str = "\n".join([f"ID[{doc_id}] DIST[{dist:.4f}] META[{metadata}] \n{doc_text}" for doc_id, dist, metadata, doc_text in k_interventions])

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
    # python pregunta1.py "pregunta a responder" -k --patron --filtro
    # -k: numero de documentos a recuperar
    # -patron: tipo de patron busqueda para hacer match en el grafo (1: Restriccion tipada, 2: Agregacion/comparacion, 3: Por atributo numerico/temporal)
    # --filtro: filtro en funcion del tipo de patron
    parser = argparse.ArgumentParser(description="Flujo de GraphRAG (Parte 2)")
    parser.add_argument("query", type=str, help="Pregunta a responder")
    parser.add_argument("-k", type=int, default=K, help="Número de documentos a recuperar")
    parser.add_argument("--patron", type=int, required=True, choices=[RESTR_TIPADA, AGREGACION, FILTRO_TEMP], help=f"El frente a usar: {RESTR_TIPADA}(Tipado), {AGREGACION}(Contraste), {FILTRO_TEMP}(Numérico/Temporal)")
    parser.add_argument("--filtro", type=str, default="2020-01-01", help="Valor del filtro: Nombre de partido (para el patrón 1) o fecha (para el patrón 3)")
    args = parser.parse_args()

    print_header("BUSQUEDA SEMÁNTICA TOP-K INTERVENCIONES")
    print_info(f"Query: '{args.query}'")
    print_info(f"Patrón de búsqueda: {"Restricción Tipada" if args.patron == RESTR_TIPADA else "Contraste" if args.patron == AGREGACION else "Numérico/Temporal"}")
    print_info(f"Filtro: {args.filtro}")
    print_info(f"K: {args.k}")


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
        k_interventions = retrieve_k_interventions_graphrag(session, query_vector, args.patron, args.filtro, args.k)
        session.close()
        print_success("Intervenciones recuperadas con éxito")
        for node_id, distance, metadata, context in k_interventions:
            print_success(f"ID[{node_id}]  ~  Distancia: {round(distance, 5)} ~ Metadata: {metadata}")
            print_success(f"{Colors.BOLD}{context[:50]} . . .{Colors.END}")
    except Exception as e:
        print_error(f"Error al recuperar intervenciones: {e}")
        return
    

    # cerrar conexion
    print_step(5, "Cerrando conexión...")
    db.close()
    print_success("Conexión cerrada")


    ### generar respuesta con GPT
    print_step(6, "Generando respuesta analítica con GPT-4o-mini...")
    try:
        answer, system_prompt, user_prompt = generate_answer(args.query, args.patron, k_interventions)
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
        strategy = f"GraphRAG_Patron[{args.patron}]"
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
