import argparse
import os
import csv
import uuid
from datetime import datetime
from dotenv import load_dotenv
from millenniumdb_driver import driver
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

 
print("Cargando modelo de embeddings...")
embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')

load_dotenv()

# gtp-key 
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# embedding de la query 
def get_embedding(text: str) -> list[float]:
    prefixed_text = f"query: {text}"
    embedding = embedding_model.encode(prefixed_text, normalize_embeddings=True, convert_to_numpy=True)
    return embedding


def conect_mdb():
    url = "ws://localhost:1234"
    db = driver(url)
    return db

# pasamos el vector/tensor a millenium 
def retrieve_from_millennium(query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:

    mdb = conect_mdb()
    
    # Consulta MQL para buscar vectores similares.
    # Formateamos el vector para MQL (convierte la lista de Python a un string de array)
    
    
    # Consulta usando HNSW_TOP_K
    # ?c es el nodo Chunk que tiene la propiedad 'value' (el vector) y 'content' (el texto)
    # mql_query = f"""
    # CALL HNSW_TOP_K("hnsw_vectors", tensorFloat("{vector_str}"), {k}, 1000)
    #      YIELD ?object AS ?c, ?distance
    # RETURN STR(?c), ?c.content, ?distance
    # """

    mql_qery = f"""
    LET ?q = ?query_embedding
    MATCH (?i :Intervention)-[:HasEmbedding]->(?chunk :Embedding)
    LET ?d = COSINE_DISTANCE(?q, ?chunk.value)
    ORDER BY ?d
    RETURN DISTINCT ?i
    LIMIT 10
"""
    
    ALL_CHUNKS ="""
                    LET ?i = ?node
                    MATCH (?i)-[:HasEmbedding]->(?chunk :Embedding)
                    ORDER BY ?chunk
                    RETURN ?chunk, ?chunk.content as ?content
                """
                
    interventions = []
    i = 0
    try:
        with mdb.session() as session:
            result = session.run(mql_qery, parameters={"query_embedding": query_vector})
            for record in result:
                print(record.values())
                texto_chunks = session.run(ALL_CHUNKS, parameters={"node": record.values()[0]})

                with open('chunks_testeo.txt', 'a', encoding='utf-8') as f:
                    f.write(f"documento {i}:\n")
                    for chunk_record in texto_chunks:
                        f.write(f"{chunk_record.values()[0]}\n")
                        f.write(f"{chunk_record.values()[1]}\n")
                i += 1
        
                # El texto está en ?c.content
                # Extraemos el contenido de forma segura independientemente de la versión del driver
                vals = list(record.values()) if hasattr(record, 'values') else list(record)
                if vals and vals[0] is not None:
                    texto = vals[0]
                    distancia = vals[1] if len(vals) > 1 else 0.0
                    interventions.append((texto, distancia))
                else:
                    interventions.append(("Contenido no encontrado", 0.0))
    except Exception as e:
        print(f"Error consultando a MillenniumDB: {e}")
        
    return interventions

def generate_answer(query: str, context_documents: list[tuple[str, float]]) -> str:
    # Crear el prompt con el contexto
    textos = [doc for doc, dist in context_documents]
    context_text = "\n\n---\n\n".join(textos)
    prompt = f"""

    Contexto:
    {context_text}

    Pregunta: {query}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             """Eres un asistente experto en analizar intervenciones del poder legislativo chileno.
    Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado.
    Si el contexto no contiene la respuesta, di que no tienes suficiente información.
    Cita las intervenciones que uses."""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def main():
    # formato esperado de ejecucion : 
    # python pregunta1.py "pregunta a responder" --k [numero maximo de documentos a recuperar]
    parser = argparse.ArgumentParser(description="Flujo de RAG Denso (Parte 1)")
    parser.add_argument("query", type=str, help="Pregunta a responder")
    parser.add_argument("-t", type= int, default=0, help = "Tipo de mecanismo de recuperación de chunks (0: HNSW_TOP_K, 1:  ALL_CHUNKS)")
    parser.add_argument("--k", type=int, default=5, help="Número de documentos a recuperar")
    args = parser.parse_args()

    print(f"\n[1] Vectorizando la pregunta: '{args.query}'")
    query_vector = get_embedding(args.query)

    print(f"\n[2] Recuperando las top-{args.k} intervenciones de MillenniumDB...")
    context = retrieve_from_millennium(query_vector, k=args.k)
    
    if not context:
        print("No se recuperaron documentos. Revisa tu consulta MQL o conexión a BD.")
        return

    print("\nContexto recuperado:")
    for i, (doc, dist) in enumerate(context):
        print(f" * Doc {i+1} [Distancia: {dist:.4f}]: {doc[:50]}...")

    print("\n[3] Generando respuesta con GPT-4o-mini...")
    answer = generate_answer(args.query, context)

    print("\n=== RESPUESTA ===")
    print(answer)
    print("=================\n")

    # Guardar en CSV para evaluación posterior
    csv_file = "evaluacion_resultados.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id_ejecucion", "fecha", "estrategia", "query", "k", "distancia_promedio", "contexto_recuperado", "respuesta_llm"])
        
        avg_dist = sum(dist for _, dist in context) / len(context) if context else 0.0
        context_str = "\n".join([f"[{dist:.4f}] {doc}" for doc, dist in context])
        
        writer.writerow([
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "RAG_Denso",
            args.query,
            args.k,
            f"{avg_dist:.4f}",
            context_str,
            answer
        ])
    print(f"[+] Resultado guardado exitosamente en {csv_file}")

if __name__ == "__main__":
    main()
