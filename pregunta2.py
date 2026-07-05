import argparse
import os
import csv
import uuid
from datetime import datetime
from millenniumdb_driver import driver
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

print("Cargando modelo de embeddings...")
embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')

# gtp-key 
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# embedding de la query 
def get_embedding(text: str) -> list[float]:
    prefixed_text = f"query: {text}"
    embedding = embedding_model.encode(prefixed_text, normalize_embeddings=True)
    return embedding.tolist()


HNSW_INDEX_NAME = "hnsw_vectors"

def conect_mdb():
    url = "ws://localhost:1234"
    db = driver(url)
    return db

def retrieve_from_millennium_graphrag(query_vector: list[float], patron: int, filtro: str, k: int = 5) -> list[tuple[str, str, float]]:
    mdb = conect_mdb()
    vector_str = str(query_vector)
    hnsw_candidates = max(k * 200, 10000)
    
    if patron == 1:
        # Frente 1: Restricción tipada (ej: Solo intervenciones de un partido específico)
        mql_query = f"""
        LET ?q = tensorFloat("{vector_str}")
        CALL HNSW_TOP_K("{HNSW_INDEX_NAME}", ?q, {hnsw_candidates}, 1000)
        YIELD ?object AS ?c, ?dist
        MATCH (?i:Intervention)-[:HasEmbedding]->(?c)
        MATCH (?i)-[:DeliveredBy]->(?pos:Position)-[:Represents]->(?party:PoliticalParty {{name: "{filtro}"}})
        ORDER BY ?dist ASC
        RETURN ?c.content, "{filtro}", ?dist
        LIMIT {k}
        """
    elif patron == 2:
        # Frente 2: Agregación / Contraste (ej: Recuperamos el género de la persona para contrastar)
        mql_query = f"""
        LET ?q = tensorFloat("{vector_str}")
        CALL HNSW_TOP_K("{HNSW_INDEX_NAME}", ?q, {hnsw_candidates}, 1000)
        YIELD ?object AS ?c, ?dist
        MATCH (?i:Intervention)-[:HasEmbedding]->(?c)
        MATCH (?p:Person)-[:ServedAs]->(?pos:Position)<-[:DeliveredBy]-(?i)
        ORDER BY ?dist ASC
        RETURN ?c.content, ?p.gender, ?dist
        LIMIT {k}
        """

    elif patron == 3:
        # Frente 3: Filtro numérico o temporal (ej: Intervenciones de sesiones posteriores a una fecha)
        # HNSW trae candidatos por similitud; luego el MATCH conserva solo los que cumplen el patron temporal.
        mql_query = f"""
        LET ?q = tensorFloat("{vector_str}")
        CALL HNSW_TOP_K("{HNSW_INDEX_NAME}", ?q, {hnsw_candidates}, 1000)
        YIELD ?object AS ?c, ?dist
        MATCH (?i:Intervention)-[:HasEmbedding]->(?c)
        MATCH (?i)-[:HasIntervention]->(?proc:Procedure)-[:OCCURRED_IN]->(?s:Session)
        WHERE ?s.date > date("{filtro}") 
        ORDER BY ?dist ASC
        RETURN ?c.content, ?s.date, ?dist
        LIMIT {k}
        """
    else:
        raise ValueError("Patrón inválido.")

    interventions = []
    try:
        with mdb.session() as session:
            result = session.run(mql_query)
            for record in result:
                vals = list(record.values()) if hasattr(record, 'values') else list(record)
                if vals and vals[0] is not None:
                    texto = vals[0]
                    # La metadata extra (partido, género, fecha, etc.) la ponemos como string
                    metadata = str(vals[1]) if len(vals) > 1 else ""
                    distancia = vals[2] if len(vals) > 2 else 0.0
                    interventions.append((texto, metadata, distancia))
    except Exception as e:
        print(f"Error consultando a MillenniumDB (GraphRAG): {e}")
        
    return interventions

def generate_answer(query: str, patron: int, context_documents: list[tuple[str, str, float]]) -> str:
    # Formatear el contexto para que incluya la metadata (género, partido, fecha) dependiendo del patrón
    textos = []
    for doc, metadata, dist in context_documents:
        if patron == 1:
            textos.append(f"[{metadata}] {doc}")
        elif patron == 2:
            textos.append(f"[Atributo a contrastar: {metadata}] {doc}")
        elif patron == 3:
            textos.append(f"[Fecha: {metadata}] {doc}")

    context_text = "\n\n---\n\n".join(textos)
    
    # Personalizar el prompt según el objetivo
    if patron == 1:
        instruccion_extra = "Tu respuesta debe resumir lo que opinan basado ÚNICAMENTE en las intervenciones entregadas."
    elif patron == 2:
        instruccion_extra = "Tu respuesta debe CONTRASTAR claramente las distintas posturas encontradas en el contexto basándote en el atributo proporcionado (por ejemplo, hombres vs mujeres, partido A vs partido B)."
    elif patron == 3:
        instruccion_extra = "Tu respuesta debe tomar en consideración la fecha o atributo numérico asociado a cada intervención para ver cómo ha evolucionado el tema."

    prompt = f"""
    Eres un asistente experto en analizar intervenciones del poder legislativo chileno.
    Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado.
    {instruccion_extra}
    
    Contexto:
    {context_text}

    Pregunta: {query}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente riguroso, imparcial y analítico."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Flujo de GraphRAG (Parte 2)")
    parser.add_argument("query", type=str, help="Pregunta a responder")
    parser.add_argument("--patron", type=int, required=True, choices=[1, 2, 3], help="El frente a usar: 1(Tipado), 2(Contraste), 3(Numérico/Temporal)")
    parser.add_argument("--filtro", type=str, default="2020-01-01", help="Valor del filtro: Nombre de partido (para el patrón 1) o fecha (para el patrón 3)")
    parser.add_argument("--k", type=int, default=5, help="Número de documentos a recuperar")
    args = parser.parse_args()

    print(f"\n[1] Vectorizando la pregunta: '{args.query}'")
    query_vector = get_embedding(args.query)

    print(f"\n[2] Ejecutando GraphRAG con Patrón {args.patron} en MillenniumDB...")
    context = retrieve_from_millennium_graphrag(query_vector, patron=args.patron, filtro=args.filtro, k=args.k)
    
    if not context:
        print("No se recuperaron documentos. Revisa tu consulta MQL o los nombres de las etiquetas en tu BD.")
        return

    print("\nContexto recuperado:")
    for i, (doc, metadata, dist) in enumerate(context):
        print(f" - Doc {i+1} [Meta: {metadata} | Dist: {dist:.4f}]: {doc[:80]}...")

    print("\n[3] Generando respuesta analítica con GPT-4o-mini...")
    answer = generate_answer(args.query, args.patron, context)

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
        
        avg_dist = sum(dist for _, _, dist in context) / len(context) if context else 0.0
        context_str = "\n".join([f"[Meta: {meta} | Dist: {dist:.4f}] {doc}" for doc, meta, dist in context])
        
        estrategia_nombre = f"GraphRAG_Patron{args.patron}"
        
        writer.writerow([
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            estrategia_nombre,
            args.query,
            args.k,
            f"{avg_dist:.4f}",
            context_str,
            answer
        ])
    print(f"[+] Resultado guardado exitosamente en {csv_file}")

if __name__ == "__main__":
    main()
