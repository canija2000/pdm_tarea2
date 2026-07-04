from millenniumdb_driver import driver
from sentence_transformers import SentenceTransformer

def get_embedding(text: str) -> list[float]:
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')
    prefixed_text = f"query: {text}"
    embedding = embedding_model.encode(prefixed_text, normalize_embeddings=True)
    return embedding.tolist()

def main():
    url = "ws://localhost:1234"
    db = driver(url)
    
    vector = get_embedding("sueldo minimo")
    vector_str = str(vector)
    
    query_exact = f"""
    LET ?q = tensorFloat("{vector_str}")
    MATCH (?i:Intervention)-[:DeliveredBy]->(?pos:Position)-[:Represents]->(?party:PoliticalParty {{name: "Partido por la Democracia"}})
    MATCH (?i)-[:HasEmbedding]->(?c)
    LET ?dist = COSINE_DISTANCE(?q, ?c.value)
    ORDER BY ?dist ASC
    RETURN ?c.content, ?dist
    LIMIT 2
    """
    
    query_hnsw = f"""
    CALL HNSW_TOP_K("hnsw_vectors", tensorFloat("{vector_str}"), 10000, 1000)
         YIELD ?object AS ?c, ?distance
    MATCH (?i:Intervention)-[:HasEmbedding]->(?c)
    MATCH (?i)-[:DeliveredBy]->(?pos:Position)-[:Represents]->(?party:PoliticalParty {{name: "Partido por la Democracia"}})
    RETURN count(?c)
    """
    
    try:
        with db.session() as session:
            print("--- EXACT DISTANCE (TOP 2) ---")
            res_exact = session.run(query_exact)
            for r in res_exact:
                print(list(r.values()) if hasattr(r, 'values') else list(r))
                
            print("--- HNSW WITH K=10000 COUNT ---")
            res_hnsw = session.run(query_hnsw)
            for r in res_hnsw:
                print(list(r.values()) if hasattr(r, 'values') else list(r))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
