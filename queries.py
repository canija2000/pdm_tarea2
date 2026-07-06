from config import INDEX_NAME

# Crear indice
CREATE_HNSW_INDEX = f'''
CREATE HNSW INDEX "{INDEX_NAME}" WITH {{
"property"= "value",
"dimension"= 768,
"maxCandidates" = 16,
"maxEdges" = 8,
"metric"= "cosineDistance"
}}
'''

# Contar aristas + nodos
ALL_COUNT = '''
MATCH (?n)
RETURN count(?n)
'''

# Contar nodos
NODE_COUNT = '''
MATCH (?x)
WHERE TYPE(?x) IS NULL
RETURN COUNT(*) as ?count
'''

# Contar aristas
EDGE_COUNT = '''
MATCH (?x)
WHERE TYPE(?x) IS NOT NULL
RETURN COUNT(*) as ?count
'''

# Contar embeddings
EMBEDDING_COUNT = '''
MATCH (?x :Embedding)
RETURN COUNT(*) as ?count
'''

# Busqueda top-k fuerza bruta
# Parametros (?query_embedding)
# Reemplazos (?k)
DENSE_TOP_K_QUERY = '''
LET ?q = ?query_embedding
MATCH (?c :Embedding)
LET ?distance = COSINE_DISTANCE(?q, ?c.value)
ORDER BY ?distance
RETURN ?c, ?c.content as ?content, ?distance
LIMIT ?k
'''

# Busqueda top-k hnsw
# Parametros (?query_embedding)
# Reemplazos (?k)
HNSW_TOP_K_QUERY = f'''
LET ?q = ?query_embedding
CALL HNSW_TOP_K("{INDEX_NAME}", ?q, ?k, 1000)
YIELD ?object AS ?c, ?distance
RETURN ?c, ?c.content AS ?content, ?distance
'''

# Busqueda top-k hnsw + nodos de intervencion
# Parametros (?query_embedding)
# Reemplazos (?k)
HNSW_TOP_K_WITH_INTERVENTION_QUERY = f'''
LET ?q = ?query_embedding
CALL HNSW_TOP_K("{INDEX_NAME}", ?q, ?k, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
RETURN ?c, ?i, ?c.content AS ?content, ?distance
'''

# Obtener intervencion de un chunk
# Parametros (?chunk)
GET_INTERVENTION = '''
LET ?c = ?chunk
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
RETURN ?i
LIMIT 1
'''

# Obtener intervencion de un chunk + data del chunk
# Parametros (?chunk)
GET_CHUNK_WITH_INTERVENTION = '''
LET ?c = ?chunk
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
RETURN ?c, ?i, ?c.content AS ?content
LIMIT 1
'''

# Recuperar todos los chunks de una intervencion
# Parametros (?node)
ALL_CHUNKS = '''
LET ?i = ?node
MATCH (?i)-[:HasEmbedding]->(?chunk :Embedding)
ORDER BY ?chunk
RETURN ?chunk, ?chunk.content as ?content
'''

# Recuperar top-k intervenciones de un partido politico
# Parametros (?query_embedding, ?filtro)
# Reemplazos (?n, ?k)
FROM_A_POLITICAL_PARTY = f'''
LET ?q = ?query_embedding
LET ?party_name = ?filtro
CALL HNSW_TOP_K("{INDEX_NAME}", ?q, ?n, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
MATCH (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?party :PoliticalParty)
WHERE ?party.name == ?party_name
ORDER BY ?distance
RETURN ?c, ?c.content AS ?content, ?party_name, ?distance
LIMIT ?k
'''

# Recuperar top-k intervenciones junto al genero del interlocutor
# Parametros (?query_embedding)
# Reemplazos (?n, ?k)
GENDER_AGG = f'''
LET ?q = ?query_embedding
CALL HNSW_TOP_K("{INDEX_NAME}", ?q, ?n, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
MATCH (?p :Person)-[:ServedAs]->(?pos :Position)<-[:DeliveredBy]-(?i)
ORDER BY ?distance
RETURN ?c, ?c.content AS ?content, ?p.gender AS ?gender, ?distance
LIMIT ?k
'''

# Recuperar top-k intervenciones filtrando por fecha minima
# Parametros (?query_embedding, ?filtro)
# Reemplazos (?n, ?k)
FILTER_MIN_DATE = f'''
LET ?q = ?query_embedding
LET ?min_date = ?filtro
CALL HNSW_TOP_K("{INDEX_NAME}", ?q, ?n, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
MATCH (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)
WHERE ?s.date > ?min_date
ORDER BY ?distance
RETURN ?c, ?c.content AS ?content, ?s.date AS ?date, ?distance
LIMIT ?k
'''