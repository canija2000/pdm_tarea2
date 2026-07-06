CREATE_HNSW_INDEX = '''
CREATE HNSW INDEX "mi_indice" WITH {
"property"= "value",
"dimension"= 768,
"maxCandidates" = 16,
"maxEdges" = 8,
"metric"= "cosineDistance"
}
'''

ALL_COUNT = '''
MATCH (?n)
RETURN count(?n)
'''

NODE_COUNT = '''
MATCH (?x)
WHERE TYPE(?x) IS NULL
RETURN COUNT(*) as ?count
'''

EDGE_COUNT = '''
MATCH (?x)
WHERE TYPE(?x) IS NOT NULL
RETURN COUNT(*) as ?count
'''

EMBEDDING_COUNT = '''
MATCH (?x :Embedding)
RETURN COUNT(*) as ?count
'''

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

# Parametros (?query_embedding)
# Reemplazos (?k)
HNSW_TOP_K_QUERY = '''
LET ?q = ?query_embedding
CALL HNSW_TOP_K("hnsw_vectors", ?q, ?k, 1000)
YIELD ?object AS ?c, ?distance
RETURN ?c, ?c.content AS ?content, ?distance
'''

# Parametros (?query_embedding)
# Reemplazos (?k)
HNSW_TOP_K_WITH_INTERVENTION_QUERY = '''
LET ?q = ?query_embedding
CALL HNSW_TOP_K("hnsw_vectors", ?q, ?k, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
RETURN ?c, ?i, ?c.content AS ?content, ?distance
'''

# Parametros (?chunk)
GET_CHUNK_WITH_INTERVENTION = '''
LET ?c = ?chunk
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
RETURN ?c, ?i, ?c.content AS ?content
LIMIT 1
'''

# Parametros (?chunk)
GET_INTERVENTION = '''
LET ?c = ?chunk
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
RETURN ?i
LIMIT 1
'''


# Parametros (?query_embedding)
# Reemplazos (?k)
# TOP_K_QUERY = '''
# LET ?q = ?query_embedding
# MATCH (?chunk_aux :Embedding)<-[:HasEmbedding]-(?i :Intervention)-[:HasEmbedding]->(?chunk :Embedding)
# LET ?d = COSINE_DISTANCE(?q, ?chunk.value)
# LET ?d_aux = COSINE_DISTANCE(?q, ?chunk_aux.value)
# GROUP BY ?i, ?chunk, ?d
# HAVING ?min_distance == ?d
# RETURN ?i, MIN(?d_aux) AS ?min_distance, ?chunk, ?d
# LIMIT ?k
# '''

# Parametros (?node)
ALL_CHUNKS = '''
LET ?i = ?node
MATCH (?i)-[:HasEmbedding]->(?chunk :Embedding)
ORDER BY ?chunk
RETURN ?chunk, ?chunk.content as ?content
'''

# Parametros (?query_embedding, ?filtro)
# Reemplazos (?n, ?k)
FROM_A_POLITICAL_PARTY = """
LET ?q = ?query_embedding
LET ?party_name = ?filtro
CALL HNSW_TOP_K("hnsw_vectors", ?q, ?n, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
MATCH (?i)-[:DeliveredBy]->(?pos :Position)-[:Represents]->(?party :PoliticalParty)
WHERE ?party.name == ?party_name
ORDER BY ?distance
RETURN ?c, ?c.content AS ?content, ?party_name, ?distance
LIMIT ?k
"""

# Parametros (?query_embedding)
# Reemplazos (?n, ?k)
GENDER_AGG = """
LET ?q = ?query_embedding
CALL HNSW_TOP_K("hnsw_vectors", ?q, ?n, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
MATCH (?p :Person)-[:ServedAs]->(?pos :Position)<-[:DeliveredBy]-(?i)
ORDER BY ?gender, ?distance
RETURN ?c, ?c.content AS ?content, ?p.gender AS ?gender, ?distance
LIMIT ?k
"""

# Parametros (?query_embedding, ?filtro)
# Reemplazos (?n, ?k)
FILTER_MIN_DATE = """
LET ?q = ?query_embedding
LET ?min_date = ?filtro
CALL HNSW_TOP_K("mi_indice", ?q, ?n, 1000)
YIELD ?object AS ?c, ?distance
MATCH (?i :Intervention)-[:HasEmbedding]->(?c)
MATCH (?i)-[:HasIntervention]->(?proc :Procedure)-[:OCCURRED_IN]->(?s :Session)
WHERE ?s.date > DATE(?min_date)
ORDER BY ?distance
RETURN ?c, ?c.content AS ?content, ?s.date AS ?date, ?distance
LIMIT ?k
"""