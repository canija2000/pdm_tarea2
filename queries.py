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
CALL HNSW_TOP_K("mi_indice", ?q, ?k, 1000)
YIELD ?object AS ?c, ?distance
RETURN ?c, ?c.content AS ?content, ?distance
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

# Parametros (?intervention)
ALL_CHUNKS = '''
LET ?i = ?node
MATCH (?i)-[:HasEmbedding]->(?chunk :Embedding)
ORDER BY ?chunk
RETURN ?chunk, ?chunk.content as ?content
'''
