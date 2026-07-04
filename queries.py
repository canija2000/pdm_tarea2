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
TOP_K_QUERY = '''
LET ?q = ?query_embedding
MATCH (?i :Intervention)-[:HasEmbedding]->(?chunk :Embedding)
LET ?d = COSINE_DISTANCE(?q, ?chunk.value)
ORDER BY ?d
RETURN DISTINCT ?i
LIMIT ?k
'''

# Parametros (?intervention)
ALL_CHUNKS = '''
LET ?i = ?node
MATCH (?i)-[:HasEmbedding]->(?chunk :Embedding)
ORDER BY ?chunk
RETURN ?chunk, ?chunk.content as ?content
'''