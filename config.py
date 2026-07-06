# SERVER URL
URL = 'ws://localhost:1234'

# NOMBRE INDICE HNSW
INDEX_NAME = "mi_indice"

# ENTERO PARA BUSQUEDA TOP-K POR SIMILITUD COSENO 
K = 5

# EMBEDDING MODEL
MODEL = 'intfloat/multilingual-e5-base'

# PROMPT DE SISTEMA
SYSTEM_PROMPT = """
Eres un asistente experto en analizar intervenciones del poder legislativo chileno.
Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si el contexto no contiene la respuesta, di que no tienes suficiente información.
Cita las intervenciones que uses.
"""