# SERVER URL
URL = 'ws://localhost:1234'

# API KEY
API_KEY = 'api-key'

# EMBEDDING MODEL
MODEL = 'intfloat/multilingual-e5-base'

# ENTERO PARA TOP-K HNSW
K = 5

# PROMPT DE SISTEMA
SYSTEM_PROMPT = """
Eres un asistente experto en analizar intervenciones del poder legislativo chileno.
Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si el contexto no contiene la respuesta, di que no tienes suficiente información.
Cita las intervenciones que uses.
"""