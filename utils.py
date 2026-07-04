from typing import Any
import re
import millenniumdb_driver as mdb
# from millenniumdb_driver.millenniumdb_error import Result

# Validar strings
def valid_str(s: str) -> bool:
    return s.isidentifier()

# Validar integers
def valid_int(s: str) -> bool:
    return bool(re.match(r"^[0]$", s) or re.match(r"^[1-9][0-9]*$", s))

# Validar floats
def valid_float(s: str) -> bool:
    return bool(re.match(r"^[0]\.[0-9]+$", s) or re.match(r"^[1-9][0-9]*\.[0-9]+$", s))

# Ejecuar una query de forma segura (anti-injection)
def q_run(
        session, query: str, parameters: dict[str, Any] = None, replaces: dict[str, str|int|float] = None, timeout: float = 0
) -> Any:
    if replaces:
        # Ordenamos de mayor a menor longitud de la clave para evitar reemplazos parciales maliciosos o accidentales
        sorted_items = sorted(replaces.items(), key=lambda x: len(str(x[0])), reverse=True)

        for k, v in sorted_items:
            str_k = str(k)
            str_v = str(v).strip() # Un .strip() extra por seguridad con espacios fantasmas

            if not valid_str(str_k):
                raise Exception("Clave de reemplazo inválida.")

            valid_v = valid_str(str_v) or valid_int(str_v) or valid_float(str_v)
            if not valid_v:
                raise Exception("Valor de reemplazo inválido (Posible Inyección).")
            
            query = query.replace(f"?{str_k}", str_v)

    return session.run(query, parameters, timeout)

def get_embedding(embedding_model, text: str) -> list[float]:
    prefixed_text = f"query: {text}"
    embedding = embedding_model.encode(prefixed_text, normalize_embeddings=True, convert_to_numpy=True)
    return embedding