from config import URL
from utils import q_run
from color_print import Colors, print_header, print_step, print_success, print_error, print_info
import queries as q

from millenniumdb_driver import driver


def main():
    print_header("INICIALIZANDO BASE DE DATOS")


    ### conectarse a la mdb
    print_step(1, f"Conectando a MillenniumDB en {URL}...")
    try:
        db = driver(URL)
        print_success(f"Conexión establecida")
    except Exception as e:
        print_error(f"No se pudo conectar: {e}")
        return
    

    ### crear indice hnsw
    print_step(2, "Creando índice HNSW...")
    try:
        session = db.session()
        result = q_run(session, q.CREATE_HNSW_INDEX)
        session.close()
        print_success("Índice HNSW creado correctamente")
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg:
            print_info("El índice HNSW ya existe en la base de datos")
        else:
            print_error(f"Error al crear índice: {error_msg}")
    

    ### testeamos con una query simple
    print_step(3, f"Ejecutando consulta simple: {q.ALL_COUNT}")
    try:
        session = db.session()
        result = q_run(session, q.ALL_COUNT)
        all_count = result.values()[0][0]
        print_success(f"Resultado consulta: {Colors.BOLD}{all_count}{Colors.END}")
    except Exception as e:
        print_error(f"Error al ejecutar: {e}")


    ### contar nodos
    print_step(4, "Contando nodos en la base de datos...")
    try:
        session = db.session()
        result = q_run(session, q.NODE_COUNT)
        session.close()
        node_count = result.values()[0][0]
        print_success(f"Nodos totales: {Colors.BOLD}{node_count}{Colors.END}")
    except Exception as e:
        print_error(f"Error al contar nodos: {e}")


    ### contar aristas
    print_step(5, "Contando aristas en la base de datos...")
    try:
        session = db.session()
        result = q_run(session, q.EDGE_COUNT)
        session.close()
        edge_count = result.values()[0][0]
        print_success(f"Aristas totales: {Colors.BOLD}{edge_count}{Colors.END}")
    except Exception as e:
        print_error(f"Error al contar aristas: {e}")


    ### contar embeddings
    print_step(6, "Contando embeddings en la base de datos...")
    try:
        session = db.session()
        result = q_run(session, q.EMBEDDING_COUNT)
        session.close()
        embedding_count = result.values()[0][0]
        print_success(f"Embeddings totales: {Colors.BOLD}{embedding_count}{Colors.END}")
    except Exception as e:
        print_error(f"Error al contar embeddings: {e}")


    ### cerrar conexion
    print_step(7, "Cerrando conexión...")
    db.close()
    print_success("Conexión cerrada")


    print_header("PROCESO COMPLETADO EXITOSAMENTE")
    
    return 0


if __name__ == "__main__":
    main()