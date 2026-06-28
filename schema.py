from millenniumdb_driver import driver

def main():
    url = "ws://localhost:1234"
    db = driver(url)

    print("--- DESCUBRIENDO EL ESQUEMA DE LA BASE DE DATOS ---")

    try:
        with db.session() as session:
            # 1. Obtenemos una lista con TODOS los tipos de aristas en el grafo
            query_edges = "MATCH (?x)-[?e]->(?y) RETURN DISTINCT TYPE(?e) LIMIT 50"
            result_edges = session.run(query_edges)

            # Extraemos los nombres
            edge_types = []
            for record in result_edges:
                vals = list(record.values()) if hasattr(record, 'values') else list(record)
                edge_types.append(vals[0])

            print(f"\nSe encontraron {len(edge_types)} tipos de aristas distintas.\n")

            # 2. Por cada arista, preguntamos qué etiqueta (label) tiene el nodo origen y el destino
            for edge in edge_types:
                query_nodes = f"MATCH (?origen)-[:{edge}]->(?destino) RETURN labels(?origen), labels(?destino) LIMIT 1"
                try:
                    res_nodes = session.run(query_nodes)
                    for rec in res_nodes:
                        vals = list(rec.values()) if hasattr(rec, 'values') else list(rec)
                        origen = vals[0][0] if vals[0] else "Desconocido"
                        destino = vals[1][0] if vals[1] else "Desconocido"

                        # Imprimimos la relación encontrada: Origen -> [Arista] -> Destino
                        print(f"({origen}) -[:{edge}]-> ({destino})")
                except Exception as e:
                    print(f"Error al inspeccionar [:{edge}]: {e}")

    except Exception as e:
        print(f"Error de conexión a MillenniumDB: {e}")

if __name__ == "__main__":
    main()