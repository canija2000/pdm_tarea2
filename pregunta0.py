from millenniumdb_driver import driver

def main():
    # conectarse a la mdb
    url = "ws://localhost:1234"
    db = driver(url)

    # index hnsw: 
    create_index = """
    CREATE HNSW INDEX "mi_indice" WITH {
    "property"= "value",
    "dimension"= 768,
    "maxCandidates" = 16,
    "maxEdges" = 8,
    "metric"= "cosineDistance"
    }
    """
    

    # testeamos con una query simple. 
    query = "MATCH (?n) RETURN count(?n)"
    
    print("Conectando a MillenniumDB en", url)
    print("Ejecutando consulta:", query)
    
    try:
        with db.session() as session:
            session.run(create_index) 
            result = session.run(query)
            print("\nResultado:")
            for record in result:
                print(record)
    except Exception as e:
        print("\nError al ejecutar:", e)

if __name__ == "__main__":
    main()