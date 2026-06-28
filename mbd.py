from millenniumdb_driver import driver

def main():
    # 1. Conectarse al servidor por WebSockets (ws://)
    url = "ws://localhost:1234"
    db = driver(url)
    
    # 2. Iniciar sesión y ejecutar la consulta (MQL usa el ? para variables)
    query = "MATCH (?n) RETURN count(?n)"
    
    print("Conectando a MillenniumDB en", url)
    print("Ejecutando consulta:", query)
    
    try:
        with db.session() as session:
            result = session.run(query)
            print("\nResultado:")
            for record in result:
                print(record)
    except Exception as e:
        print("\nError al ejecutar:", e)

if __name__ == "__main__":
    main()