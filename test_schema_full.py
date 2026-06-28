from millenniumdb_driver import driver

def main():
    url = "ws://localhost:1234"
    db = driver(url)
    
    queries = {
        "OCCURRED_IN": "MATCH (?x)-[:OCCURRED_IN]->(?y) RETURN labels(?x), labels(?y) LIMIT 1",
        "Involves": "MATCH (?x)-[:Involves]->(?y) RETURN labels(?x), labels(?y) LIMIT 1",
        "Adressed": "MATCH (?x)-[:Adressed]->(?y) RETURN labels(?x), labels(?y) LIMIT 1",
        "InStage": "MATCH (?x)-[:InStage]->(?y) RETURN labels(?x), labels(?y) LIMIT 1",
    }
    
    try:
        with db.session() as session:
            for name, q in queries.items():
                print(f"--- {name} ---")
                try:
                    result = session.run(q)
                    for record in result:
                        print(list(record.values()) if hasattr(record, 'values') else list(record))
                except Exception as e:
                    print(f"Error in {name}: {e}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
