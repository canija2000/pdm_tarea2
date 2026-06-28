from millenniumdb_driver import driver

def main():
    url = "ws://localhost:1234"
    db = driver(url)
    
    queries = [
        "MATCH (?i:Intervention)-[:HasIntervention]->(?y) RETURN labels(?y) LIMIT 2",
        "MATCH (?x)-[:IsDeliveredBy]->(?y) RETURN labels(?x), labels(?y) LIMIT 2",
        "MATCH (?x)-[:IsAdressedBy]->(?y) RETURN labels(?x), labels(?y) LIMIT 2"
    ]
    
    try:
        with db.session() as session:
            for i, q in enumerate(queries):
                try:
                    result = session.run(q)
                    print(f"Query {i+1} result:")
                    for record in result:
                        print(list(record.values()) if hasattr(record, 'values') else list(record))
                except Exception as e:
                    print(f"Error in query {i+1}: {e}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
