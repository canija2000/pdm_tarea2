from millenniumdb_driver import driver

def main():
    url = "ws://localhost:1234"
    db = driver(url)
    
    queries = [
        "MATCH (?i:Intervention)-[?e]->(?s:Session) RETURN TYPE(?e) LIMIT 1",
        "MATCH (?p:Person)-[?e]->(?pp:PoliticalParty) RETURN TYPE(?e) LIMIT 1",
        "MATCH (?p:Person)-[?e1]->(?x)-[?e2]->(?pp:PoliticalParty) RETURN TYPE(?e1), TYPE(?e2) LIMIT 1",
        "MATCH (?i:Intervention)-[?e]->(?p:Person) RETURN TYPE(?e) LIMIT 1",
        "MATCH (?s:Session) RETURN ?s.date LIMIT 1"
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
