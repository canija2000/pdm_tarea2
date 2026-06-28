from millenniumdb_driver import driver

def main():
    url = "ws://localhost:1234"
    db = driver(url)
    
    query = "MATCH (?i:Intervention)-[:HasEmbedding]->(?c) RETURN ?c.content LIMIT 1"
    
    try:
        with db.session() as session:
            result = session.run(query)
            for record in result:
                vals = list(record.values()) if hasattr(record, 'values') else list(record)
                print("Content exists?", vals[0] is not None)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
