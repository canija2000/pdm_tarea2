from millenniumdb_driver import driver

def main():
    url = "ws://localhost:1234"
    db = driver(url)
    
    queries = {
        "P1_Path": "MATCH (?i:Intervention)-[:DeliveredBy]->(?pos:Position)-[:Represents]->(?party:PoliticalParty) RETURN ?party.name LIMIT 5",
        "P1_PartyNames": "MATCH (?p:PoliticalParty) RETURN ?p.name LIMIT 5",
        "P2_Path": "MATCH (?p:Person)-[:ServedAs]->(?pos:Position)<-[:DeliveredBy]-(?i:Intervention) RETURN ?p.gender LIMIT 5",
        "P3_Path": "MATCH (?i:Intervention)-[:HasIntervention]->(?proc:Procedure)-[:OCCURRED_IN]->(?s:Session) RETURN ?s.date LIMIT 5"
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
