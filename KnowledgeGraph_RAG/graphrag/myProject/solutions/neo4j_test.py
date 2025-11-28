# neo4j_test.py
import os
from neo4j import GraphDatabase, exceptions
from dotenv import load_dotenv

load_dotenv()   # only if you use a .env file in project root

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
pwd  = os.getenv("NEO4J_PASSWORD")
db   = os.getenv("NEO4J_DATABASE", "neo4j")

print("URI:", uri)
print("USERNAME:", user)
print("DATABASE:", db)
# don't print pwd for security

if not uri or not user or not pwd:
    raise SystemExit("NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set in env")

try:
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    print("Driver created, verifying connectivity...")
    driver.verify_connectivity()
    print("verify_connectivity succeeded — opening a session and running RETURN 1")
    with driver.session(database=db) as session:
        result = session.run("RETURN 1 AS result")
        print("Query result:", result.single()["result"])
    driver.close()
    print("Done — connection & query OK")
except exceptions.AuthError:
    print("AUTH ERROR: invalid username/password.")
except exceptions.ServiceUnavailable as e:
    print("SERVICE UNAVAILABLE:", e)
except Exception as e:
    print("OTHER ERROR:", type(e), e)
