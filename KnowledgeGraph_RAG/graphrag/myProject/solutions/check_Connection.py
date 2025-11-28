import socket
host = "307436bd.databases.neo4j.io"
# Aura uses the default secure bolt port; testing 7687 is helpful
port = 7687
try:
    with socket.create_connection((host, port), timeout=5) as s:
        print("TCP connect OK")
except Exception as e:
    print("TCP connect failed:", e)
