
# db_test.py
import os
import psycopg2
from psycopg2 import OperationalError

dsn = os.getenv("SUPABASE_DATABASE_URL")
print("Using DSN:", dsn)

try:
    conn = psycopg2.connect(dsn, sslmode="require", connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT now();")
    print("Connected OK. server time:", cur.fetchone())
    cur.close()
    conn.close()
except OperationalError as e:
    print("OperationalError:", e)
except Exception as e:
    print("Other error:", type(e).__name__, e)
