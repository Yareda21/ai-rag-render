# rag_api/supabase_rag.py
import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.pool import SimpleConnectionPool

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("SUPABASE_DATABASE_URL not set in environment")

connection_pool = None

def init_db_pool():
    """Initialize the database connection pool"""
    global connection_pool
    connection_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL,
        sslmode="require"
    )
    print("Database connection pool initialized")


def retrieve_topk(query_embedding: list, k: int = 4):
    """
    Retrieve top-k nearest vectors using connection pool
    """
    conn = None
    try:
        # Get connection from pool
        conn = connection_pool.getconn()
        register_vector(conn)
        cur = conn.cursor()
        
        # Use cosine similarity for better results with embeddings
        sql = """
        SELECT id, chunk_text, 1 - (embedding <=> %s) as similarity
        FROM rag_chunks
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        cur.execute(sql, (query_embedding, query_embedding, k))
        rows = cur.fetchall()
        cur.close()
        
        results = [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]
        return results
    except Exception as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn is not None:
            connection_pool.putconn(conn)