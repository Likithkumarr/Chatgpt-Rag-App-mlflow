import sqlite3
import chromadb
import streamlit as st
from config import CHROMA_PATH, SQLITE_PATH

# --- CHROMA VECTOR DB (Exclusively for PDF context) ---
@st.cache_resource
def get_vector_db():
    return chromadb.PersistentClient(path=CHROMA_PATH)

chroma_client = get_vector_db()
# Keep ONLY the vector chunks collection in Chroma
rag_coll = chroma_client.get_or_create_collection("pdf_vectors")


# --- SQLITE 3 (For User, Chat History, and Feedback Data) ---
def get_sqlite_conn():
    """Returns a connection to SQLite. 
    Note: Streamlit is multi-threaded; we handle connections per-thread safely."""
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_sqlite_tables():
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    
    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            display_name TEXT
        )
    """)
    
    # 2. Chat Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT,
            messages_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 3. Feedback Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id TEXT PRIMARY KEY,
            username TEXT,
            prompt TEXT,
            response TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize tables immediately on startup
init_sqlite_tables()