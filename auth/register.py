import streamlit as st
from db import get_sqlite_conn

def register():
    new_u = st.text_input("Create Username")
    new_p = st.text_input("Create Password", type="password")

    if st.button("Register"):
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (new_u,))
        if cursor.fetchone():
            st.error("User exists!")
            conn.close()
        else:
            cursor.execute(
                "INSERT INTO users (username, password, display_name) VALUES (?, ?, ?)",
                (new_u, new_p, new_u)
            )
            conn.commit()
            conn.close()
            st.success("Registered! You can now log in.")