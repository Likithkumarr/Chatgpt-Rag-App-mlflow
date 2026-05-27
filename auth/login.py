import streamlit as st
import time
from db import get_sqlite_conn

def login():
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (u,))
        user = cursor.fetchone()
        conn.close()

        if user and user["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.display_name = user["display_name"] if user["display_name"] else u

            st.success(f"✅ Welcome {st.session_state.display_name}")
            st.balloons()
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid Credentials")