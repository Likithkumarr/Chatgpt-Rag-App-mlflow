import streamlit as st
import json
import uuid
import time
from db import get_sqlite_conn
from rag.rag_pipeline import build_rag
from extract_dataset import export_mlflow_feedback_dataset

def render_sidebar():
    with st.sidebar:
        st.subheader(f"👤 {st.session_state.username}")

        st.write("### 🛠️ Admin Control Panel")
        
        # This button allows you to compile your evaluation sets on demand
        if st.button("🔄 Compile Latest Evaluation Dataset"):
            with st.spinner(f"Compiling dataset for {st.session_state.username}..."):
                try:
                    export_mlflow_feedback_dataset(target_user=st.session_state.username)
                    st.sidebar.success(f"Personal Evaluation Set refreshed!")
                except Exception as e:
                    st.sidebar.error(f"Compilation error: {e}")
        
        st.divider()
        # ---------------- CONTROL BUTTONS ----------------
        if st.button("➕ New Chat Session"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.pop("mlflow_parent_run_id", None)
            st.rerun()

        if st.button("🧹 Clear Current Screen"):
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ DELETE ALL MY HISTORY"):
            conn = get_sqlite_conn()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE username = ?", (st.session_state.username,))
            conn.commit()
            conn.close()
            
            st.session_state.messages = []
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.pop("mlflow_parent_run_id", None)
            st.warning("All history deleted forever!")
            st.rerun()

        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

        st.divider()
        # ---------------- FILE UPLOAD ----------------
        uploaded_files = st.file_uploader("Upload Knowledge Files", accept_multiple_files=True)
        if uploaded_files:
            st.session_state.uploaded_filenames = [f.name for f in uploaded_files]
    
            if uploaded_files and not st.session_state.retriever:
                with st.status("🚀 Initializing Document Engine...", expanded=True) as s:
                    st.session_state.retriever = build_rag(uploaded_files)
                    st.write("📂 Processing, Chunking and Vectorizing into ChromaDB...")
                    time.sleep(1)
                    s.update(label="✨ Indexing Complete!", state="complete", expanded=False)
                    
        st.divider()

        # ---------------- CHAT HISTORY FROM SQLITE ----------------
        st.subheader("📜 Your Past Chats")
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, title, messages_json FROM chat_history WHERE username = ? ORDER BY updated_at DESC", 
            (st.session_state.username,)
        )
        user_chats = cursor.fetchall()
        conn.close()

        if user_chats:
            for chat in user_chats:
                sid = chat["session_id"]
                title = chat["title"] if chat["title"] else f"Chat {sid[:8]}"

                if st.button(f"💬 {title[:15]}...", key=f"h_{sid}"):
                    st.session_state.current_session_id = sid
                    st.session_state.messages = json.loads(chat["messages_json"])
                    st.session_state.pop("mlflow_parent_run_id", None)
                    st.rerun()

        return uploaded_files