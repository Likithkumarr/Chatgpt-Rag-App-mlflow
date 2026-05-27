import streamlit as st
import re
import time
import json
import os
import mlflow
import mlflow.data
import pandas as pd
from utils.session import init_session
from auth.login import login
from auth.register import register
from chat.chat_engine import get_llms
from chat.chat_ui import render_sidebar
from chat.history import get_history
from feedback.rlhf import handle_feedback
from extract_dataset import update_local_csv
from db import get_sqlite_conn

# Initialize session structures safely
init_session()

# --- PRODUCTION MLFLOW MONITORING ENGINE ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RAG_Enterprise_Chatbot")

# Enable automatic trace generation context tracking for all Langchain calls
mlflow.langchain.autolog(log_traces=True)

st.set_page_config(page_title="Production RAG Platform", page_icon="🛡️", layout="wide")

if not st.session_state.logged_in:
    st.title("🔐 Multi-User Secure Chat")
    st.info("If you don't have an account, please **register** first!,If you already have an account, please **login** to continue.")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1: login()
    with tab2: register()
else:
    current_name = st.session_state.display_name or st.session_state.username
    st.title(f"🤖 Operational Instance: {current_name}")
    
    render_sidebar()
    llm1, llm2 = get_llms()
    
    # Display historical elements
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Message...")
    
    if prompt or st.session_state.retry_trigger:
        if st.session_state.retry_trigger:
            user_query = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
            bad_answer = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
            st.warning("🔄 Generating alternative model inference perspective...")
            active_llm = llm2  
            instruction = f"The user disliked your previous response: {bad_answer}. Provide a completely alternative view."
        else:
            user_query = prompt
            active_llm = llm1
            instruction = "Be concise and helpful."
            name_match = re.search(r"(?:my name is|i am) (\w+)", user_query.lower())
            if name_match: 
                new_display_name = name_match.group(1).capitalize()
                st.session_state.display_name = new_display_name
                # Persist display name update back down to SQLite base
                conn = get_sqlite_conn()
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET display_name = ? WHERE username = ?", (new_display_name, st.session_state.username))
                conn.commit()
                conn.close()
            
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.markdown(user_query)

        # Consolidated Execution Context: Single Flat Run for immediate artifact visibility
        run_name = f"Chat_{time.strftime('%H:%M:%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.perf_counter()
            current_run_id = run.info.run_id
            
            # Log tags for dashboard grid tracking
            mlflow.set_tags({
                "operator": st.session_state.username,
                "mlflow.user": st.session_state.username,
                "session_id": st.session_state.current_session_id,
                "retry_mode_active": str(st.session_state.retry_trigger),
                "user_question": user_query[:150]
            })
            
            # Explicitly log prompt as a parameter for high visibility
            mlflow.log_param("user_prompt", user_query[:250])
            # Log full prompt as an artifact
            mlflow.log_text(user_query, "conversation/user_prompt.txt")
            
            # Construct a tracking config dictionary to link traces with this Run ID
            run_config = {"run_id": current_run_id}
            
            # Dynamically convert this interaction sequence into an MLflow Dataset input
            try:
                current_input_df = pd.DataFrame([{
                    "user_query": user_query,
                    "session_user": st.session_state.username,
                    "project_type": "PDF_RAG_App"
                }])
                
                eval_dataset = mlflow.data.from_pandas(
                    current_input_df, 
                    name="pdf_chat_inputs", 
                    source="streamlit_chat_session"
                )
                mlflow.log_input(eval_dataset, context="evaluation")
            except Exception as e:
                print(f"Dataset column logging bypassed: {e}")

            tool_calls = 0
            
            with st.chat_message("assistant"):
                all_past_interactions = get_history(st.session_state.username)
                doc_context = ""
                used_doc_final = False
                fallback_to_general_ai = False
                response = ""
                
                # Metadata validation bypass logic
                if "UPLOAD" in user_query.upper() and ("WHAT" in user_query.upper() or "NAME" in user_query.upper()):
                    filenames = st.session_state.get('uploaded_filenames', [])
                    response = f"Active context includes files: {', '.join(filenames)}" if filenames else "No documents uploaded."
                    st.info("📄 File Metadata Context")
                    used_doc_final = True

                # Check if a PDF vector index is active
                if not used_doc_final:
                    if not st.session_state.get('retriever'):
                        fallback_to_general_ai = True
                    else:
                        tool_calls += 1
                        try:
                            docs = st.session_state.retriever.invoke(user_query, config=run_config)
                        except TypeError:
                            docs = st.session_state.retriever.invoke(user_query)
                        
                        if docs and len(docs) > 0:
                            doc_context = "\n".join([d.page_content for d in docs])
                            doc_prompt = f"Context:\n{doc_context}\n\nQuestion: {user_query}\n\nInstruction: Answer using the context. If the answer is not found in the context, reply exactly with 'I_DO_NOT_KNOW_CONTEXT'."
                            
                            try:
                                response_content = active_llm.invoke(doc_prompt, config=run_config).content
                                uncertainty_phrases = ["I_DO_NOT_KNOW_CONTEXT", "I DON'T KNOW", "DOES NOT PROVIDE", "NO INFORMATION", "NOT MENTIONED", "CONTEXT DOES NOT"]
                                
                                if not any(phrase in response_content.upper() for phrase in uncertainty_phrases):
                                    used_doc_final = True
                                    response = response_content
                                    st.info("📄 Responded using PDF Document Context")
                                else:
                                    fallback_to_general_ai = True
                            except Exception as e:
                                if "content_filter" in str(e):
                                    st.warning("⚠️ Content filter triggered. Falling back safely.")
                                    fallback_to_general_ai = True
                                else:
                                    raise e
                        else:
                            fallback_to_general_ai = True

                if fallback_to_general_ai and not used_doc_final:
                    ai_prompt = f"{instruction}\nSystem: Helper\nHistory: {all_past_interactions}\nUser: {user_query}"
                    response = active_llm.invoke(ai_prompt, config=run_config).content
                    
                    if st.session_state.get('retriever'):
                        st.info("💡 Question not found in PDF context. Answering using General AI knowledge.")
                    else:
                        st.info("🤖 Answering using General AI (No PDF Uploaded)")

                st.markdown(response)
                
                # Log assistant response for quick audit in MLflow UI
                mlflow.log_param("assistant_response", response[:250])
                mlflow.log_text(response, "conversation/assistant_response.txt")
                
                # --- TELEMETRY PERFORMANCE INGEST ---
                latency = (time.perf_counter() - start_time) * 1000
                mlflow.log_metrics({
                    "latency_ms": latency,
                    "vector_tool_calls": tool_calls,
                    "retrieval_success": 1.0 if used_doc_final else 0.0
                })
                
                # 📝 LOG PROMPT-RESPONSE TABLE: This populates the "Prompts" view in MLflow UI
                chat_table = {
                    "user_prompt": [user_query],
                    "assistant_response": [response],
                    "retrieval_used": [used_doc_final]
                }
                mlflow.log_table(data=chat_table, artifact_file="chat_logs/interaction_table.json")

                # 🚀 IMMEDIATE ARTIFACT LOGGING: Save details so the dataset catches this run right away!
                immediate_payload = {
                    "username": st.session_state.username,
                    "user_prompt": user_query,
                    "assistant_response": response,
                    "sentiment": "unrated", 
                    "score": None,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                os.makedirs("temp_evals", exist_ok=True)
                temp_file_path = f"temp_evals/eval_{current_run_id[:8]}.json"
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    json.dump(immediate_payload, f, indent=4, ensure_ascii=False)
                
                # Upload artifact to 'evals' directory so extract_dataset can read it immediately
                mlflow.log_artifact(temp_file_path, artifact_path="evals")
                
                # 📝 IMMEDIATE CSV UPDATE: Add interaction to local CSV file instantly
                update_local_csv(
                    run_id=current_run_id,
                    username=st.session_state.username,
                    user_prompt=user_query,
                    assistant_response=response,
                    sentiment="unrated",
                    score=None
                )
                
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

                st.session_state.last_mlflow_run_id = current_run_id
                st.session_state.last_user_query_for_feedback = user_query
                st.session_state.last_assistant_response_for_feedback = response

        # Add message and save chat state to SQLite
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        chat_title = st.session_state.messages[0]["content"][:30]
        messages_str = json.dumps(st.session_state.messages)
        
        cursor.execute("""
            INSERT INTO chat_history (session_id, username, title, messages_json, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                messages_json=excluded.messages_json,
                updated_at=CURRENT_TIMESTAMP
        """, (st.session_state.current_session_id, st.session_state.username, chat_title, messages_str))
        
        conn.commit()
        conn.close()

        st.session_state.retry_trigger = False

    # Render operational evaluation panel elements
    handle_feedback(st)