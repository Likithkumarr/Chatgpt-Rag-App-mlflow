import uuid
import time
from datetime import datetime
import mlflow
from db import get_sqlite_conn
from extract_dataset import update_local_csv

def save_feedback(username, prompt, response):
    """
    Persists explicit RLHF records to the local relational SQLite database.
    """
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO feedback (feedback_id, username, prompt, response, timestamp) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), username, prompt, response, str(datetime.now()))
        )
        conn.commit()
        print("Successfully committed feedback row to SQLite database.")
    except Exception as e:
        print(f"Error writing feedback to SQLite: {e}")
    finally:
        conn.close()

def log_mlflow_feedback(run_id, score, feedback_type, username, prompt, response):
    """
    Production Observation Logging Pattern.
    Forces MLflow Trace viewers to render text variables as a structural grid block.
    """
    if run_id:
        try:
            client = mlflow.tracking.MlflowClient()
            
            # 1. Update the parent run tracking metrics
            client.log_metric(run_id, "human_feedback_score", float(score))
            # 🚀 NEW: Import the official MLflow system tag constants
            from mlflow.utils.mlflow_tags import MLFLOW_USER
            
            # Force the core UI "User" column to populate directly
            client.set_tag(run_id, MLFLOW_USER, username)
            # 2. Log separate metadata tags so they show up as clear run line items
            tags_to_set = {
                "feedback_status": feedback_type.capitalize(),
                "operator": username,
                "mlflow.user": username,
                "evaluated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            for key, val in tags_to_set.items():
                client.set_tag(run_id, key, val)
            
            # 3. Build an explicit Markdown Matrix Table layout.
            markdown_table_summary = (
                f"### 📊 Human Evaluation Record\n\n"
                f"| Evaluation Parameter | Logged Value Context |\n"
                f"| :--- | :--- |\n"
                f"| 👤 **Username** | {username} |\n"
                f"| ❓ **User Question** | {prompt} |\n"
                f"| 🤖 **Model Response** | {response} |\n"
                f"| 🎯 **Feedback State** | {'Positive (👍 Good)' if score == 1.0 else 'Negative (👎 Bad)'} |\n"
                f"| ⏰ **Timestamp** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |"
            )
            
            # 4. Attach evaluation directly to the MLflow Trace context
            try:
                from mlflow.entities import AssessmentSource
                from mlflow.entities.assessment_source import AssessmentSourceType
                
                # Obtain experiment context from the run to avoid scope errors
                run_info = client.get_run(run_id).info
                
  
                recent_traces = client.search_traces(
                    locations=[run_info.experiment_id],
                    filter_string=f"run_id = '{run_id}'",
                    max_results=1
                )
                
                if recent_traces:
                    target_trace_id = recent_traces[0].info.trace_id
                    
                    # Log feedback directly to the trace
                    mlflow.log_feedback(
                        trace_id=target_trace_id,
                        name="user_feedback",
                        value=score,
                        rationale=markdown_table_summary,
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.HUMAN,
                            source_id=username
                        )
                    )
                    print(f"Successfully pinned line-by-line assessment directly to Trace ID: {target_trace_id}")
                else:
                    print("Could not find an active trace context for this run to attach feedback.")
                    
            except Exception as e:
                print(f"Trace assessment UI logging bypassed: {e}")

            # 5. Backup: Log explicit flat text configuration to the run files folder
            client.log_text(run_id, markdown_table_summary, f"evals/feedback_{str(uuid.uuid4())[:8]}.md")
            print(f"Async trace alignment grid processed for Run ID: {run_id}")
            
            # 📝 IMMEDIATE CSV UPDATE: Update the local CSV record with the new feedback score
            update_local_csv(
                run_id=run_id,
                username=username,
                user_prompt=prompt,
                assistant_response=response,
                sentiment=feedback_type,
                score=score
            )
            
        except Exception as e:
            print(f"MLflow feedback logging skipped: {e}")
    else:
        print("Production tracking trace target undefined.")


def handle_feedback(st):
    if (st.session_state.get("messages") and 
        st.session_state.messages[-1]["role"] == "assistant" and
        st.session_state.get("last_mlflow_run_id")):

        c1, c2 = st.columns([1, 8])
        
        last_user_query = st.session_state.last_user_query_for_feedback
        last_assistant_response = st.session_state.last_assistant_response_for_feedback
        last_mlflow_run_id = st.session_state.last_mlflow_run_id
        current_user = st.session_state.username

        with c1:
            if st.button("👍", key="thumbs_up_action"):
                save_feedback(current_user, last_user_query, last_assistant_response)
                
                log_mlflow_feedback(
                    run_id=last_mlflow_run_id, 
                    score=1.0, 
                    feedback_type="good",
                    username=current_user,
                    prompt=last_user_query,
                    response=last_assistant_response
                )
        
                st.success("Saved to Production Registry!")
                st.session_state.last_mlflow_run_id = None
                time.sleep(0.5)
                st.rerun()
                
        with c2:
            if st.button("👎", key="thumbs_down_action"):
                log_mlflow_feedback(
                    run_id=last_mlflow_run_id, 
                    score=0.0, 
                    feedback_type="bad",
                    username=current_user,
                    prompt=last_user_query,
                    response=last_assistant_response
                )

                st.session_state.messages.pop() 
                st.session_state.retry_trigger = True
                st.session_state.last_mlflow_run_id = None
                st.rerun()
