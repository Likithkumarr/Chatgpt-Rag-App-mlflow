import mlflow
import mlflow.data
import pandas as pd
import json
import os
import warnings
import time

# Clear redundant path interpretation warnings in terminal console output
warnings.filterwarnings("ignore", category=UserWarning)

def update_local_csv(run_id, username, user_prompt, assistant_response, sentiment="unrated", score=None):
    """
    Immediately updates or appends a record to the local CSV dataset file.
    """
    os.makedirs("datasets", exist_ok=True)
    user_suffix = f"_{username}" if username else ""
    output_file = f"datasets/mlflow_rlhf_dataset{user_suffix}.csv"
    
    dataset_id = f"eval_{run_id[:8]}"
    new_row = {
        "dataset_id": dataset_id,
        "username": username,
        "user_question": user_prompt,
        "assistant_response": assistant_response,
        "feedback_sentiment": sentiment.capitalize() if sentiment else "unrated",
        "binary_score": score,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        if not df.empty and dataset_id in df["dataset_id"].values:
            df.loc[df["dataset_id"] == dataset_id, ["feedback_sentiment", "binary_score"]] = [new_row["feedback_sentiment"], score]
        else:
            df_new = pd.DataFrame([new_row])
            # Prevent FutureWarning by ensuring we don't concat with empty DataFrames
            if df.empty:
                df = df_new
            else:
                df = pd.concat([df_new, df], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(output_file, index=False, encoding="utf-8")

def export_mlflow_feedback_dataset(experiment_name="RAG_Enterprise_Chatbot", output_format="csv", target_user=None):
    """
    Connects to the local MLflow server, parses all logged user feedback 
    artifacts for a specific user, and packages them into a clean tabular evaluation dataset.
    """
    print(f"Connecting to MLflow Tracking Server to extract '{experiment_name}'...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # 1. Fetch the experiment metadata
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Error: Experiment '{experiment_name}' not found. Check your spelling.")
        return
    
    #  Hardcode tracking URI directly into client context to stop ghost leaks
    client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
    
    # Filter by user if provided
    filter_string = f"tags.operator = '{target_user}'" if target_user else ""
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], 
        filter_string=filter_string
    )
    
    dataset_rows = []
    print(f"Found {len(runs)} total workspace logs available. Checking for artifacts...")

    for run in runs:
        run_id = run.info.run_id
        
        # Skip processing our own evaluation compile logs to prevent infinite loop checking
        if run.data.tags.get("type") == "dataset_aggregation":
            continue
            
        operator = run.data.tags.get("operator", "unknown_user")
        status = run.data.tags.get("feedback_status", "unrated")
        score = run.data.metrics.get("human_feedback_score", None)
        
        try:
            artifacts = client.list_artifacts(run_id, path="evals")
        except Exception:
            continue  # Pass on if the run doesn't have an 'evals' folder yet
        
        for artifact in artifacts:
            if artifact.path.endswith(".json"):
                local_path = client.download_artifacts(run_id, artifact.path)
                with open(local_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Use current live tag values if user updated feedback since creation
                current_sentiment = status if status != "unrated" else data.get("sentiment", "unrated")
                current_score = score if score is not None else data.get("score", None)

                # Append row structure to workspace registry matrix
                dataset_rows.append({
                    "dataset_id": f"eval_{run_id[:8]}",
                    "username": data.get("username", operator),
                    "user_question": data.get("user_prompt", ""),
                    "assistant_response": data.get("assistant_response", ""),
                    "feedback_sentiment": current_sentiment,
                    "binary_score": current_score,
                    "timestamp": data.get("timestamp", "")
                })
                
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    # 3. Convert array matrix to a clean Pandas DataFrame
    if not dataset_rows:
        print(f"No evaluation records found for user '{target_user}' yet.")
        return
        
    df = pd.DataFrame(dataset_rows)
    
    # Sort by timestamp descending so newest data sits at the top row slot
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    
    # 4. Save dataset to disk
    os.makedirs("datasets", exist_ok=True)
    user_suffix = f"_{target_user}" if target_user else ""
    
    if output_format.lower() == "csv":
        output_file = f"datasets/mlflow_rlhf_dataset{user_suffix}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
    else:
        output_file = f"datasets/mlflow_rlhf_dataset{user_suffix}.json"
        df.to_json(output_file, orient="records", indent=4, force_ascii=False)

    # 5. Log and Register the Dataset
    try:
        run_name = f"User_Dataset_{target_user}" if target_user else "Compiled_RLHF_Dataset_Log"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tag("type", "dataset_aggregation")
            mlflow.set_tag("operator", target_user or "system")
            mlflow.set_tag("mlflow.user", target_user or "system")
            
            absolute_path = os.path.abspath(output_file)
            file_uri = f"file://{absolute_path}" if os.name != 'nt' else f"file:///{absolute_path.replace('\\', '/')}"
            
            mlflow_dataset = mlflow.data.from_pandas(
                df, 
                source=file_uri, 
                name="rlhf_golden_evaluation_set",
                targets="binary_score"
            )
            
            # Log as a dataset input and as a physical CSV artifact
            mlflow.log_input(mlflow_dataset, context="evaluation")
            mlflow.log_artifact(output_file)
            
            print(f"✨ Successfully refreshed global dataset: {output_file}")

    except Exception as e:
        print(f"Global dataset build sync bypassed: {e}")

    print(f"🎉 Success! Generated dataset saved to: {output_file}")