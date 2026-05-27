import json
from db import get_sqlite_conn

def get_history(username):
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT messages_json FROM chat_history WHERE username = ?", (username,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    if rows:
        for row in rows:
            try:
                msgs = json.loads(row["messages_json"])
                for m in msgs:
                    role = "User" if m["role"] == "user" else "Assistant"
                    history.append(f"{role}: {m['content']}")
            except Exception:
                continue

    return "\n".join(history[-20:])