from langchain_core.runnables import RunnableLambda

def get_table_question_answer_fn(state):
    import sqlite3
    import json
    import pandas as pd

    raw_db_file_path = state["raw_db_file_path"]
    json_df_file_path = state["json_df_file_path"]
    index = state["index"]

    print(f"[DataLoader] Loading question index: {index}")
    # print(f"[DataLoader] DB Path: {raw_db_file_path}, JSON Path: {json_df_file_path}")

    with open(json_df_file_path, 'r') as f:
        json_df = json.load(f)

    conn = sqlite3.connect(raw_db_file_path)

    raw_table_id = json_df["original_table_id"][str(index)]
    table_id = f'table_{raw_table_id}'.replace("-", "_")
    safe_table_id = f'"{table_id}"'

    # print(f"[DataLoader] Constructed table name: {safe_table_id}")

    sql_query = f"SELECT * FROM {safe_table_id}"

    try:
        df = pd.read_sql_query(sql_query, conn)
        raw_table = df
        print(f"[DataLoader] Table loaded with shape: {df.shape}")
    except Exception as e:
        raw_table = f"[SQL Error] {e}"
        df = None
        print(f"[DataLoader] SQL Error: {e}")

    conn.close()

    def serialize_table(df):
        if df is None:
            return "[No Table]"
        lines = [" | ".join(df.columns)]
        for _, row in df.iterrows():
            lines.append(" | ".join(map(str, row)))
        return "\n".join(lines)

    table_text = serialize_table(raw_table)
    question = json_df["question"][str(index)]
    answer = json_df["answer"][str(index)]

    print(f"[DataLoader] Question: {question}")
    print(f"[DataLoader] Answer: {answer}")

    return {**state,
            "raw_table": raw_table,
            "table_text": table_text,
            "question": question,
            "answer": answer
            }

dataload_node = RunnableLambda(get_table_question_answer_fn)