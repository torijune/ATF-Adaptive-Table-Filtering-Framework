'''
실험 및 평가를 위한 data extraction:
index, table_id, question, answer, raw_table, raw_table_size, raw_table_cell_counts, table_columns
'''

from langchain_core.runnables import RunnableLambda

def get_table_question_answer_fn(state):
    import sqlite3
    import json
    import pandas as pd

    raw_db_file_path = state["raw_db_file_path"]
    json_df_file_path = state["json_df_file_path"]
    index = state["index"]

    print(f"[DataLoader] Loading question index: {index}")

    with open(json_df_file_path, 'r') as f:
        json_df = json.load(f)

    conn = sqlite3.connect(raw_db_file_path)

    raw_table_id = json_df["original_table_id"][str(index)]
    table_id = f'table_{raw_table_id}'.replace("-", "_")
    safe_table_id = f'"{table_id}"'

    sql_query = f"SELECT * FROM {safe_table_id}"

    try:
        df = pd.read_sql_query(sql_query, conn)
        raw_table = df
        print(f"[DataLoader] Table loaded with shape: {df.shape}")
    except Exception as e:
        print(f"[DataLoader] SQL Error: {e}")
        raw_table = pd.DataFrame()
        df = raw_table

    conn.close()

    def serialize_table(df):
        if df.empty:
            return "[No Table]"
        lines = [" | ".join(df.columns)]
        for _, row in df.iterrows():
            lines.append(" | ".join(map(str, row)))
        return "\n".join(lines)

    table_text = serialize_table(raw_table)
    question = json_df["question"][str(index)]
    answer = json_df["answer"][str(index)]

    # New fields
    raw_table_size = raw_table.shape  # (rows, cols)
    raw_table_cell_counts = raw_table_size[0] * raw_table_size[1]
    table_columns = list(raw_table.columns)

    print(f"[DataLoader] Question: {question}")
    print(f"[DataLoader] Answer: {answer}")

    return {
        **state,
        "table_id": table_id,
        "raw_table": raw_table,
        "table_text": table_text,
        "question": question,
        "answer": answer,
        "raw_table_size": raw_table_size,
        "raw_table_cell_counts": raw_table_cell_counts,
        "table_columns": table_columns,
    }

dataload_node = RunnableLambda(get_table_question_answer_fn)