'''
실험 및 평가를 위한 data extraction:
index, table_id, question, answer, raw_table, raw_table_size, raw_table_cell_counts, table_columns
'''

from langchain_core.runnables import RunnableLambda

def get_table_question_answer_fn(state):
    import sqlite3
    import json
    import pandas as pd

    # ✅ 이미 직접 입력된 경우: 그대로 반환
    if "raw_table" in state and "question" in state and "answer" in state:
        raw_table = state["raw_table"]
        question = state["question"]
        answer = state["answer"]
        table_id = state.get("table_id", "unknown_table")

        def serialize_table(df):
            if df.empty:
                return "[No Table]"
            lines = [" | ".join(df.columns)]
            for _, row in df.iterrows():
                lines.append(" | ".join(map(str, row)))
            return "\n".join(lines)

        table_text = serialize_table(raw_table)
        raw_table_size = raw_table.shape
        raw_table_cell_counts = raw_table_size[0] * raw_table_size[1]
        table_columns = list(raw_table.columns)

        return {
            **state,
            "table_id": table_id,
            "table_text": table_text,
            "raw_table_size": raw_table_size,
            "raw_table_cell_counts": raw_table_cell_counts,
            "table_columns": table_columns,
        }

    # ✅ 기존 방식 유지
    raw_db_file_path = state["raw_db_file_path"]
    json_df_file_path = state["json_df_file_path"]
    index = state["index"]

    with open(json_df_file_path, 'r') as f:
        json_df = json.load(f)

    conn = sqlite3.connect(raw_db_file_path)
    raw_table_id = json_df["original_table_id"][str(index)]
    table_id = f'table_{raw_table_id}'.replace("-", "_")
    safe_table_id = f'"{table_id}"'

    sql_query = f"SELECT * FROM {safe_table_id}"
    try:
        raw_table = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        raw_table = pd.DataFrame()

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
    raw_table_size = raw_table.shape
    raw_table_cell_counts = raw_table_size[0] * raw_table_size[1]
    table_columns = list(raw_table.columns)

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