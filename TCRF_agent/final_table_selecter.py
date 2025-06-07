# final columns & rows로 클러스터링

from langchain_core.runnables import RunnableLambda


def linearize_row_wise(df):
    return " | ".join(
        ["; ".join([f"{col}: {val}" for col, val in row.items()]) for _, row in df.iterrows()]
    )

def final_table_select_fn(state):
    import pandas as pd

    selected_rows = state["selected_rows"]
    filtered_columns = state["filtered_columns"]

    # 선택된 컬럼만 추출
    filtered_df = selected_rows[filtered_columns]

    state["filtered_df"] = filtered_df
    # linearized 텍스트로 변환
    table_text = linearize_row_wise(filtered_df)

    # print(f"[FinalTableSelector] ✅ Filtered columns: {filtered_columns}")
    # print(f"[FinalTableSelector] 📋 Final table (linearized):\n{table_text}")

    return {
        **state,
        "final_table_text": table_text,
        "filtered_df": filtered_df
    }

final_table_select_node = RunnableLambda(final_table_select_fn)