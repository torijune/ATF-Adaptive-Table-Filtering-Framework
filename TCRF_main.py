from TCRF_graph.graph_workflow import build_workflow_graph

def make_response(index):
    workflow = build_workflow_graph()

    raw_db_file_path = "data/table.db"
    json_df_file_path = "data/test.json"

    result = workflow.invoke({
        "raw_db_file_path": raw_db_file_path,
        "json_df_file_path": json_df_file_path,
        "index": index
    })

    # raw_table과 filtered_df 둘 다 가져오기
    raw_table = result.get("raw_table", "⚠️ raw_table 존재하지 않습니다.")
    filtered_df = result.get("filtered_df", "⚠️ filtered_df 존재하지 않습니다.")
    question = result.get("question", "⚠️ filtered_df 존재하지 않습니다.")
    answer = result.get("answer", "⚠️ filtered_df 존재하지 않습니다.")

    return raw_table, filtered_df, question, answer

if __name__ == "__main__":
    index = str(input("질문을 원하는 질문 Index를 입력하세요. : \n"))
    raw_table, filtered_df, question = make_response(index)
    print("✅ Raw Table:")
    print(raw_table)
    print("\n✅ Filtered Table:")
    print(filtered_df)