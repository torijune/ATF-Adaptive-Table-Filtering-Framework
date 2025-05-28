from QA_graph.graph_workflow import build_workflow_graph

def make_response(index):
    workflow = build_workflow_graph()

    raw_db_file_path = "data/table.db"
    json_df_file_path = "data/test.json"

    result = workflow.invoke({
        "raw_db_file_path": raw_db_file_path,
        "json_df_file_path": json_df_file_path,
        "index": index
    })
    # # Mermaid 기반 그래프 시각화
    # with open("workflow_graph.png", "wb") as f:
    #     f.write(workflow.get_graph(xray=True).draw_mermaid_png())

    return result.get("LLM_answer", "⚠️ LLM_answer 존재하지 않습니다.")

if __name__ == "__main__":
    index = str(input("질문을 원하는 질문 Index를 입력하세요. : \n"))
    result = make_response(index)
    print(f"[LLM Responder] ✅ Answer:\n{result}")