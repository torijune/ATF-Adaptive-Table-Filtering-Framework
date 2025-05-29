from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from langchain_core.runnables import Runnable
from pandas import DataFrame

from QA_agent.data_loader import dataload_node
from QA_agent.LLM import responder_node
from QA_agent.column_relevance_agent import column_relevance_node
from QA_agent.column_clustering_agent import column_clustering_node
from QA_agent.row_ranker import row_ranker_node
from QA_agent.final_table_selecter import final_table_select_node
from QA_agent.clustering_evlauation import ensemble_evlauation_node
from QA_agent.essential_columns import extract_essential_columns_node

class AgentState(TypedDict):
    raw_db_file_path: Annotated[str, "SQLite로 저장된 table 경로"]
    json_df_file_path: Annotated[str, "질문 및 메타데이터가 포함된 train.json 경로"]
    index: Annotated[int, "데이터셋 상에서 접근할 질문 인덱스"]

    question: Annotated[str, "주어진 table에 대한 질문"]
    raw_table: Annotated[DataFrame, "Raw table의 데이터프레임"]
    table_text: Annotated[str, "text 형태의 table"]
    answer: Annotated[str, "Question에 대한 real answer"]

    # Column relevance
    column_description: Annotated[dict[str, str], "LLM이 생성한 Column에 대한 설명 (묘사)"]
    column_relevance_scores: Annotated[dict[str, list[float]], "각 column의 [LLM 점수, cosine 유사도 점수]"]
    column_clusters: Annotated[dict[str, int], "각 column이 속한 클러스터 ID"]
    cluster_centers: Annotated[list[list[float]], "각 클러스터의 중심 벡터"]

    selected_cluster: Annotated[int, "선택된 클러스터 ID"]
    essential_columns: Annotated[list[str], "Question과 column 명을 보고 LLM이 판단한 필수 컬럼 리스트"]
    filtered_columns: Annotated[list[str], "선택된 클러스터에 속한 컬럼 + 이외의 클러스터 컬럼 + 필수 컬럼 리스트"]

    selected_rows: Annotated[list[str], "선택된 클러스터에 속한 row 리스트"]
    final_table_text: Annotated[str, "최종 선형화 된 table"]

    # Answers
    real_answer: Annotated[str, "실제 정답"]
    LLM_answer: Annotated[str, "LLM이 생성한 정답"]

def build_workflow_graph() -> Runnable:
    builder = StateGraph(state_schema=AgentState)

    builder.add_node("data_loader", dataload_node)
    builder.add_node("column_relevance_checker", column_relevance_node)
    builder.add_node("column_cluster_agent", column_clustering_node)
    builder.add_node("select_column_agent", ensemble_evlauation_node)
    builder.add_node("essential_column_node", extract_essential_columns_node)
    builder.add_node("row_ranker", row_ranker_node)
    builder.add_node("final_table_selecter", final_table_select_node)
    builder.add_node("responder", responder_node)


    builder.set_entry_point("data_loader")
    builder.add_edge("data_loader", "column_relevance_checker")
    builder.add_edge("column_relevance_checker", "column_cluster_agent")
    builder.add_edge("column_cluster_agent", "essential_column_node")
    builder.add_edge("essential_column_node", "select_column_agent")
    builder.add_edge("select_column_agent", "row_ranker")
    builder.add_edge("row_ranker", "final_table_selecter")
    builder.add_edge("final_table_selecter", "responder")
    builder.add_edge("responder", END)

    return builder.compile()