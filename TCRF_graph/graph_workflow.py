from langgraph.graph import StateGraph, END
from typing import Annotated, TypedDict
from langchain_core.runnables import Runnable
from pandas import DataFrame

from TCRF_agent.data_loader import dataload_node
from TCRF_agent.LLM import responder_node
from TCRF_agent.column_relevance_agent import column_relevance_node
from TCRF_agent.column_clustering_agent import column_clustering_node
from TCRF_agent.row_ranker import row_ranker_node
from TCRF_agent.final_table_selecter import final_table_select_node
from TCRF_agent.clustering_evluation import ensemble_evluation_node
from TCRF_agent.essential_columns import extract_essential_columns_node
from TCRF_agent.predict_answer_entity import predict_answer_entity_node

class AgentState(TypedDict):
    # 데이터를 불러오기 위한
    raw_db_file_path: Annotated[str, "SQLite로 저장된 table 경로"]
    json_df_file_path: Annotated[str, "질문 및 메타데이터가 포함된 train.json 경로"]
    index: Annotated[int, "데이터셋 상에서 접근할 질문 인덱스"]

    # 불러온 데이터에 대한 정보들
    question: Annotated[str, "주어진 table에 대한 질문"]
    raw_table: Annotated[DataFrame, "Raw table의 데이터프레임"]
    table_text: Annotated[str, "text 형태의 table"]
    answer: Annotated[str, "Question에 대한 real answer"]

    # 중간 분석 과정들
    ## predict_answer_entity
    predicted_answer_entity: Annotated[str, "LLM이 예측한 Question에 대한 Answer의 entity 타입"]
    predicted_answer_entity_scores: Annotated[dict[str, float], "LLM이 예측한 엔티티 후보 리스트 및 각 타입에 대한 confidence 점수 (예: {'person': 0.9, 'organization': 0.8, ...})"]

    ## column_relevance_agent
    column_description: Annotated[dict[str, str], "LLM이 생성한 Column에 대한 설명 (묘사)"]
    column_relevance_scores: Annotated[dict[str, list[float]], "각 column의 [LLM 점수, cosine 유사도 점수]"]

    ## column_clustering_agent
    column_clusters: Annotated[dict[str, int], "각 column이 속한 클러스터 ID"]
    cluster_centers: Annotated[list[list[float]], "각 클러스터의 중심 벡터"]

    ## clustering_evaluation
    selected_cluster: Annotated[int, "선택된 클러스터 ID"]
    essential_columns: Annotated[list[str], "Question과 column 명을 보고 LLM이 판단한 필수 컬럼 리스트"]
    filtered_columns: Annotated[list[str], "선택된 클러스터에 속한 컬럼 + 이외의 클러스터 컬럼 + 필수 컬럼 리스트"]

    ## row_ranker
    selected_rows: Annotated[list[str], "선택된 클러스터에 속한 row 리스트"]
    tfidf_scores: Annotated[list[float], "TF-IDF 기반의 각 row 유사도 점수"]
    bm25_scores: Annotated[list[float], "BM25 기반의 각 row 유사도 점수"]
    dense_scores: Annotated[list[float], "SentenceTransformer 기반 dense 유사도 점수"]
    row_similarity_scores: Annotated[list[float], "TF-IDF, BM25, Dense 유사도 점수의 가중 평균 (softmax normalized)"]
    final_table_text: Annotated[str, "최종 선형화 된 table"]
    top_row_indices: Annotated[list[int], "유사도 기준 상위 N개의 row index 리스트"]

    filtered_df: Annotated[DataFrame, "최종 필터링된 테이블의 pandas 데이터프레임"]

    ## LLM
    real_answer: Annotated[str, "실제 정답"]
    LLM_answer: Annotated[str, "LLM이 생성한 정답"]

    ## Experiment components
    table_id: Annotated[str, "SQLite에서 추출된 테이블의 고유 ID"]
    raw_table_size: Annotated[tuple[int, int], "Raw 테이블의 크기 (행, 열)"]
    raw_table_cell_counts: Annotated[int, "Raw 테이블의 전체 셀 개수 (행 × 열)"]
    table_columns: Annotated[list[str], "Raw 테이블의 컬럼 이름 리스트"]
    filtered_table_size: Annotated[tuple[int, int], "선택된 테이블의 크기 (행, 열)"]
    filtered_table_cell_counts: Annotated[int, "선택된 테이블의 셀 개수 (행 × 열)"]
    selected_column_cluster_id: Annotated[int, "선택된 컬럼 클러스터 ID"]
    llm_score: Annotated[dict[str, float], "각 column에 대한 LLM 기반 relevance 점수 (단일 값)"]
    llm_debug_scores: Annotated[dict[str, dict[str, float]], "LLM score 계산에 사용된 세부 지표 (예: rank matrix, mean/std 등)"]
    column_similarity_scores: Annotated[dict[str, float], "각 column에 대한 cosine 유사도 기반 relevance 점수"]
    semantic_scores: Annotated[dict[str, float], "semantic 방식으로 선택된 클러스터별 점수"]
    mcdm_scores: Annotated[dict[str, float], "MCDM 방식으로 선택된 클러스터별 점수"]
    adaptive_scores: Annotated[dict[str, float], "adaptive 방식으로 선택된 클러스터별 점수"]
    ensemble_scores: Annotated[dict[str, float], "ensemble 방식으로 선택된 클러스터별 점수"]
    ensemble_details: Annotated[dict[str, dict[str, float]], "ensemble 방식에서 사용된 개별 스코어들의 구성 정보 (예: {'cluster_0': {'llm': 0.9, 'semantic': 0.85, ...}})"]
    ensemble_confidences: Annotated[dict[str, float], "ensemble 방식에서 각 클러스터에 대해 계산된 최종 confidence score"]
    voting_results: Annotated[dict[int, int], "ensemble 방식에서 클러스터별 투표 수 (예: {0: 1, 1: 2, 2: 0})"]

    raw_table_token_counts: Annotated[int, "Raw 테이블의 토큰 수"]
    filtered_table_token_counts: Annotated[int, "Filtered 테이블의 토큰 수"]

    column_embedding_variance: Annotated[dict[str, float], "선택된 컬럼들의 semantic 분산 (다양성 분석)"]

    ranking_score_weights: Annotated[dict[str, float], "TF-IDF / BM25 / Dense score의 가중치 구성"]
    

def build_workflow_graph() -> Runnable:
    builder = StateGraph(state_schema=AgentState)

    builder.add_node("data_loader", dataload_node)
    builder.add_node("column_relevance_checker", column_relevance_node)
    builder.add_node("column_cluster_agent", column_clustering_node)
    builder.add_node("select_column_agent", ensemble_evluation_node)
    builder.add_node("essential_column_node", extract_essential_columns_node)
    builder.add_node("row_ranker", row_ranker_node)
    builder.add_node("final_table_selecter", final_table_select_node)
    builder.add_node("responder", responder_node)
    builder.add_node("predict_answer_entity_node", predict_answer_entity_node)


    builder.set_entry_point("data_loader")
    builder.add_edge("data_loader", "predict_answer_entity_node")
    builder.add_edge("predict_answer_entity_node", "column_relevance_checker")
    builder.add_edge("column_relevance_checker", "column_cluster_agent")
    builder.add_edge("column_cluster_agent", "essential_column_node")
    builder.add_edge("essential_column_node", "select_column_agent")
    builder.add_edge("select_column_agent", "row_ranker")
    builder.add_edge("row_ranker", "final_table_selecter")
    builder.add_edge("final_table_selecter", END)
    # builder.add_edge("final_table_selecter", "responder")
    # builder.add_edge("responder", END)

    return builder.compile()