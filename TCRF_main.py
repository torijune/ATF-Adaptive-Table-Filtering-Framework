import json
import os
import pandas as pd
from TCRF_graph.graph_workflow import build_workflow_graph
from tqdm import tqdm
import time

def make_response(index):
    workflow = build_workflow_graph()

    raw_db_file_path = "data/table.db"
    json_df_file_path = "data/test.json"

    result = workflow.invoke({
        "raw_db_file_path": raw_db_file_path,
        "json_df_file_path": json_df_file_path,
        "index": index
    })

    # 기본적인 index 내용 - 주로 dataloader
    question = result.get("question", "⚠️ question 없음")
    answer = result.get("answer", "⚠️ answer 없음")
    raw_table_df = result.get("raw_table", pd.DataFrame())
    raw_table = raw_table_df.to_dict(orient="records") if isinstance(raw_table_df, pd.DataFrame) else "⚠️ raw_table 없음"
    table_id = result.get("table_id", "⚠️ table_id 없음")
    table_text = result.get("table_text", "⚠️ table_text 없음")

    # TCRF 중 column과 관련 있는 정보
    ## Answer Entity
    predicted_answer_entity = result.get("predicted_answer_entity", "⚠️ 예측된 엔티티 없음")
    predicted_answer_entity_scores = result.get("predicted_answer_entity_scores", {})

    ## Answer Essential column
    essential_columns = result.get("essential_columns", [])

    ## column relevance scoring
    column_description = result.get("column_description", {})
    column_relevance_scores = result.get("column_relevance_scores", {})
    llm_score = result.get("llm_score", {})
    llm_debug_scores = result.get("llm_debug_scores", {})
    column_similarity_scores = result.get("column_similarity_scores", {})

    ## column clustering
    column_clusters = result.get("column_clusters", {})
    cluster_centers = result.get("cluster_centers", [])

    ## clustering evaluation
    selected_cluster = result.get("selected_cluster", -1)
    filtered_columns = result.get("filtered_columns", [])
    voting_results = result.get("voting_results", {})
    ensemble_details = result.get("ensemble_details", {})
    ensemble_confidences = result.get("ensemble_confidences", {})

    ## row ranker
    selected_rows = result.get("selected_rows", [])
    row_similarity_scores = result.get("row_similarity_scores", [])
    top_row_indices = result.get("top_row_indices", [])
    tfidf_scores = result.get("tfidf_scores", [])
    bm25_scores = result.get("bm25_scores", [])
    dense_scores = result.get("dense_scores", [])

    ## final table selection
    final_table_text = result.get("final_table_text", "⚠️ table text 없음")
    filtered_df_raw = result.get("filtered_df", pd.DataFrame())
    filtered_df = filtered_df_raw.to_dict(orient="records") if isinstance(filtered_df_raw, pd.DataFrame) else "⚠️ filtered_df 없음"

    return {
        "index": index,
        "question": question,
        "answer": answer,
        "raw_table": raw_table,
        "table_id": table_id,
        "table_text": table_text,
        "predicted_answer_entity": predicted_answer_entity,
        "predicted_answer_entity_scores": predicted_answer_entity_scores,
        "essential_columns": essential_columns,
        "column_description": column_description,
        "column_relevance_scores": column_relevance_scores,
        "llm_score": llm_score,
        "llm_debug_scores": llm_debug_scores,
        "column_similarity_scores": column_similarity_scores,
        "column_clusters": column_clusters,
        "cluster_centers": cluster_centers,
        "selected_cluster": selected_cluster,
        "filtered_columns": filtered_columns,
        "voting_results": voting_results,
        "ensemble_details": ensemble_details,
        "ensemble_confidences": ensemble_confidences,
        "selected_rows": selected_rows.to_dict(orient="records") if isinstance(selected_rows, pd.DataFrame) else selected_rows,
        "row_similarity_scores": row_similarity_scores,
        "top_row_indices": top_row_indices,
        "tfidf_scores": tfidf_scores,
        "bm25_scores": bm25_scores,
        "dense_scores": dense_scores,
        "final_table_text": final_table_text,
        "filtered_df": filtered_df
    }

def load_existing_results(output_path):
    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                idx = str(entry.get("index"))
                results[idx] = entry
    return results

def save_result(index, result, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    def recursive_format(value):
        if isinstance(value, dict):
            return {k: recursive_format(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [recursive_format(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(recursive_format(v) for v in value)
        elif isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        elif isinstance(value, (float, int, np.float64, np.float32, np.int64, np.int32)):
            return round(float(value), 4)
        else:
            return value

    # Load all indices from the test.json
    json_df = pd.read_json("data/test.json")
    output_path = "outputs/tcrf_results.jsonl"
    existing_results = load_existing_results(output_path)

    valid_indices = list(json_df.index.astype(str))
    total = len(valid_indices)

    for idx, index_str in enumerate(tqdm(valid_indices, desc="Processing", unit="index")):
        # index_str is already defined from valid_indices
        if index_str in existing_results:
            print(f"🔁 Index {index_str} already exists. Skipping...")
            continue

        print(f"🚀 Processing index {index_str}...")
        try:
            result = make_response(int(index_str))
            formatted_result = {k: recursive_format(v) for k, v in result.items()}
            save_result(int(index_str), formatted_result, output_path)
            print(f"✅ Saved index {index_str}")
        except Exception as e:
            print(f"❌ Failed at index {index_str}: {e}")