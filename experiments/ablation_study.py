import json
import pandas as pd
from langchain_core.runnables import RunnableLambda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import os

def get_data(index):
    jsonl_path = "outputs/tcrf_results.jsonl"
    target_index = index
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # index가 문자열이나 정수로 저장되어 있을 수 있으므로, 비교 시 모두 문자열로 변환
            if float(entry.get("index")) == float(target_index):
                answer = entry['answer']
                question = entry['question']
                raw_table = pd.DataFrame(entry['raw_table'])
                filtered_columns = entry['filtered_columns']
                column_relevance_scores = entry['column_relevance_scores']
                top_row_indices = entry['top_row_indices']

    return raw_table, filtered_columns, answer, question, column_relevance_scores, top_row_indices

def normalize_softmax(arr):
    arr = np.array(arr)
    e_x = np.exp(arr - np.max(arr))
    return e_x / e_x.sum()

# w/o column filtering
def row_column_filtering(index):

    raw_table, filtered_columns, answer, question, column_relevance_scores, top_row_indices = get_data(index)

    column_filtered_df = raw_table[filtered_columns]
    
    # 전체 row를 question과 유사도 기반으로 랭킹
    row_texts = raw_table.astype(str).agg(" ".join, axis=1).tolist()

    # TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + row_texts)
    question_vector = vectors[0]
    row_vectors = vectors[1:]
    tfidf_sim = cosine_similarity(question_vector, row_vectors)[0]

    # BM25
    tokenized_rows = [text.lower().split() for text in row_texts]
    tokenized_query = question.lower().split()
    bm25 = BM25Okapi(tokenized_rows)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Dense
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    dense_question_emb = dense_model.encode(question, convert_to_tensor=True)
    dense_row_embs = dense_model.encode(row_texts, convert_to_tensor=True)
    dense_sim = util.pytorch_cos_sim(dense_question_emb, dense_row_embs)[0].cpu().numpy()

    # Score fusion
    tfidf_norm = normalize_softmax(tfidf_sim)
    bm25_norm = normalize_softmax(bm25_scores)
    dense_norm = normalize_softmax(dense_sim)
    fusion_scores = 0.4 * tfidf_norm + 0.3 * bm25_norm + 0.3 * dense_norm

    # Top-K row 선택
    top_k_ratio = 0.4
    top_k = max(1, int(np.ceil(len(row_texts) * top_k_ratio)))
    top_indices = fusion_scores.argsort()[-top_k:][::-1]
    row_filtered_df = raw_table.iloc[top_indices].reset_index(drop=True)

    print("\nRaw Table Shape: ", raw_table.shape)
    print("\nw/o Row Filtering Table Shape: ", column_filtered_df.shape)
    print("\nw/o Column Filtering Table Shape: ", row_filtered_df.shape)
    print("Raw table column length: ", len(raw_table.columns), "Raw table row length: ", len(raw_table))
    print("w/o Row filter row length: ", len(column_filtered_df), "w/o Column filter column length: ", len(row_filtered_df.columns))


    return column_filtered_df, row_filtered_df, answer, question

def Top_k_vs_KMenas(index):

    raw_table, filtered_columns, answer, question, column_relevance_scores, top_row_indices = get_data(index)

    # ✅ L2 Norm 기반 점수 계산
    combined_scores = {
        col: np.linalg.norm(score)
        for col, score in column_relevance_scores.items()
    }

    # ✅ 점수 목록 추출
    all_scores = list(combined_scores.values())

    # ✅ 전체 컬럼 수 계산
    num_columns = len(combined_scores)

    # ✅ 조건에 따른 임계값 또는 Top-3 선택
    if num_columns >= 10:
        threshold = np.percentile(all_scores, 60)  # 상위 40% → 하위 60%를 임계값으로
        selected_columns = [col for col, score in combined_scores.items() if score >= threshold]
    else:
        # 점수 내림차순 정렬 후 상위 3개 선택
        selected_columns = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]

    print(f"[DEBUG] index: {index}")
    print(f"[DEBUG] raw_table.columns: {list(raw_table.columns)}")
    print(f"[DEBUG] selected_columns: {selected_columns}")
    print(f"[DEBUG] top_row_indices: {top_row_indices}")
        
    selected_rows = raw_table[selected_columns].iloc[top_row_indices].reset_index(drop=True)
    TopK_filtered_df = selected_rows[selected_columns]

    return selected_columns, TopK_filtered_df, answer, question

def ablation_dataset(index):
    print("="*25,f"{index} Processing", "="*25)

    column_filtered_df, row_filtered_df, answer, question = row_column_filtering(index)
    selected_columns, TopK_filtered_df, answer, question = Top_k_vs_KMenas(index)

    print("="*25,f"{index} Completed", "="*25)

    answer = answer
    question = question
    column_filtered_df = column_filtered_df
    row_filtered_df = row_filtered_df
    TopK_filtered_df = TopK_filtered_df
    TopK_selected_columns = selected_columns

    # ✅ 결과 저장용 딕셔너리 구성
    result = {
        "index": index,
        "question": question,
        "answer": answer,
        "column_filtered_df": column_filtered_df.to_dict(orient="records"),
        "row_filtered_df": row_filtered_df.to_dict(orient="records"),
        "TopK_filtered_df": TopK_filtered_df.to_dict(orient="records"),
        "TopK_selected_columns": TopK_selected_columns
    }

    # ✅ JSONL로 저장 (append 모드)
    with open("outputs/ablation_dataset.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

json_df = pd.read_json("Open-WikiTable_data/test.json")
valid_indices = list(json_df.index.astype(str))


from tqdm import tqdm

def run_ablation_for_all_with_tqdm():
    processed_indices = set()

    # ✅ 기존 처리된 index 불러오기
    if os.path.exists("outputs/ablation_dataset.jsonl"):
        with open("outputs/ablation_dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_indices.add(str(entry.get("index")))
                except:
                    continue

    # ✅ tqdm으로 처리되지 않은 index만 추림
    unprocessed_indices = [idx for idx in valid_indices if str(idx) not in processed_indices]

    print(f"🚀 전체 {len(valid_indices)}개 중, 처리되지 않은 {len(unprocessed_indices)}개 인덱스 실행 시작")

    for index in tqdm(unprocessed_indices, desc="Generating Ablation Dataset"):
        try:
            ablation_dataset(index)
        except Exception as e:
            print(f"\n❌ Failed to process index {index}: {e}")

run_ablation_for_all_with_tqdm()