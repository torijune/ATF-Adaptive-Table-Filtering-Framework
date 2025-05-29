from langchain_core.runnables import RunnableLambda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import torch

def normalize_softmax(arr):
    arr = np.array(arr)
    e_x = np.exp(arr - np.max(arr))
    return e_x / e_x.sum()

def row_ranker_fn(state):
    raw_table = state["raw_table"]
    question = state["question"]
    selected_columns = state["filtered_columns"]

    print(f"\n[RowRanker] ✅ Selected columns: {selected_columns}")

    # 각 row를 선택된 컬럼 기준으로 문자열로 병합
    row_texts = raw_table[selected_columns].astype(str).agg(" ".join, axis=1).tolist()

    # TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + row_texts)
    question_vector = vectors[0]
    row_vectors = vectors[1:]
    tfidf_sim = cosine_similarity(question_vector, row_vectors)[0]

    # BM25 점수 계산
    tokenized_rows = [text.lower().split() for text in row_texts]
    tokenized_query = question.lower().split()
    bm25 = BM25Okapi(tokenized_rows)
    bm25_scores = bm25.get_scores(tokenized_query)

    from sentence_transformers import SentenceTransformer, util
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    dense_question_emb = dense_model.encode(question, convert_to_tensor=True)
    dense_row_embs = dense_model.encode(row_texts, convert_to_tensor=True)
    dense_sim = util.pytorch_cos_sim(dense_question_emb, dense_row_embs)[0].cpu().numpy()

    tfidf_norm = normalize_softmax(tfidf_sim)
    bm25_norm = normalize_softmax(bm25_scores)
    dense_norm = normalize_softmax(dense_sim)
    fusion_scores = 0.4 * tfidf_norm + 0.3 * bm25_norm + 0.3 * dense_norm

    # Top-K row 인덱스 선택 (비율 기반 + 올림 처리)
    top_k_ratio = 0.4  # 예: 상위 20%만 선택
    top_k = max(1, int(np.ceil(len(row_texts) * top_k_ratio)))
    top_indices = fusion_scores.argsort()[-top_k:][::-1]

    print(f"[RowRanker] 🔍 Row fusion scores (TF-IDF + BM25 + Dense, softmax normalized):\n{fusion_scores}")
    print(f"[RowRanker] 🏅 Top-{top_k} row indices (based on {top_k_ratio*100}% of {len(row_texts)} rows): {top_indices.tolist()}")

    # 상위 row만 선택
    selected_rows = raw_table[selected_columns].iloc[top_indices].reset_index(drop=True)
    print(f"[RowRanker] 📋 Selected rows:\n{selected_rows}")

    return {
        **state,
        "selected_rows": selected_rows,
        "row_similarity_scores": fusion_scores.tolist(),
        "top_row_indices": top_indices.tolist()
    }

row_ranker_node = RunnableLambda(row_ranker_fn)