import os
from typing import Dict
import openai
import numpy as np

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def column_description(table_columns, question, raw_table=None) -> Dict:
    from collections import defaultdict
    describing_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    column_blocks = []
    for col in table_columns:
        if raw_table is not None:
            try:
                col_values = raw_table[col].dropna().astype(str)
                val_counts = col_values.value_counts()
                frequent_vals = val_counts[val_counts > 1].index.tolist()
                if frequent_vals:
                    sample_text = f" (categorical, e.g., {', '.join(frequent_vals[:5])})"
                else:
                    unique_vals = col_values.unique().tolist()
                    sample_text = f" (non-categorical, e.g., {', '.join(unique_vals[:5])})" if unique_vals else ""
            except Exception:
                sample_text = ""
        else:
            sample_text = ""
        column_blocks.append(f"{col}{sample_text}")

    formatted_columns = "\n".join(column_blocks)

    Describe_PROMPT = f"""
    You are a table analysis expert.

    Your task is to generate a short description for each table column in the context of the given question. 
    Use the column name and sample values to infer meaning.

    Question: {question}

    Columns with Sample Values:
    {formatted_columns}

    Return format:
    Column1: description
    Column2: description
    ...
    """

    response = describing_llm.invoke(Describe_PROMPT)

    lines = response.content.strip().split("\n")
    descriptions = {}
    for line in lines:
        if ":" in line:
            col, desc = line.split(":", 1)
            descriptions[col.strip()] = desc.strip()
            
    return descriptions

def cosine_sim_with_context_aware(column_descriptions, question, table_context="") -> Dict:
    """
    테이블 컨텍스트를 고려한 cross-encoder 유사도 측정
    """
    from sentence_transformers import CrossEncoder
    
    model = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")
    
    column_names = list(column_descriptions.keys())
    column_texts = list(column_descriptions.values())
    
    # 컨텍스트가 있는 경우 질문에 추가
    enhanced_question = f"{table_context} {question}".strip() if table_context else question
    
    # Create pairs with enhanced context
    pairs = []
    for desc in column_texts:
        # Column description에도 컨텍스트 정보 추가 가능
        enhanced_desc = f"Table column description: {desc}"
        pairs.append([enhanced_question, enhanced_desc])
    
    # Cross-encoder 점수 계산
    similarity_scores = model.predict(pairs)
    
    print(f"\n Context-aware cross-encoder scores: {similarity_scores}")
    
    # STSB 모델은 0-1 점수를 반환하므로 그대로 사용
    return {col: float(score) for col, score in zip(column_names, similarity_scores)}

def llm_score(column_descriptions: Dict[str, str], question: str) -> Dict:
    scoring_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, top_p=0.95)
    formatted_columns = "\n".join([f"{col}: {desc}" for col, desc in column_descriptions.items()])
    
    Scoring_PROMPT = f"""
    You are an expert in question answering using tabular data.

    Given the following question and the descriptions of table columns, assign a relevance score (from 0 to 1) to each column indicating how important it is for answering the question.

    Scoring Guidelines:
    - 1.0: Essential for answering the question
    - 0.7-0.9: Highly relevant, likely needed
    - 0.4-0.6: Moderately relevant, might be useful
    - 0.1-0.3: Low relevance, probably not needed
    - 0.0: Not relevant at all

    Columns and Descriptions:
    {formatted_columns}

    Question:
    {question}

    Return format (strictly adhere):
    Column1: score
    Column2: score
    ...
    """

    response = scoring_llm.invoke(Scoring_PROMPT)
    lines = response.content.strip().split("\n")
    scores = {}
    for line in lines:
        if ":" in line:
            col, score = line.split(":", 1)
            try:
                scores[col.strip()] = float(score.strip())
            except ValueError:
                continue
    return scores

def column_relevance_fn(state):

    raw_table = state["raw_table"]
    question = state["question"]
    table_columns = list(raw_table.columns)

    print(f"[ColumnRelevance] Number of columns: {len(table_columns)}")
    print(f"[ColumnRelevance] Columns: {table_columns}")
    # print(f"[ColumnRelevance] Question: {question}")

    column_desc = column_description(table_columns, question, raw_table)
    print(f"[ColumnRelevance] Column Descriptions: {column_desc}")

    llm_scores = llm_score(column_desc, question)
    print(f"\n [ColumnRelevance] LLM Scores: {llm_scores}")

    cosine_scores = cosine_sim_with_context_aware(column_desc, question)
    print(f"\n [ColumnRelevance] Cosine Scores: {cosine_scores}")

    merged_scores = {
        col: [llm_scores.get(col, 0.0), cosine_scores.get(col, 0.0)]
        for col in table_columns
    }

    print(f"[ColumnRelevance] Merged Scores: {merged_scores}")

    return {**state, "column_relevance_scores": merged_scores}

column_relevance_node = RunnableLambda(column_relevance_fn)