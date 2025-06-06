import os
from typing import Dict
import openai
import numpy as np
from collections import defaultdict
import statistics

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

'''
LLM Column Description Robustness 업그레이드 전략
1. 현재의 column 속 실제 unique들을 보여주는 전략 유지
2. 주어진 내용에 대한 묘사를 Benchmark 성능 평가를 하는 것을 중간에 도입하여 일정 점수 이하면 reject을 하여 재생성하는 loop 추가
3. 고민중...
'''
def column_description(table_columns, question, raw_table=None, predicted_entity=None) -> Dict:

    # Temperature 낮춰서 최대한 보수적으로 출력하도록 -> 일관성을 위해
    describing_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 각각의 column의 특징 (한번만 등장하는 unique한 column & 중복 등장이 있는 범주형 column)
    ## 각각의 column에 맞춰서 다르게 Description에 추가하여 성능을 더 높임
    column_blocks = []
    col_examples_map = {}
    for col in table_columns:
        if raw_table is not None:
            try:
                col_values = raw_table[col].dropna().astype(str)
                val_counts = col_values.value_counts()
                
                # More stable sampling strategy
                
                # 같은 값이 두번이상 등장하는 column -> 범주형으로 판단
                ## 10개 이하이면 모든 것들을 다 넣고
                if len(val_counts[val_counts > 1]) <= 10:
                    frequent_vals = val_counts[val_counts > 1].index.tolist() 
                else:
                ## 11개 이상이면 5개만 넣는 방식
                    frequent_vals = val_counts[val_counts > 1].head(5).index.tolist()
                sample_values = []
                sample_values.extend(frequent_vals)
                
                # 각각 한번만 등장하는 것은 각 row에 unique한 값들이 등장하는 column -> 범주형이 아닌 것으로 판단
                unique_vals = val_counts[val_counts == 1].head(5).index.tolist()
                sample_values.extend(unique_vals)
                
                # head로 넣는 경우에 중복을 방지하기 위해
                seen = set()
                sample_values = [x for x in sample_values if not (x in seen or seen.add(x))]
                
                # 에러 예외처리를 위한 If문 -> 넣을만한게 없으면 그냥 해당 column의 dtype으로 대체
                if sample_values:
                    sample_text = f" (Examples: {', '.join(sample_values[:5])})"
                else:
                    sample_text = f" (Data type: {raw_table[col].dtype})"
                    
            except Exception as e:
                sample_text = f" (Data type: {raw_table[col].dtype if col in raw_table.columns else 'unknown'})"
        else:
            sample_text = ""
        col_examples_map[col] = sample_text
        column_blocks.append(f"{col}{sample_text}")

    formatted_columns = "\n".join(column_blocks)

    describe_prompt = f"""
    You are a table analysis expert. Generate concise, consistent descriptions for each column.

    Question Context: {question}
    {"Expected Answer Type: " + predicted_entity if predicted_entity else ""}

    Columns with Examples:
    {formatted_columns}

    Rules:
    1. Keep descriptions factual and concise (max 15 words)
    2. Focus on what the column represents, not just examples
    3. Use consistent terminology
    4. Include data type when relevant

    Format:
    Column1: description
    Column2: description
    """

    response = describing_llm.invoke(describe_prompt)
    lines = response.content.strip().split("\n")
    descriptions = {}
    
    for line in lines:
        if ":" in line:
            col, desc = line.split(":", 1)
            descriptions[col.strip()] = desc.strip()
            
    # Append the original example text to each description
    for col in descriptions:
        if col in col_examples_map:
            descriptions[col] += f" {col_examples_map[col]}"
            
    return descriptions

'''
LLM Scoring Robustness 업그레이드 전략
1. 현재의 다수(3회)의 LLM 호출을 통해 Multi-Scoring 후 Ensemble하는 전략 유지
2. question에서의 조건부(특히, AND로 묶인 것들)의 객체가 되는 변수들을 잘 잡을 수 있는 prompting이 필요
3. 추후에 Bi-Encoder Similiarity과 함께 Clustering하긴하지만 LLM Scoring 과정에 수식적인 부분으로 Robustness를 보장해주면 좋을 것 같음
4. 고민중
'''

# LLM Socring - LLM Ensemble + Ranking을 통해 LLM의 평가에 Robustness를 더함
def ensemble_llm_score(column_descriptions: Dict[str, str], question: str, predicted_entity=None, num_iterations: int = 3) -> Dict:
    """Run LLM scoring multiple times and ensemble the results using rank-based aggregation."""
    scoring_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # Set to 0 for deterministic results
    
    formatted_columns = "\n".join([f"{col}: {desc}" for col, desc in column_descriptions.items()])
    
    scoring_prompt = f"""
    You are an expert in question-answering with tabular data.
    
    Question: {question}
    {"Expected Answer Type: " + predicted_entity if predicted_entity else ""}
    
    Table Columns:
    {formatted_columns}
    
    Rate each column's relevance for answering the question (0.0 to 1.0):
    - 1.0: Essential/Primary key for the answer
    - 0.8: Highly relevant, likely needed
    - 0.6: Moderately relevant, could be useful
    - 0.4: Somewhat relevant, might provide context
    - 0.2: Low relevance, probably not needed
    - 0.0: Not relevant at all
    
    Be consistent and precise. Consider:
    1. Direct relevance to the question
    2. Potential for filtering/grouping
    3. Contextual importance
    
    Format (exact format required):
    column_name: score
    """

    all_scores = []
    columns = list(column_descriptions.keys())
    
    for i in range(num_iterations):
        try:
            response = scoring_llm.invoke(scoring_prompt)
            lines = response.content.strip().split("\n")
            iteration_scores = {}
            
            for line in lines:
                if ":" in line and not line.strip().startswith("Question") and not line.strip().startswith("Table"):
                    try:
                        col, score = line.split(":", 1)
                        col = col.strip()
                        score_val = float(score.strip())
                        if 0.0 <= score_val <= 1.0:  # Validate score range
                            iteration_scores[col] = score_val
                    except (ValueError, IndexError):
                        continue
            
            if iteration_scores:  # Only add if we got valid scores
                all_scores.append(iteration_scores)
                
        except Exception as e:
            print(f"Warning: LLM scoring iteration {i+1} failed: {e}")
            continue
    
    # rank-based aggregation를 활용하여 LLM Score ensemble 진행
    if not all_scores:
        return {col: 0.0 for col in columns}

    # 각 iteration의 score를 랭크로 변환 후 정규화 (1/n ~ 1)
    rank_matrix = []
    for scores in all_scores:
        # columns 순서대로 값 추출 (없으면 0.0)
        values = [scores.get(c, 0.0) for c in columns]
        # 랭크 산출 -> LLM의 Score가 높을수록 좋은 랭크
        ranks = np.argsort(np.argsort(values))  # 0이 최저, n-1이 최고
        norm_ranks = (ranks + 1) / len(columns)  # 1/n ~ 1
        rank_matrix.append(dict(zip(columns, norm_ranks)))

    # column별로 median rank 산출
    ensembled_scores = {}
    for c in columns:
        col_ranks = [ranks[c] for ranks in rank_matrix]
        ensembled_scores[c] = float(np.median(col_ranks))

    # --- Soft Thresholding based on LLM score variance ---
    # Columns with high variance (low reliability) get penalized
    # Final score = mean_score * (1 / (1 + std_dev))
    # Step 1. Extract raw scores per column across iterations
    score_matrix = {col: [] for col in columns}
    for scores in all_scores:
        for col in columns:
            score_matrix[col].append(scores.get(col, 0.0))

    # Compute mean, std, and reliability per column
    mean_std_reliability = {}
    for col in columns:
        col_scores = score_matrix[col]
        mean = float(np.mean(col_scores))
        std = float(np.std(col_scores))
        reliability = 1.0 / (1.0 + std)  # lower std = higher reliability
        mean_std_reliability[col] = (mean, std, reliability)

    # Apply soft threshold adjustment
    adjusted_scores = {col: mean * reliability for col, (mean, std, reliability) in mean_std_reliability.items()}

    return adjusted_scores


def stable_cosine_similarity(column_descriptions: Dict[str, str], question: str) -> Dict:
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Normalize question and descriptions
        normalized_question = question.lower().strip()
        
        column_names = list(column_descriptions.keys())
        normalized_descriptions = [
            f"{col.lower().replace('_', ' ')} {desc.lower()}" 
            for col, desc in column_descriptions.items()
        ]
        
        # question descriptions 인코딩
        question_embedding = model.encode([normalized_question])
        desc_embeddings = model.encode(normalized_descriptions)
        
        # cosine similarity 계산
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(question_embedding, desc_embeddings)[0]
        
        # 0-1으로 정규화 진행
        min_sim = similarities.min()
        max_sim = similarities.max()
        if max_sim > min_sim:
            normalized_sims = (similarities - min_sim) / (max_sim - min_sim)
        else:
            normalized_sims = similarities
            
        return {col: float(score) for col, score in zip(column_names, normalized_sims)}
        
    except ImportError:
        print("Warning: sentence_transformers not available, using fallback method")
        # Fallback to simple keyword matching
        return keyword_similarity_fallback(column_descriptions, question)

def keyword_similarity_fallback(column_descriptions: Dict[str, str], question: str) -> Dict:
    """Fallback similarity method using keyword overlap"""
    question_words = set(question.lower().split())
    
    scores = {}
    for col, desc in column_descriptions.items():
        col_words = set(col.lower().replace('_', ' ').split())
        desc_words = set(desc.lower().split())
        all_col_words = col_words.union(desc_words)
        
        # Calculate Jaccard similarity
        intersection = len(question_words.intersection(all_col_words))
        union = len(question_words.union(all_col_words))
        
        if union == 0:
            scores[col] = 0.0
        else:
            scores[col] = intersection / union
    
    return scores

# Removed weighted_ensemble_scores function as we're using the original format

def column_relevance_fn(state):
    """Enhanced column relevance function with improved robustness"""
    
    raw_table = state["raw_table"]
    question = state["question"]
    predicted_entity = state.get("predicted_answer_entity", None)
    table_columns = list(raw_table.columns)

    print(f"[ColumnRelevance] Processing {len(table_columns)} columns")
    print(f"[ColumnRelevance] Question: {question}")

    # Generate stable column descriptions
    column_desc = column_description(table_columns, question, raw_table, predicted_entity)
    print(f"[ColumnRelevance] Column Descriptions Generated")

    # Get ensemble LLM scores (multiple runs for stability)
    llm_scores = ensemble_llm_score(column_desc, question, predicted_entity, num_iterations=3)
    print(f"[ColumnRelevance] Ensemble LLM Scores: {llm_scores}")

    # Get stable similarity scores
    similarity_scores = stable_cosine_similarity(column_desc, question)
    print(f"[ColumnRelevance] Similarity Scores: {similarity_scores}")

    # Output in required format [LLM 점수, cosine 유사도 점수]
    merged_scores = {
        col: [llm_scores.get(col, 0.0), similarity_scores.get(col, 0.0)]
        for col in table_columns
    }

    print(f"[ColumnRelevance] Final Merged Scores: {merged_scores}")

    return {**state, 
            "column_relevance_scores": merged_scores,
            "column_description": column_desc
            }

column_relevance_node = RunnableLambda(column_relevance_fn)