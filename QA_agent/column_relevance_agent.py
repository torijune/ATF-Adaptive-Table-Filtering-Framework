import os
from typing import Dict, List, Tuple
import openai
import numpy as np
from collections import defaultdict
import statistics

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

'''
LLM Column Description Robustness 업그레이드 전략
1. 현재의 column 속 실제 unique들을 보여주는 전략 유지
2. 주어진 내용에 대한 묘사를 Benchmark 성능 평가를 하는 것을 중간에 도입하여 일정 점수 이하면 reject을 하여 재생성하는 loop 추가
3. 고민중...
'''
def column_description(table_columns, question, raw_table=None) -> Dict:
    """Enhanced column description with more stable sampling"""
    describing_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # Lower temperature for consistency

    column_blocks = []
    for col in table_columns:
        if raw_table is not None:
            try:
                col_values = raw_table[col].dropna().astype(str)
                val_counts = col_values.value_counts()
                
                # More stable sampling strategy
                sample_values = []
                
                # First, get most frequent values (if any)
                frequent_vals = val_counts[val_counts > 1].head(3).index.tolist()
                sample_values.extend(frequent_vals)
                
                # Then, get some unique values to show diversity
                unique_vals = val_counts[val_counts == 1].head(2).index.tolist()
                sample_values.extend(unique_vals)
                
                # Remove duplicates while preserving order
                seen = set()
                sample_values = [x for x in sample_values if not (x in seen or seen.add(x))]
                
                if sample_values:
                    sample_text = f" (Examples: {', '.join(sample_values[:5])})"
                else:
                    sample_text = f" (Data type: {raw_table[col].dtype})"
                    
            except Exception as e:
                sample_text = f" (Data type: {raw_table[col].dtype if col in raw_table.columns else 'unknown'})"
        else:
            sample_text = ""
        column_blocks.append(f"{col}{sample_text}")

    formatted_columns = "\n".join(column_blocks)

    describe_prompt = f"""
    You are a table analysis expert. Generate concise, consistent descriptions for each column.

    Question Context: {question}
    
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
            
    return descriptions

'''
LLM Scoring Robustness 업그레이드 전략
1. 현재의 다수(3회)의 LLM 호출을 통해 Multi-Scoring 후 Ensemble하는 전략 유지
2. question에서의 조건부(특히, AND로 묶인 것들)의 객체가 되는 변수들을 잘 잡을 수 있는 prompting이 필요
3. 추후에 Bi-Encoder Similiarity과 함께 Clustering하긴하지만 LLM Scoring 과정에 수식적인 부분으로 Robustness를 보장해주면 좋을 것 같음
4. 고민중
'''
def ensemble_llm_score(column_descriptions: Dict[str, str], question: str, num_iterations: int = 3) -> Dict:
    """Run LLM scoring multiple times and ensemble the results"""
    scoring_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # Set to 0 for deterministic results
    
    formatted_columns = "\n".join([f"{col}: {desc}" for col, desc in column_descriptions.items()])
    
    scoring_prompt = f"""
    You are an expert in question-answering with tabular data.
    
    Question: {question}
    
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
    
    # Ensemble the scores
    if not all_scores:
        return {col: 0.0 for col in column_descriptions.keys()}
    
    ensembled_scores = {}
    for col in column_descriptions.keys():
        col_scores = [scores.get(col, 0.0) for scores in all_scores if col in scores]
        if col_scores:
            # Use median for robustness against outliers
            ensembled_scores[col] = statistics.median(col_scores)
        else:
            ensembled_scores[col] = 0.0
    
    return ensembled_scores

def stable_cosine_similarity(column_descriptions: Dict[str, str], question: str) -> Dict:
    """More stable semantic similarity using sentence transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use a more stable model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Normalize question and descriptions
        normalized_question = question.lower().strip()
        
        column_names = list(column_descriptions.keys())
        normalized_descriptions = [
            f"{col.lower().replace('_', ' ')} {desc.lower()}" 
            for col, desc in column_descriptions.items()
        ]
        
        # Encode question and descriptions
        question_embedding = model.encode([normalized_question])
        desc_embeddings = model.encode(normalized_descriptions)
        
        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(question_embedding, desc_embeddings)[0]
        
        # Normalize to 0-1 range and apply slight smoothing
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
    table_columns = list(raw_table.columns)

    print(f"[ColumnRelevance] Processing {len(table_columns)} columns")
    print(f"[ColumnRelevance] Question: {question}")

    # Generate stable column descriptions
    column_desc = column_description(table_columns, question, raw_table)
    print(f"[ColumnRelevance] Column Descriptions Generated")

    # Get ensemble LLM scores (multiple runs for stability)
    llm_scores = ensemble_llm_score(column_desc, question, num_iterations=3)
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

# Updated node
column_relevance_node = RunnableLambda(column_relevance_fn)