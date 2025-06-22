# Advanced Cluster Selection Methods for Column Filtering
from langchain_core.runnables import RunnableLambda
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter

def calculate_cluster_quality(column_clusters, score_dict):
    """클러스터의 품질 점수 계산 (응집도 + 분리도)"""
    clusters = {}
    for col, cluster_id in column_clusters.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(col)
    
    quality_scores = []
    
    for cluster_id in sorted(clusters.keys()):
        columns = clusters[cluster_id]
        
        # 클러스터 내 응집도
        cluster_vectors = np.array([score_dict[col] for col in columns])
        cohesion = 1.0 - np.mean([np.std(cluster_vectors[:, i]) for i in range(cluster_vectors.shape[1])])
        
        # 다른 클러스터와의 분리도
        other_columns = [col for col, cid in column_clusters.items() if cid != cluster_id]
        if other_columns:
            other_vectors = np.array([score_dict[col] for col in other_columns])
            separation = np.mean(np.abs(cluster_vectors.mean(axis=0) - other_vectors.mean(axis=0)))
        else:
            separation = 1.0
        
        quality_scores.append(cohesion * 0.6 + separation * 0.4)
    
    return np.array(quality_scores)

def analyze_question_complexity(question):
    """질문의 복잡도 분석"""
    words = question.lower().split()
    
    # 복잡도 지표들
    word_count = len(words)
    has_multiple_conditions = len(re.findall(r'\band\b|\bor\b', question.lower())) > 0
    has_aggregation = any(word in words for word in ['sum', 'count', 'average', 'total', 'maximum', 'minimum'])
    has_subquery_pattern = 'in' in words or 'exists' in words
    
    complexity_score = word_count / 10  # 기본 점수
    if has_multiple_conditions: complexity_score += 0.5
    if has_aggregation: complexity_score += 0.3
    if has_subquery_pattern: complexity_score += 0.7
    
    return min(complexity_score, 3.0)  # 최대 3.0으로 제한

def calculate_column_diversity(columns):
    """컬럼들의 다양성 점수 계산"""
    # 컬럼명 기반 다양성 (간단한 휴리스틱)
    prefixes = set()
    suffixes = set()
    
    for col in columns:
        parts = col.lower().split('_')
        if len(parts) > 1:
            prefixes.add(parts[0])
            suffixes.add(parts[-1])
    
    diversity = (len(prefixes) + len(suffixes)) / (2 * len(columns) + 1e-8)
    return min(diversity, 1.0)

def calculate_info_density(columns, score_dict):
    """정보 밀도 계산 (높은 점수 컬럼들의 비율)"""
    high_score_threshold = 0.7
    high_score_count = 0
    
    for col in columns:
        if np.mean(score_dict[col]) > high_score_threshold:
            high_score_count += 1
    
    return high_score_count / len(columns)

def calculate_size_match(cluster_size, question_complexity):
    """클러스터 크기와 질문 복잡도 매칭 점수"""
    # 복잡한 질문일수록 더 많은 컬럼이 필요할 수 있음
    optimal_size = min(question_complexity * 3, 10)  # 최대 10개 컬럼
    size_diff = abs(cluster_size - optimal_size)
    
    return 1.0 / (1.0 + size_diff * 0.2)

def classify_question_type(question):
    """질문 유형 분류"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['sum', 'total', 'count', 'average', 'max', 'min']):
        return 'aggregation'
    elif any(word in question_lower for word in ['where', 'filter', 'condition', 'specific']):
        return 'filtering'
    elif any(word in question_lower for word in ['show', 'list', 'display', 'all']):
        return 'exploration'
    elif any(word in question_lower for word in ['compare', 'difference', 'versus']):
        return 'comparison'
    else:
        return 'general'

def get_question_type_weight(question_type, cluster_size):
    """질문 유형별 가중치 계산"""
    weights = {
        'aggregation': 0.8 if cluster_size <= 5 else 0.6,  # 집계는 적은 컬럼 선호
        'filtering': 1.0,  # 필터링은 클러스터 크기에 중립
        'exploration': 0.6 if cluster_size >= 5 else 0.8,  # 탐색은 많은 컬럼 선호
        'comparison': 0.9,  # 비교는 중간 정도
        'general': 0.7
    }
    return weights.get(question_type, 0.7)

def calculate_adaptive_threshold(confidence_scores):
    """적응형 임계값 계산"""
    if not confidence_scores:
        return 0.5
    
    mean_score = np.mean(confidence_scores)
    std_score = np.std(confidence_scores)
    
    # 평균 - 0.5 * 표준편차를 임계값으로 설정
    threshold = mean_score - 0.5 * std_score
    return max(threshold, 0.3)  # 최소 0.3 보장

# Method 1: Semantic Similarity + Cluster Center Analysis
def semantic_cluster_selection_fn(state):
    """
    질문과 클러스터 중심점 간의 의미적 유사도를 계산하여 최적 클러스터 선택
    """
    question = state["question"]
    column_clusters = state["column_clusters"]
    cluster_centers = np.array(state["cluster_centers"])
    score_dict = state["column_relevance_scores"]
    
    # 질문 벡터를 column relevance score 평균으로 표현 (dimension match)
    question_vector = np.mean(list(score_dict.values()), axis=0)

    # 각 클러스터 중심점과 질문 벡터 간 유사도 계산
    similarities = cosine_similarity([question_vector], cluster_centers)[0]
    
    # 클러스터 품질 점수 계산 (응집도 + 분리도)
    cluster_quality_scores = calculate_cluster_quality(column_clusters, score_dict)
    
    # 최종 점수: 유사도 * 품질 점수
    final_scores = similarities * cluster_quality_scores
    selected_cluster = np.argmax(final_scores)
    
    # 선택된 클러스터의 컬럼들 필터링
    selected_columns = [col for col, cluster in column_clusters.items() 
                       if cluster == selected_cluster]
    
    return {
        **state, 
        "selected_cluster": int(selected_cluster),
        "filtered_columns": selected_columns,
        "cluster_selection_scores": final_scores.tolist(),
        "semantic_similarity_scores": similarities.tolist(),
        "semantic_cluster_quality_scores": cluster_quality_scores.tolist(),
        "semantic_final_scores": final_scores.tolist()
    }

# Method 2: Multi-Criteria Decision Making (MCDM) Approach
def mcdm_cluster_selection_fn(state):
    """
    다중 기준 의사결정을 통한 클러스터 선택
    - 기준: 관련성, 다양성, 정보량, 질문 복잡도 매칭
    """
    question = state["question"]
    column_clusters = state["column_clusters"]
    score_dict = state["column_relevance_scores"]
    
    clusters = {}
    for col, cluster_id in column_clusters.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(col)
    
    cluster_scores = {}
    question_complexity = analyze_question_complexity(question)
    
    for cluster_id, columns in clusters.items():
        # 기준 1: 평균 관련성 점수
        relevance_score = np.mean([np.mean(score_dict[col]) for col in columns])
        
        # 기준 2: 컬럼 다양성 (서로 다른 도메인의 컬럼들)
        diversity_score = calculate_column_diversity(columns)
        
        # 기준 3: 정보 밀도 (높은 점수의 컬럼 비율)
        info_density = calculate_info_density(columns, score_dict)
        
        # 기준 4: 질문 복잡도와 클러스터 크기 매칭
        size_match_score = calculate_size_match(len(columns), question_complexity)
        
        # 가중 평균 계산
        weights = [0.4, 0.2, 0.2, 0.2]  # 관련성을 가장 중요하게
        final_score = (relevance_score * weights[0] + 
                      diversity_score * weights[1] + 
                      info_density * weights[2] + 
                      size_match_score * weights[3])
        
        cluster_scores[cluster_id] = final_score
    
    selected_cluster = max(cluster_scores, key=cluster_scores.get)
    selected_columns = clusters[selected_cluster]
    
    return {
        **state,
        "selected_cluster": selected_cluster,
        "filtered_columns": selected_columns,
        "cluster_scores": cluster_scores,
        "mcdm_relevance_scores": {cid: np.mean([np.mean(score_dict[col]) for col in clusters[cid]]) for cid in clusters},
        "mcdm_diversity_scores": {cid: calculate_column_diversity(clusters[cid]) for cid in clusters},
        "mcdm_info_density_scores": {cid: calculate_info_density(clusters[cid], score_dict) for cid in clusters},
        "mcdm_size_match_scores": {cid: calculate_size_match(len(clusters[cid]), question_complexity) for cid in clusters},
        "mcdm_cluster_scores": cluster_scores
    }

# Method 3: Adaptive Threshold with Confidence Scoring
def adaptive_threshold_selection_fn(state):
    """
    적응형 임계값과 신뢰도 점수를 활용한 동적 클러스터 선택
    """
    question = state["question"]
    column_clusters = state["column_clusters"]
    score_dict = state["column_relevance_scores"]
    
    # 질문 타입 분석 (집계형, 필터링형, 탐색형 등)
    question_type = classify_question_type(question)
    
    clusters = {}
    for col, cluster_id in column_clusters.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(col)
    
    cluster_confidences = {}
    
    for cluster_id, columns in clusters.items():
        # 클러스터 내 점수 분포 분석
        all_scores = [score_dict[col] for col in columns]
        score_matrix = np.array(all_scores)
        
        # 신뢰도 계산: 일관성 + 강도
        consistency = 1.0 - np.std(score_matrix.mean(axis=1)) / (np.mean(score_matrix) + 1e-8)
        strength = np.mean(score_matrix)
        
        # 질문 타입별 가중치 적용
        type_weight = get_question_type_weight(question_type, len(columns))
        
        confidence = (consistency * 0.4 + strength * 0.4 + type_weight * 0.2)
        cluster_confidences[cluster_id] = confidence
    
    # 적응형 임계값 설정
    threshold = calculate_adaptive_threshold(list(cluster_confidences.values()))
    
    # 임계값을 넘는 클러스터들 중 최고 점수 선택
    valid_clusters = {k: v for k, v in cluster_confidences.items() if v >= threshold}
    
    if valid_clusters:
        selected_cluster = max(valid_clusters, key=valid_clusters.get)
    else:
        selected_cluster = max(cluster_confidences, key=cluster_confidences.get)
    
    selected_columns = clusters[selected_cluster]
    
    return {
        **state,
        "selected_cluster": selected_cluster,
        "filtered_columns": selected_columns,
        "cluster_confidences": cluster_confidences,
        "selection_threshold": threshold,
        "adaptive_cluster_confidences": cluster_confidences,
        "adaptive_selection_threshold": threshold
    }

# Method 4: Ensemble Selection with Voting
def ensemble_cluster_selection_fn(state):
    """
    여러 선택 방법의 앙상블을 통한 robust한 클러스터 선택
    - state["top_k_per_cluster"]를 통해 각 클러스터에서 선택할 상위 컬럼 수 조정 가능 (기본값 1)
    """
    # 각 방법으로 클러스터 선택
    table_columns = state['table_columns']
    
    result1 = semantic_cluster_selection_fn(state)
    result2 = mcdm_cluster_selection_fn(state)
    result3 = adaptive_threshold_selection_fn(state)
    
    # 투표 시스템
    votes = [result1["selected_cluster"], 
             result2["selected_cluster"], 
             result3["selected_cluster"]]
    
    vote_counts = Counter(votes)
    # Ensure all clusters are included in the vote count
    all_cluster_ids = set(state["column_clusters"].values())
    for cluster_id in all_cluster_ids:
        if cluster_id not in vote_counts:
            vote_counts[cluster_id] = 0
    
    # 최다 득표 클러스터 선택, 동점시 confidence 점수로 결정
    if len(set(votes)) == len(votes):  # 모두 다른 경우
        # 각 방법의 신뢰도 점수를 비교하여 선택
        confidences = [
            max(result1.get("cluster_selection_scores", [0])),
            max(result2.get("cluster_scores", {}).values()) if result2.get("cluster_scores") else 0,
            max(result3.get("cluster_confidences", {}).values()) if result3.get("cluster_confidences") else 0
        ]
        selected_cluster = votes[np.argmax(confidences)]
    else:
        selected_cluster = vote_counts.most_common(1)[0][0]
    
    # 선택된 클러스터의 컬럼들
    column_clusters = state["column_clusters"]
    score_dict = state["column_relevance_scores"]
    selected_columns = [col for col, cluster in column_clusters.items() 
                       if cluster == selected_cluster]

    # Ensure at least one high-relevance column is included from each non-selected cluster
    non_selected_clusters = {cid: [] for cid in set(column_clusters.values()) if cid != selected_cluster}
    for col, cid in column_clusters.items():
        if cid in non_selected_clusters:
            non_selected_clusters[cid].append(col)

    # 선택되지 않은 클러스터에서 뽑아올 column의 수
    top_k_per_cluster = 1
    additional_columns = []
    for cid, cols in non_selected_clusters.items():
        if cols:
            sorted_cols = sorted(cols, key=lambda c: np.mean(score_dict[c]), reverse=True)
            top_k_cols = sorted_cols[:top_k_per_cluster]
            additional_columns.extend(top_k_cols)
    
    #print(f"[EnsembleSelection] Votes: {vote_counts}")
    #print(f"[EnsembleSelection] Selected cluster: {selected_cluster}")
    #print(f"[EnsembleSelection] Selected Cluster columns: {selected_columns}")
    #print(f"[EnsembleSelection] Other cluster selected columns: {additional_columns}")

    selected_columns += additional_columns

    # Add essential columns if not already included
    essential_columns = state.get("essential_columns", [])
    for col in essential_columns:
        if col not in selected_columns:
            selected_columns.append(col)

    if "Row Header" in table_columns and "Row Header" not in selected_columns:
        selected_columns += ["Row Header"]

    

    #print(f"[EnsembleSelection] Final selected columns: {selected_columns}")


    return {
        **state,
        "selected_cluster": selected_cluster,
        "filtered_columns": selected_columns,
        "voting_results": vote_counts,
        "ensemble_details": {
            "semantic_choice": result1["selected_cluster"],
            "mcdm_choice": result2["selected_cluster"],
            "adaptive_choice": result3["selected_cluster"]
        },
        "ensemble_confidences": {
            "semantic": max(result1.get("cluster_selection_scores", [0])),
            "mcdm": max(result2.get("cluster_scores", {}).values()) if result2.get("cluster_scores") else 0,
            "adaptive": max(result3.get("cluster_confidences", {}).values()) if result3.get("cluster_confidences") else 0
        }
    }

# Create runnable node for ensemble method only
ensemble_evluation_node = RunnableLambda(ensemble_cluster_selection_fn)