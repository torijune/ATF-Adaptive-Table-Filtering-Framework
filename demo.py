import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Tuple

class TableEnhancer:
    def __init__(self, df):
        self.df = df.copy()
        self.enhanced_features = {}
        
    def create_data_DNA(self):
        """각 행의 '데이터 DNA' 생성 - 데이터의 고유한 특성 패턴"""
        dna_features = []
        
        for idx, row in self.df.iterrows():
            dna = {
                'row_id': idx,
                'completeness_score': (len(row.dropna()) / len(row)) * 100,
                'text_density': sum(len(str(val)) for val in row if pd.notna(val)) / len(row),
                'numeric_ratio': sum(1 for val in row if pd.notna(val) and str(val).replace('.','').replace('-','').isdigit()) / len(row),
                'uniqueness_factor': len(set(str(val).lower() for val in row if pd.notna(val))) / len(row),
                'temporal_hints': self._extract_temporal_patterns(row),
                'data_freshness': self._calculate_data_freshness(row)
            }
            dna_features.append(dna)
        
        return pd.DataFrame(dna_features)
    
    def generate_context_bridges(self):
        """데이터 간 숨겨진 연결고리 발견"""
        bridges = []
        
        for i, row1 in self.df.iterrows():
            for j, row2 in self.df.iterrows():
                if i >= j:
                    continue
                    
                bridge_strength = 0
                connections = []
                
                # 부분 문자열 매칭 (이름의 일부가 다른 곳에 나타나는 경우)
                for col1, val1 in row1.items():
                    if pd.isna(val1):
                        continue
                    for col2, val2 in row2.items():
                        if pd.isna(val2) or col1 == col2:
                            continue
                        
                        val1_words = set(str(val1).lower().split())
                        val2_words = set(str(val2).lower().split())
                        
                        if val1_words & val2_words:  # 공통 단어 존재
                            bridge_strength += len(val1_words & val2_words)
                            connections.append(f"{col1}↔{col2}: {val1_words & val2_words}")
                
                if bridge_strength > 0:
                    bridges.append({
                        'row1': i, 'row2': j,
                        'strength': bridge_strength,
                        'connections': connections
                    })
        
        return sorted(bridges, key=lambda x: x['strength'], reverse=True)[:20]
    
    def create_semantic_neighborhoods(self):
        """의미적으로 유사한 데이터 그룹 생성"""
        neighborhoods = defaultdict(list)
        
        # 각 행에 대해 '의미적 지문' 생성
        for idx, row in self.df.iterrows():
            semantic_fingerprint = []
            
            for col, val in row.items():
                if pd.isna(val):
                    continue
                
                val_str = str(val).lower()
                
                # 패턴 기반 분류
                if re.match(r'\d{4}', val_str):  # 연도 패턴
                    semantic_fingerprint.append('temporal')
                elif any(word in val_str for word in ['university', 'college', 'school']):
                    semantic_fingerprint.append('educational')
                elif any(word in val_str for word in ['forward', 'guard', 'center']):
                    semantic_fingerprint.append('positional')
                elif len(val_str.split()) == 2 and val_str.replace(' ', '').isalpha():
                    semantic_fingerprint.append('name_like')
                elif val_str.replace('.', '').replace('-', '').isdigit():
                    semantic_fingerprint.append('numeric')
                else:
                    semantic_fingerprint.append('descriptive')
            
            # 지문을 기반으로 이웃 그룹 생성
            fingerprint_key = tuple(sorted(set(semantic_fingerprint)))
            neighborhoods[fingerprint_key].append(idx)
        
        return dict(neighborhoods)
    
    def generate_data_story_arcs(self):
        """데이터를 시간순/논리순으로 연결한 '스토리 아크' 생성"""
        story_arcs = []
        
        # 시간 관련 컬럼 찾기
        time_columns = []
        for col in self.df.columns:
            if any(time_word in col.lower() for time_word in ['year', 'date', 'time', 'period']):
                time_columns.append(col)
        
        if time_columns:
            for time_col in time_columns:
                timeline_data = []
                
                for idx, row in self.df.iterrows():
                    time_val = row[time_col]
                    if pd.notna(time_val):
                        # 시간 정보 파싱
                        timeline_entry = {
                            'row_id': idx,
                            'time_value': str(time_val),
                            'time_score': self._parse_time_complexity(str(time_val)),
                            'context': {k: v for k, v in row.items() if k != time_col and pd.notna(v)}
                        }
                        timeline_data.append(timeline_entry)
                
                # 시간순 정렬
                timeline_data.sort(key=lambda x: x['time_score'])
                story_arcs.append({
                    'theme': f'timeline_{time_col}',
                    'sequence': timeline_data
                })
        
        return story_arcs
    
    def create_anomaly_heat_map(self):
        """데이터의 '이상함' 정도를 시각화하는 히트맵 정보"""
        anomaly_scores = {}
        
        for col in self.df.columns:
            col_scores = []
            values = self.df[col].dropna()
            
            if len(values) == 0:
                continue
                
            for val in values:
                anomaly_score = 0
                val_str = str(val)
                
                # 길이 기반 이상함
                lengths = [len(str(v)) for v in values]
                avg_length = np.mean(lengths)
                if len(val_str) > avg_length * 2:
                    anomaly_score += 3
                elif len(val_str) < avg_length * 0.5:
                    anomaly_score += 2
                
                # 패턴 기반 이상함
                if self.df[col].dtype == 'object':
                    # 대부분이 알파벳인데 숫자가 섞인 경우
                    alpha_ratio = sum(c.isalpha() for c in val_str) / len(val_str) if val_str else 0
                    digit_ratio = sum(c.isdigit() for c in val_str) / len(val_str) if val_str else 0
                    
                    if 0.1 < digit_ratio < 0.9 and 0.1 < alpha_ratio < 0.9:
                        anomaly_score += 2
                
                # 빈도 기반 이상함
                value_counts = values.value_counts()
                if value_counts[val] == 1 and len(values) > 5:  # 유일한 값
                    anomaly_score += 1
                
                col_scores.append(anomaly_score)
            
            anomaly_scores[col] = col_scores
        
        return anomaly_scores
    
    def generate_question_hints(self, query_context=""):
        """질문 유형별 힌트 시스템"""
        hints = {
            'direct_lookup': [],  # 직접 찾기 힌트
            'aggregation': [],    # 집계 힌트
            'comparison': [],     # 비교 힌트
            'temporal': [],       # 시간 관련 힌트
            'relational': []      # 관계 힌트
        }
        
        # 컬럼별 특성 분석
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            
            # 직접 찾기 힌트
            if unique_count == total_count:  # 모든 값이 고유함
                hints['direct_lookup'].append(f"{col}은 고유 식별자로 사용 가능")
            
            # 집계 힌트
            if self.df[col].dtype in ['int64', 'float64']:
                hints['aggregation'].append(f"{col}은 수치 집계 가능 (평균: {self.df[col].mean():.2f})")
            
            # 비교 힌트
            if 2 <= unique_count <= 10:  # 카테고리형
                top_values = self.df[col].value_counts().head(3)
                hints['comparison'].append(f"{col} 주요 값들: {dict(top_values)}")
            
            # 시간 힌트
            if any(time_word in col.lower() for time_word in ['year', 'date', 'time']):
                hints['temporal'].append(f"{col}에서 시간 범위 분석 가능")
        
        return hints
    
    def create_data_confidence_map(self):
        """각 데이터 포인트의 '신뢰도' 계산"""
        confidence_map = {}
        
        for col in self.df.columns:
            col_confidence = []
            
            for idx, val in enumerate(self.df[col]):
                confidence = 100  # 기본 신뢰도
                
                if pd.isna(val):
                    confidence = 0
                else:
                    val_str = str(val)
                    
                    # 일관성 체크
                    similar_values = self.df[col].apply(lambda x: 
                        len(set(str(x).lower().split()) & set(val_str.lower().split())) > 0 
                        if pd.notna(x) else False).sum()
                    
                    if similar_values > 1:
                        confidence += 10  # 유사한 값들이 있으면 신뢰도 증가
                    
                    # 포맷 일관성
                    if col in self.df.select_dtypes(include=['object']).columns:
                        # 다른 값들과 포맷이 비슷한지 체크
                        formats = [self._get_format_pattern(str(v)) for v in self.df[col].dropna()]
                        current_format = self._get_format_pattern(val_str)
                        
                        format_consistency = formats.count(current_format) / len(formats)
                        confidence += format_consistency * 20
                
                col_confidence.append(min(confidence, 100))
            
            confidence_map[col] = col_confidence
        
        return confidence_map
    
    def generate_enhanced_prompt_context(self, original_question=""):
        """LLM을 위한 종합적인 컨텍스트 생성"""
        context = {
            'table_metadata': {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            'data_dna': self.create_data_DNA().to_dict('records'),
            'context_bridges': self.generate_context_bridges()[:5],  # 상위 5개만
            'semantic_neighborhoods': self.create_semantic_neighborhoods(),
            'story_arcs': self.generate_data_story_arcs(),
            'anomaly_heat_map': self.create_anomaly_heat_map(),
            'question_hints': self.generate_question_hints(original_question),
            'confidence_map': self.create_data_confidence_map(),
            'raw_table': self.df.to_dict('records')
        }
        
        return context
    
    def create_llm_optimized_prompt(self, question, context):
        """LLM을 위한 최적화된 프롬프트 생성"""
        prompt = f"""
# 테이블 분석 작업

## 질문
{question}

## 테이블 기본 정보
- 크기: {context['table_metadata']['shape'][0]}행 x {context['table_metadata']['shape'][1]}열
- 컬럼: {', '.join(context['table_metadata']['columns'])}

## 데이터 품질 인사이트
"""
        
        # 높은 신뢰도를 가진 데이터 포인트 강조
        high_confidence_data = []
        for col, confidences in context['confidence_map'].items():
            avg_confidence = np.mean(confidences)
            if avg_confidence > 80:
                high_confidence_data.append(f"{col} (신뢰도: {avg_confidence:.1f}%)")
        
        if high_confidence_data:
            prompt += f"### 고신뢰도 컬럼들: {', '.join(high_confidence_data)}\n\n"
        
        # 데이터 연결고리 정보
        if context['context_bridges']:
            prompt += "### 데이터 간 연결 패턴:\n"
            for bridge in context['context_bridges'][:3]:
                prompt += f"- 행 {bridge['row1']}과 {bridge['row2']} 연결강도: {bridge['strength']}\n"
            prompt += "\n"
        
        # 질문 유형별 힌트
        if context['question_hints']:
            prompt += "### 분석 힌트:\n"
            for hint_type, hints in context['question_hints'].items():
                if hints:
                    prompt += f"**{hint_type}**: {hints[0]}\n"
            prompt += "\n"
        
        # 실제 테이블 데이터
        prompt += "## 테이블 데이터\n"
        prompt += self.df.to_string(index=False)
        
        prompt += f"""

## 지시사항
1. 위의 품질 인사이트와 연결 패턴을 참고하여 답변하세요
2. 고신뢰도 데이터를 우선적으로 활용하세요  
3. 데이터 간 연결고리를 고려하여 포괄적으로 분석하세요
4. 답변의 근거를 명확히 제시하세요
"""
        
        return prompt
    
    # 헬퍼 메서드들
    def _extract_temporal_patterns(self, row):
        temporal_count = 0
        for val in row:
            if pd.notna(val) and re.search(r'\d{4}', str(val)):
                temporal_count += 1
        return temporal_count
    
    def _calculate_data_freshness(self, row):
        current_year = datetime.now().year
        freshness_score = 0
        
        for val in row:
            if pd.notna(val):
                years = re.findall(r'\d{4}', str(val))
                for year in years:
                    year_int = int(year)
                    if 1900 <= year_int <= current_year:
                        freshness_score += max(0, 100 - (current_year - year_int))
        
        return freshness_score / len(row) if len(row) > 0 else 0
    
    def _parse_time_complexity(self, time_str):
        if 'present' in time_str.lower():
            return 9999  # 현재까지 계속되는 것은 가장 최신
        
        years = re.findall(r'\d{4}', time_str)
        if years:
            return int(years[0])  # 첫 번째 연도 반환
        
        return 0
    
    def _get_format_pattern(self, val_str):
        pattern = ""
        for char in val_str:
            if char.isalpha():
                pattern += "A"
            elif char.isdigit():
                pattern += "9"
            else:
                pattern += char
        return pattern

# 사용 예시
def demo_enhanced_system():
    # 샘플 데이터
    data = {
        'Player': ['Quincy Acy', 'Hassan Adams', 'Alexis Ajinca', 'Solomon Alabi'],
        'No_': [4, 3, 42, 50],
        'Nationality': ['United States', 'United States', 'France', 'Nigeria'],
        'Position_': ['Forward', 'Guard-Forward', 'Center', 'Center'],
        'Years_in_Toronto': ['2012-present', '2008-09', '2011', '2010-2012'],
        'School_Club_Team': ['Baylor', 'Arizona', 'Hyères-Toulon ( France )', 'Florida State']
    }
    
    df = pd.DataFrame(data)
    enhancer = TableEnhancer(df)
    
    question = "From which school did the player, who has been in Toronto since 2012, come from?"
    
    # 향상된 컨텍스트 생성
    enhanced_context = enhancer.generate_enhanced_prompt_context(question)
    
    # LLM 최적화 프롬프트 생성
    optimized_prompt = enhancer.create_llm_optimized_prompt(question, enhanced_context)
    
    print("=== LLM용 최적화된 프롬프트 ===")
    print(optimized_prompt)
    
    print("\n=== 생성된 인사이트 요약 ===")
    print(f"데이터 DNA 특성: {len(enhanced_context['data_dna'])}개 행 분석")
    print(f"발견된 연결고리: {len(enhanced_context['context_bridges'])}개")
    print(f"의미적 그룹: {len(enhanced_context['semantic_neighborhoods'])}개")
    print(f"스토리 아크: {len(enhanced_context['story_arcs'])}개")

if __name__ == "__main__":
    demo_enhanced_system()