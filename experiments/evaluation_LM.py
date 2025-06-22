'''
같은 전처리 - TCRF를 활용하여 table을 filtering 후, 다양한 모델로 QA를 진행하여 TCRF의 성능을 실험
1. TAPAS
2. TAPEX
3. OmniTab
4. UnifiedSKG
5. LLM-based prompting - GPT-4o-mini
'''

from transformers import (
    TapasTokenizer, TapasForQuestionAnswering,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from TCRF_main import make_response
import pandas as pd
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load all model/tokenizer objects globally, only once
tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-mini-finetuned-wtq")
tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-mini-finetuned-wtq").to(device)

import os
import openai
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def LLM_based_answer(filtered_df, question):
    def linearize_row_wise(df):
        return " | ".join(
            ["; ".join([f"{col}: {val}" for col, val in row.items()]) for _, row in df.iterrows()]
        )
    
    def extract_json_list(text: str) -> list:
        """
        텍스트에서 JSON 리스트를 추출하는 함수
        """
        # JSON 리스트 패턴 찾기
        json_pattern = r'\[.*?\]'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # JSON 파싱 시도
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # JSON을 찾지 못한 경우, 다른 패턴으로 시도
        answer_pattern = r'(?:Answer|answer):\s*(\[.*?\])'
        match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 마지막 줄에서 리스트 찾기
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if '[' in line and ']' in line:
                try:
                    # 줄에서 JSON 리스트 부분만 추출
                    start = line.find('[')
                    end = line.rfind(']') + 1
                    potential_json = line[start:end]
                    return json.loads(potential_json)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # 모든 시도가 실패한 경우
        return ["Answer not found in the table."]

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)  # 더 낮은 temperature로 일관성 향상

    Responder_PROMPT = """
    You are a highly capable data analyst working with tabular data.

    Your task is to read the following table and answer the question as accurately as possible, using only the information in the table.

    ### Instructions
    1. **Read the table carefully** - Each row is separated by "|" and each field within a row is separated by ";"
    2. **Identify all conditions** in the question that need to be satisfied simultaneously
    3. **For each row**, check if **ALL conditions are strictly matched (==)** using exact string comparison
    4. **Extract the answer** - This should be one of the column values, such as a name, number, or label.
    5. **Only use the table information provided**, do not guess or infer from external knowledge.
    6. **Format your response** as a **JSON list**
    7. **If no rows satisfy all conditions**, return: ["None"]

    ---

    ### Step-by-step reasoning:

    1. **Identify what is being asked for** → what column name should be returned? (e.g., Band, Area_served, Purpose)
    2. **List all conditions that must be matched** → extract conditions as exact comparisons, e.g., Callsign == 4BCB
    3. **Go through each row**:
        - For each row, check if **ALL conditions match exactly**
        - If yes, collect the value from the **target column**
    4. **Prepare final answer** → collect the answers into a JSON list format

    ---

    ### Table
    {table_text}

    ### Question
    {question}

    ---

    ### Analysis
    Let me work through this step by step:

    1. **Target column to extract** (must be one of the table column names):
    2. **Conditions to match** (exact string comparison only):
    3. **Row-by-row check**:
    4. **Extracted answers**:

    ### Final Answer (as JSON list only)
    """
    table_text = linearize_row_wise(filtered_df)
    
    response = llm.invoke(
        Responder_PROMPT.format(
            question=question,
            table_text=table_text
        )
    )
    
    raw_answer = response.content.strip()
    parsed_answer = extract_json_list(raw_answer)

    return parsed_answer


def tapas_answer(filtered_df, question, tokenizer, model) -> list:

    filtered_df = pd.DataFrame(filtered_df)
    inputs = tokenizer(table=filtered_df, queries=[question], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach().cpu(),
        outputs.logits_aggregation.detach().cpu()
    )

    if not predicted_answer_coordinates[0]:
        return ["None"]
    return [filtered_df.iat[row, col] for row, col in predicted_answer_coordinates[0]]


def tapex_base_answer(filtered_df, question, tokenizer, model) -> list:
    """TAPEX Base 구현 (공식 모델)"""
    
    # 테이블을 TAPEX 형식으로 변환
    table = pd.DataFrame(filtered_df)
    
    encoding = tokenizer(table, question, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    outputs = model.generate(
        **encoding,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    
    predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return [predicted_answer] if predicted_answer else ["None"]


def omnitab_answer(filtered_df, question, tokenizer, model) -> list:
    # Ensure input is a DataFrame
    if not isinstance(filtered_df, pd.DataFrame):
        filtered_df = pd.DataFrame(filtered_df)

    # OmniTab 입력 형식
    table_text = filtered_df.to_csv(index=False, sep='\t')
    input_text = f"Question: {question} Table: {table_text}"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [answer] if answer else ["None"]


def unifiedskg_answer(filtered_df, question, tokenizer, model) -> list:
    # UnifiedSKG 형식으로 테이블 구조화
    table_linearized = ""
    headers = filtered_df.columns.tolist()
    table_linearized += " | ".join(headers) + " "
    
    for _, row in filtered_df.iterrows():
        row_text = " | ".join([str(val) for val in row.values])
        table_linearized += row_text + " "
    
    input_text = f"Question: {question} Table: {table_linearized}"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [answer] if answer else ["None"]


def run_all_models(index) -> dict:
    """모든 모델 실행 및 결과 비교 - make_response 한 번만 실행"""

    jsonl_path = "outputs/tcrf_results.jsonl"
    target_index = index
    found = False
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # index가 문자열이나 정수로 저장되어 있을 수 있으므로, 비교 시 모두 문자열로 변환
            if float(entry.get("index")) == float(target_index):
                answer = entry['answer']
                question = entry['question']
                filtered_df = pd.DataFrame(entry['filtered_df'])
    results = {}
    print(f"Question: {question}")
    print(f"Real Answer: {answer}")
    print(f"Filtered table shape: {filtered_df.shape}")
    
    models = {
        'LLM-based': LLM_based_answer
    }
    
    for model_name, model_func in models.items():
        try:
            print(f"Running {model_name}...")
            # 동일한 데이터를 모든 모델에 전달
            result = model_func(filtered_df, question)
            results[model_name] = result
            print(f"{model_name} result: {result}")
        except Exception as e:
            print(f"{model_name} failed: {e}")
            results[model_name] = ["Error"]
    
    return results, answer


def compute_em_f1(predicted_answers, true_answers):
    """EM, F1 점수 계산"""
    def normalize(ans):
        return str(ans).lower().strip()

    if isinstance(true_answers, list):
        gold_set = set(normalize(a) for a in true_answers)
    else:
        gold_set = {normalize(true_answers)}

    if isinstance(predicted_answers, list):
        pred_set = set(normalize(a) for a in predicted_answers)
    else:
        pred_set = {normalize(predicted_answers)}

    # EM
    em = int(gold_set == pred_set)

    # F1
    common = gold_set & pred_set
    if len(common) == 0:
        f1 = 0.0
    else:
        precision = len(common) / len(pred_set)
        recall = len(common) / len(gold_set)
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return em, f1


def batch_evaluation(indices_list):
    """배치 평가 함수"""
    all_results = {}
    accuracy_results = {}
    
    for idx in indices_list:
        print(f"\n{'='*50}")
        print(f"Processing Index: {idx}")
        print(f"{'='*50}")
        
        try:
            results, true_answer = run_all_models(idx)
            all_results[idx] = {
                'results': results,
                'true_answer': true_answer
            }
            
            # EM, F1 계산
            accuracy_results[idx] = {}
            for model_name, predicted in results.items():
                if predicted != ["Error"]:
                    em, f1 = compute_em_f1(predicted, true_answer)
                    accuracy_results[idx][model_name] = {'EM': em, 'F1': f1}
                    print(f"{model_name} EM: {em}, F1: {f1:.2f}")
                else:
                    accuracy_results[idx][model_name] = {'EM': 0, 'F1': 0.0}
                    print(f"{model_name} EM: 0, F1: 0.00 (Error)")
                    
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            all_results[idx] = {'error': str(e)}
    
    return all_results, accuracy_results


def calculate_overall_accuracy(accuracy_results):
    """전체 EM, F1 평균 계산"""
    model_scores = {}

    all_models = set()
    for idx_results in accuracy_results.values():
        all_models.update(idx_results.keys())

    for model in all_models:
        em_total = 0
        f1_total = 0
        count = 0
        for idx_results in accuracy_results.values():
            if model in idx_results:
                em_total += idx_results[model]['EM']
                f1_total += idx_results[model]['F1']
                count += 1
        if count > 0:
            model_scores[model] = {
                'EM': em_total / count,
                'F1': f1_total / count
            }
        else:
            model_scores[model] = {'EM': 0.0, 'F1': 0.0}
    return model_scores


if __name__ == "__main__":
    import os

    SAVE_PATH = "outputs/eval_scores.csv"

    # 기존 결과 불러오기
    if os.path.exists(SAVE_PATH):
        df_results = pd.read_csv(SAVE_PATH)
        done_indices = set(df_results["index"].unique())
    else:
        df_results = pd.DataFrame(columns=["index", "model", "EM", "F1"])
        done_indices = set()

    with open("outputs/tcrf_results.jsonl", "r") as f:
        lines = f.readlines()
        index_list = list(range(len(lines)))

    pending_indices = [idx for idx in index_list if idx not in done_indices]

    for idx in pending_indices:
        print(f"\n{'='*50}")
        print(f"Processing Index: {idx}")
        print(f"{'='*50}")
        try:
            results, true_answer = run_all_models(idx)
            for model_name, predicted in results.items():
                if predicted != ["Error"]:
                    em, f1 = compute_em_f1(predicted, true_answer)
                else:
                    em, f1 = 0, 0.0

                df_results = pd.concat([
                    df_results,
                    pd.DataFrame([{
                        "index": idx,
                        "model": model_name,
                        "EM": em,
                        "F1": f1
                    }])
                ], ignore_index=True)

            # ✅ 자동 저장
            df_results.to_csv(SAVE_PATH, index=False)
        except Exception as e:
            print(f"[Error] Failed at index {idx}: {e}")

    print(f"\n✅ 전체 완료! 결과는 {SAVE_PATH}에 저장됨")