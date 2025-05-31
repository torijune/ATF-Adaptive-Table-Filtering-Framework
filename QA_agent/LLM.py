import os
import openai
import json
import re

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # 더 낮은 temperature로 일관성 향상

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
7. **If no rows satisfy all conditions**, return: ["Answer not found in the table."]

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

'''
Final LLM Responder Robustness 업그레이드 전략
1. LLM에게 이전의 전략처럼 question의 조건부의 AND를 잘 잡아내고, 주어진 rows를 순차적으로 비교하면서 추론하도록 유도하는 prompting 필요
2. 복수 응답인데 단수 응답을 하거나 반대로 단수 응답인데 복수 응답을 하는 등의 실수를 보완해야함
'''
def responder_fn(state: dict) -> dict:
    question = state.get("question", "")
    answer = state["answer"]
    final_table_text = state.get("final_table_text", "")

    print(f"[LLM Responder] ❓ Question:\n{question}")
    print(f"[LLM Responder] 📋 Table:\n{final_table_text}")

    # Column descriptions 추가 (있는 경우)
    selected_column_descriptions = state.get("column_description", {})
    column_desc_text = ""
    if selected_column_descriptions:
        column_desc_text = "\n\n### Column Descriptions\n" + "\n".join(
            f"- {col}: {desc}" for col, desc in selected_column_descriptions.items()
        )

    # 프롬프트 구성
    full_prompt = Responder_PROMPT + column_desc_text
    
    try:
        response = llm.invoke(
            full_prompt.format(
                question=question,
                table_text=final_table_text
            )
        )
        
        raw_answer = response.content.strip()
        print(f"[LLM Responder] 🤖 Raw Response:\n{raw_answer}")
        
        # JSON 리스트 추출 시도
        parsed_answer = extract_json_list(raw_answer)

        # Try to extract predicted entity type if specified in response
        predicted_entity_type = None
        entity_type_match = re.search(r'"?predicted_answer_entity"?\s*[:=]\s*"?(.*?)"?[\n,}]', raw_answer, re.IGNORECASE)
        if entity_type_match:
            predicted_entity_type = entity_type_match.group(1).strip()

        print(f"[LLM Responder] 🧠 Predicted Entity Type: {predicted_entity_type}")
        
        print(f"[LLM Responder] ✅ Parsed Answer: {parsed_answer}")
        print(f"\n [LLM Responder] ✅ Real Answer: {answer}")
        
        return {**state, "LLM_answer": parsed_answer, "predicted_answer_entity": predicted_entity_type}
        
    except Exception as e:
        print(f"[LLM Responder] ❌ Error: {e}")
        return {**state, "LLM_answer": ["Answer not found in the table."], "predicted_answer_entity": None}

# 노드 생성
responder_node = RunnableLambda(responder_fn)