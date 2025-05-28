import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# TabFact jsonl 파일 로딩 함수
def load_tabfact_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Table 데이터를 문자열로 직렬화 (단순히 셀들 연결)
def serialize_table(table):
    lines = []
    for row in table:
        lines.append(" | ".join(row))
    return "\n".join(lines)

# TabFact 항목을 query-answer 포맷으로 변환
def process_tabfact(data):
    processed = []
    for item in data:
        table = serialize_table(item["table"])
        statement = item["statement"]
        label = item["label"]
        query = f"Table:\n{table}\n\nStatement:\n{statement}\nIs this statement entailed by the table?"
        answer = "entailed" if label == 1 else "refuted"
        processed.append({"query": query, "answer": answer})
    return processed

# 실제 실행
tabfact_path = "tabfact_train.jsonl"  # 직접 다운로드한 경로로 수정
raw_tabfact = load_tabfact_jsonl(tabfact_path)
processed = process_tabfact(raw_tabfact)
train_data, val_data = train_test_split(processed, test_size=0.1, random_state=42)

tabfact_dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data)
})