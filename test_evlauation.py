from QA_graph.graph_workflow import build_workflow_graph
import json
from sklearn.metrics import accuracy_score, f1_score
import re


def normalize(text):
    """간단한 정규화 (공백, 소문자 등)"""
    return re.sub(r"[\W_]+", " ", text.lower()).strip()

def compute_em(pred, label):
    return int(normalize(pred) == normalize(label))

def compute_f1(pred, label):
    pred_tokens = set(normalize(pred).split())
    label_tokens = set(normalize(label).split())
    if not pred_tokens or not label_tokens:
        return 0.0
    common = pred_tokens & label_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


# 기존의 make_response 함수
def make_response(index, json_file="data/test.json", db_file="data/table.db"):
    workflow = build_workflow_graph()

    result = workflow.invoke({
        "raw_db_file_path": db_file,
        "json_df_file_path": json_file,
        "index": index
    })

    return result.get("LLM_answer", "⚠️ LLM_answer 존재하지 않습니다.")

# 메인 평가 함수
def evaluate_open_wikitable(json_path="data/test.json", db_path="data/table.db", max_samples=100):
    with open(json_path, "r") as f:
        data = json.load(f)

    keys = list(data["question"].keys())
    if max_samples:
        keys = keys[:max_samples]

    em_list = []
    f1_list = []

    for idx in keys:
        gold = data["answer"][idx]
        gold = gold[0] if isinstance(gold, list) else gold

        print(f"\n🔍 Evaluating index: {idx} — Question: {data['question'][idx]}")
        pred = make_response(idx, json_file=json_path, db_file=db_path)

        # 리스트 응답 대응: ["Baylor"] → "Baylor"
        if isinstance(pred, list):
            pred = pred[0] if pred else ""

        print(f"💡 Pred: {pred} | ✅ Gold: {gold}")

        em = compute_em(pred, gold)
        f1 = compute_f1(pred, gold)
        em_list.append(em)
        f1_list.append(f1)

    print("\n📊 전체 평가 결과")
    print(f"EM (Exact Match): {sum(em_list) / len(em_list):.4f}")
    print(f"F1 Score        : {sum(f1_list) / len(f1_list):.4f}")


if __name__ == "__main__":
    evaluate_open_wikitable(
        json_path="data/test.json",
        db_path="data/table.db",
        max_samples=10  # 필요시 전체 평가
    )