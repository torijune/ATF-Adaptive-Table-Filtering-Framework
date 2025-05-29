from typing import Dict
import os
import openai

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def extract_essential_columns_fn(state: Dict) -> Dict:
    import ast
    question = state["question"]
    raw_table = state["raw_table"]
    all_columns = raw_table.columns

    prompt = f"""
    You are a table question answering expert.

    Your task is to identify the essential table columns required to answer a given question, from the list of available columns.

    Please strictly follow these instructions:
    - Output only a Python list of column names (e.g., ["ColumnA", "ColumnB"])
    - Do not include any explanation or extra text.
    - If you're unsure, choose conservatively by selecting columns that appear most relevant to keywords in the question.
    - Use exact column names as they appear in the list.

    Question: {question}

    Available Columns: {', '.join(all_columns)}

    Essential Columns:
    """

    response = llm.invoke(prompt)
    extracted = response.content.strip()

    try:
        # Safer parsing than eval
        essential_cols = ast.literal_eval(extracted)
        if not isinstance(essential_cols, list) or not all(isinstance(col, str) for col in essential_cols):
            raise ValueError("Parsed content is not a valid list of strings.")
        # Ensure all selected columns exist
        essential_cols = [col for col in essential_cols if col in all_columns]
    except Exception as e:
        print(f"[Essential Column Extracter] ⚠️ Failed to parse response: {extracted} — {e}")
        essential_cols = list(all_columns)

    state["essential_columns"] = essential_cols
    print(f"\n[Essential Column Extracter] ✅ Essential Column List: {essential_cols}")
    return state

extract_essential_columns_node = RunnableLambda(extract_essential_columns_fn)