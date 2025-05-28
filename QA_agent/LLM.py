import os
import openai

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

Responder_PROMPT = """
You are a highly capable data analyst working with tabular data.

Your task is to read the following table and answer the question as accurately as possible, using only the information in the table.

### Instructions
- Base your answer strictly on the contents of the table.
- Do not use any external knowledge.
- If the answer is not found in the table, respond with: ["Answer not found in the table."]
- Return the answer as a list containing only the main answer keyword(s), e.g., ["ANSWER"]
- Do not include any explanation, just return the list.

Let's think step by step

### Table
{table_text}

### Question
{question}

### Answer
"""

def responder_fn(state: dict) -> dict:
    question = state.get("question", "")
    final_table_text = state.get("final_table_text", "")

    print(f"[LLM Responder] ❓ Question:\n{question}")
    print(f"[LLM Responder] 📋 Table:\n{final_table_text}")

    response = llm.invoke(
        Responder_PROMPT.format(
            question=question,
            table_text=final_table_text
        )
    )

    return {**state, "LLM_answer": response.content}

responder_node = RunnableLambda(responder_fn)