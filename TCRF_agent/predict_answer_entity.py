import os
import openai
import json
import re

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

Enetity_PROMPT = """
You are a Question Answering expert.

First, carefully read the question and **rephrase** it in a clearer and more specific way that makes the target of the answer obvious.

Then, based on the rephrased question, determine what type of entity the answer is asking for. This might be:
- a person (e.g., someone's name),
- an organization (e.g., a team, company, etc.),
- a date (e.g., a specific day),
- a number (e.g., age, count, price),
- a location (e.g., country, city, place),
- or some other type.

You may include new types if needed. Then, assign a **confidence score** (0.0 to 1.0) to each type, depending on how likely it is to be the expected answer.

Respond in **exactly** this JSON format:
{{
  "EntityType1": score,
  "EntityType2": score
}}

Question: {question}
"""

def predict_answer_entity_fn(state):
    question = state["question"]
    prompt = Enetity_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    try:
        scores = json.loads(response.content)
        # print(f"[PredictAnswerEntity] ✅ LLM Predicted Answer Enetity Type Candidates List: {scores}")
        if isinstance(scores, dict):
            predicted_type = max(scores.items(), key=lambda x: x[1])[0]
        else:
            predicted_type = "Other"
        state["predicted_answer_entity_scores"] = scores
    except Exception:
        predicted_type = "Other"

    # print(f"[PredictAnswerEntity] ✅ LLM Predicted Answer Enetity Type: {predicted_type}")
    # No fixed set of valid types — use top scoring type directly

    state["predicted_answer_entity"] = predicted_type
    return state

predict_answer_entity_node = RunnableLambda(predict_answer_entity_fn)