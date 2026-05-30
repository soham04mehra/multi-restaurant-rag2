# test_chatbot.py
# This script checks how well our AI is answering restaurant questions.

import os
import time
import asyncio
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
# pyright: ignore [reportMissingImports]
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
# 1. Import your chatbot logic
import chatbot
from chatbot import get_answer, embedding_model

# 2. Load your secret API keys
load_dotenv()

# ==========================================
# STEP 1: SETUP THE EVALUATION MODELS
# ==========================================
# Using Llama 3.3 70B with Gemini 2.5 Flash as fallback for Evaluation
primary_eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
fallback_eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.0)
eval_llm = primary_eval_llm.with_fallbacks([fallback_eval_llm])

ragas_llm = LangchainLLMWrapper(eval_llm)

# Wrap the EXACT same BGE (BAAI/bge-small-en-v1.5) model that the chatbot uses.
# This ensures evaluation embeddings are in the same vector space as production,
# which is required for answer_relevancy scores to be non-zero.
class BGEWrapper(BaseRagasEmbeddings):
    """Thin wrapper so RAGAS can call our production FastEmbed BGE model."""
    def embed_query(self, text: str):
        return embedding_model.embed_query(text)

    def embed_documents(self, texts):
        return embedding_model.embed_documents(texts)

    async def aembed_query(self, text: str):
        # FastEmbed has no async interface; run synchronously
        return embedding_model.embed_query(text)

    async def aembed_documents(self, texts):
        return embedding_model.embed_documents(texts)

ragas_embeddings = BGEWrapper()

# ==========================================
# STEP 2: DEFINE OUR TEST DATA
# ==========================================
test_cases = [
    {"question": "suggest me a spicy roll", "ground_truth": "Spicy Chicken Doner or Harissa Falafel Doner"},
    {"question": "suggest me a chicken shawarma", "ground_truth": "Chicken Shawarma wrapped in soft khubz bread priced at 329 rupees"},
    {"question": "show me veg options", "ground_truth": "Crispy Falafel Doner, Harissa Falafel Doner, Paneer Doner, Paneer Shawarma, French Fries, Walnut Brownie"},
    {"question": "I am allergic to gluten, what can I eat", "ground_truth": "French Fries, Chicken Popcorn, Walnut Brownie"},
    {"question": "what is the cheapest dish", "ground_truth": "French Fries"}
]

# ==========================================
# STEP 3: GET ANSWERS FROM THE CHATBOT
# ==========================================
async def main():
    print("--- Starting Chatbot Tests ---")
    results_list = []

    for case in test_cases:
        print(f"Testing: {case['question']}")

        # Retry up to 3 times in case of Gemini rate limit (429)
        response = None
        for attempt in range(3):
            response = await get_answer(case['question'], session_id="eval_user", restaurant_id="rest_delhi_01")
            # If we got a real answer (not the fallback error message), break
            if response["dishes"] or "trouble" not in response["answer"]:
                break
            print(f"   - Attempt {attempt+1} failed (rate limit?), waiting 35s...")
            await asyncio.sleep(35)

        # Robust Context Extraction
        # We use 'content' if available, otherwise fallback to name/description
        contexts = []
        for dish in response.get("dishes", []):
            c = dish.get("content", "").strip()
            if not c:
                c = f"Dish: {dish.get('name')}. Description: {dish.get('description')}. Price: {dish.get('price')}"
            contexts.append(c)

        results_list.append({
            "question": case['question'],
            "answer": response["answer"],
            "contexts": contexts,
            "ground_truth": case["ground_truth"]
        })
        print(f"   ✅ Found {len(contexts)} dishes. Answer: {response['answer'][:80]}...")

        # Wait 35s between each question so Gemini free-tier quota (2 req/min) resets
        print("   Waiting 35s for rate limit reset...")
        await asyncio.sleep(35)
        
    return results_list

results_list = asyncio.run(main())

# ==========================================
# STEP 4: SCORE THE ANSWERS (EVALUATION)
# ==========================================
dataset = Dataset.from_list(results_list)
print("\n--- Scoring Answers (This takes a moment) ---")
print("Waiting 60s before scoring to let Gemini rate limit reset...")
time.sleep(60)

evaluation_results = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    raise_exceptions=False
)

# ==========================================
# STEP 5: FINAL PASS / FAIL SUMMARY
# ==========================================
report_df = evaluation_results.to_pandas()
report_df['question'] = [r['question'] for r in results_list]
report_df = report_df.fillna(0)

print("\n--- SCORE TABLE ---")
print(report_df[['question', 'faithfulness', 'answer_relevancy', 'context_recall']])

print("\n--- FINAL PASS / FAIL SUMMARY ---")

# 1. Check Faithfulness
avg_faith = report_df["faithfulness"].mean()
if avg_faith >= 0.7:
    print(f"FAITHFULNESS: {avg_faith:.2f} -> ✅ PASS")
else:
    print(f"FAITHFULNESS: {avg_faith:.2f} -> ❌ FAIL (Target: 0.7)")

# 2. Check Answer Relevancy
avg_relevancy = report_df["answer_relevancy"].mean()
if avg_relevancy >= 0.7:
    print(f"RELEVANCY: {avg_relevancy:.2f} -> ✅ PASS")
else:
    print(f"RELEVANCY: {avg_relevancy:.2f} -> ❌ FAIL (Target: 0.7)")

# 3. Check Context Recall
avg_recall = report_df["context_recall"].mean()
if avg_recall >= 0.5:
    print(f"RECALL: {avg_recall:.2f} -> ✅ PASS")
else:
    print(f"RECALL: {avg_recall:.2f} -> ❌ FAIL (Target: 0.5)")

print("\nEvaluation Complete!")
