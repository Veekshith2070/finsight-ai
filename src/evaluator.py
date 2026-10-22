import json
import time
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY
from src.rag import answer_financial_query
from src.logger import logger

# 1. Define the RAG Triad Scorecard Schema
class RAGTriadScore(BaseModel):
    faithfulness: float = Field(
        ..., ge=0.0, le=1.0, 
        description="1.0 if all claims in the generated answer are strictly supported by the context; 0.0 if hallucinated."
    )
    faithfulness_reasoning: str
    answer_relevance: float = Field(
        ..., ge=0.0, le=1.0, 
        description="1.0 if the answer directly and completely answers the user question; 0.0 if evasive or irrelevant."
    )
    answer_relevance_reasoning: str
    context_precision: float = Field(
        ..., ge=0.0, le=1.0, 
        description="1.0 if the retrieved context contains the necessary facts to answer the question; 0.0 if irrelevant noise."
    )
    context_precision_reasoning: str

# 2. Golden Financial Benchmark Test Suite
GOLDEN_BENCHMARK = [
    {
        "id": "TC-01",
        "category": "Factual Line Item",
        "question": "What were Apple's Products revenue and Services revenue for the three months ended June 27, 2026?",
        "expected_facts": "Products: $78,678 million, Services: $30,739 million on Page 1."
    },
    {
        "id": "TC-02",
        "category": "Cost of Sales",
        "question": "What was Apple's total Cost of sales for the three months ended June 27, 2026?",
        "expected_facts": "Total Cost of sales: $58,844 million (Products: $47,153M, Services: $11,691M)."
    },
    {
        "id": "TC-03",
        "category": "Multi-Period Comparison",
        "question": "What were Apple's total Net sales for the nine months ended June 27, 2026 compared to June 28, 2025?",
        "expected_facts": "$364,357 million for 2026 vs $313,695 million for 2025."
    },
    {
        "id": "TC-04",
        "category": "Anti-Hallucination Guardrail",
        "question": "What was Microsoft's Azure cloud revenue in 2026?",
        "expected_facts": "Must refuse: 'The provided document does not contain information to answer this question.'"
    },
    {
        "id": "TC-05",
        "category": "Unstated Derivative Metric",
        "question": "What were the profits of Products in Apple for the past three months?",
        "expected_facts": "Must state profits of Products is not a direct line item, citing Net sales ($78,678M) and Cost of sales ($47,153M)."
    }
]

def evaluate_rag_sample(question: str, context: str, answer: str, expected_facts: str) -> RAGTriadScore:
    """
    Uses an LLM Judge with structured outputs to score Faithfulness,
    Answer Relevance, and Context Precision according to the RAG Triad.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    judge_system_prompt = (
        "You are an impartial, highly rigorous AI Benchmark Judge evaluating a Retrieval-Augmented Generation (RAG) system.\n"
        "Evaluate the candidate based STRICTLY on the RAG Triad:\n"
        "1. FAITHFULNESS: Can every single claim in the generated answer be directly inferred from the retrieved context? If there is any outside information or hallucination, score 0.0 - 0.5.\n"
        "2. ANSWER RELEVANCE: Does the generated answer directly address the user's question without unnecessary tangents?\n"
        "3. CONTEXT PRECISION: Does the retrieved context contain the information required to answer the question, or is it irrelevant noise?\n"
        "Return structured JSON matching the requested schema."
    )

    eval_prompt = (
        f"USER QUESTION: {question}\n\n"
        f"EXPECTED FACTS / GROUND TRUTH: {expected_facts}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}\n"
    )

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=eval_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=judge_system_prompt,
                    response_mime_type="application/json",
                    response_schema=RAGTriadScore,
                    temperature=0.0
                )
            )
            return RAGTriadScore.model_validate_json(response.text)
        except Exception as e:
            if attempt == 3:
                raise e
            wait_s = (attempt + 1) * 7
            logger.warning(f"Judge API temporary error ({e}). Retrying in {wait_s}s...")
            time.sleep(wait_s)

def run_rag_benchmark():
    """
    Runs the entire golden benchmark dataset through the RAG pipeline,
    evaluates each sample, prints a scorecard, and writes the JSON report.
    """
    print("\n==================================================================")
    print("      FinSight AI - Scientific RAG Triad Benchmark Suite         ")
    print("==================================================================\n")

    results = []
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_precision = 0.0

    for idx, test_case in enumerate(GOLDEN_BENCHMARK, 1):
        print(f"[{idx}/{len(GOLDEN_BENCHMARK)}] Evaluating: {test_case['id']} - {test_case['category']}...")
        print(f"      Q: '{test_case['question']}'")

        # 1. Run through our live RAG system
        rag_output = answer_financial_query(test_case["question"], top_k=2)
        generated_answer = rag_output["answer"]
        retrieved_context = rag_output.get("context", "")

        # 2. Judge evaluation
        score = evaluate_rag_sample(
            question=test_case["question"],
            context=retrieved_context,
            answer=generated_answer,
            expected_facts=test_case["expected_facts"]
        )

        total_faithfulness += score.faithfulness
        total_relevance += score.answer_relevance
        total_precision += score.context_precision

        results.append({
            "test_case_id": test_case["id"],
            "category": test_case["category"],
            "question": test_case["question"],
            "answer": generated_answer,
            "faithfulness": score.faithfulness,
            "faithfulness_reasoning": score.faithfulness_reasoning,
            "answer_relevance": score.answer_relevance,
            "answer_relevance_reasoning": score.answer_relevance_reasoning,
            "context_precision": score.context_precision,
            "context_precision_reasoning": score.context_precision_reasoning,
        })

        print(f"      -> Faithfulness: {score.faithfulness:.2f} | Relevance: {score.answer_relevance:.2f} | Precision: {score.context_precision:.2f}\n")

    n = len(GOLDEN_BENCHMARK)
    mean_faith = total_faithfulness / n
    mean_rel = total_relevance / n
    mean_prec = total_precision / n
    overall_score = (mean_faith + mean_rel + mean_prec) / 3

    print("==================================================================")
    print("                   FINAL BENCHMARK SCORECARD                      ")
    print("==================================================================")
    print(f"  * Mean Faithfulness (Anti-Hallucination): {mean_faith * 100:.1f}%")
    print(f"  * Mean Answer Relevance:                  {mean_rel * 100:.1f}%")
    print(f"  * Mean Context Precision (Retrieval):     {mean_prec * 100:.1f}%")
    print("------------------------------------------------------------------")
    print(f"  * OVERALL RAG TRIAD ACCURACY:             {overall_score * 100:.1f}%")
    print("==================================================================\n")

    output_path = Path("data/evaluation_report.json")
    output_path.parent.mkdir(exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_test_cases": n,
            "mean_faithfulness": mean_faith,
            "mean_answer_relevance": mean_rel,
            "mean_context_precision": mean_prec,
            "overall_accuracy": overall_score
        },
        "test_results": results
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Detailed scientific report saved to: {output_path}")

def print_cached_report():
    report_path = Path("data/evaluation_report.json")
    if not report_path.exists():
        print("No evaluation report found on disk.")
        return
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n==================================================================")
    print("      FinSight AI - Scientific RAG Triad Benchmark Suite         ")
    print("==================================================================\n")
    for r in data["test_results"]:
        print(f"Evaluating: {r['test_case_id']} - {r['category']}...")
        print(f"      Q: '{r['question']}'")
        print(f"      -> Faithfulness: {r['faithfulness']:.2f} | Relevance: {r['answer_relevance']:.2f} | Precision: {r['context_precision']:.2f}\n")

    s = data["summary"]
    print("==================================================================")
    print("                   FINAL BENCHMARK SCORECARD                      ")
    print("==================================================================")
    print(f"  * Mean Faithfulness (Anti-Hallucination): {s['mean_faithfulness'] * 100:.1f}%")
    print(f"  * Mean Answer Relevance:                  {s['mean_answer_relevance'] * 100:.1f}%")
    print(f"  * Mean Context Precision (Retrieval):     {s['mean_context_precision'] * 100:.1f}%")
    print("------------------------------------------------------------------")
    print(f"  * OVERALL RAG TRIAD ACCURACY:             {s['overall_accuracy'] * 100:.1f}%")
    print("==================================================================\n")
    print(f"Detailed scientific report loaded from: {report_path}")

if __name__ == "__main__":
    if "--cached" in sys.argv:
        print_cached_report()
    else:
        try:
            run_rag_benchmark()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"\n[Notice] Google Gemini daily free-tier quota (20 requests/day) reached.")
                print("[Fallback] Displaying verified benchmark report from disk:\n")
                print_cached_report()
            else:
                raise e
