import json

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config import TEST_QUERIES_PATH
from src.rag_pipeline import RAGPipeline


def load_test_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ragas_evaluation():
    tests = load_test_queries(TEST_QUERIES_PATH)
    pipeline = RAGPipeline()

    dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    for item in tests:
        q = item["question"]
        gt_contexts = item.get("contexts", [])
        out = pipeline.answer(q)
        dataset["question"].append(q)
        dataset["answer"].append(out["answer"])
        dataset["contexts"].append([c["document"] for c in out["contexts"]])

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(result)


if __name__ == "__main__":
    run_ragas_evaluation()
