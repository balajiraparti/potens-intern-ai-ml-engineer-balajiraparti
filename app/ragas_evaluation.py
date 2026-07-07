
""" Production-grade rag evaluation through open-source RAGS framework code snippet used from https://atalupadhyay.wordpress.com/2026/01/30/rag-evaluation-from-bleu-scores-to-production-ready-metrics/"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from typing import List,Dict
from datasets import Dataset
import pandas as pd
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

csv_path = BASE_DIR / "HR_Manual_Golden_QA_Dataset.csv"
# Prepare data in RAGAS format
# RAGAS expects a Dataset with specific columns:
# - question: the query
# - answer: the generated answer
# - contexts: list of retrieved chunks
# - ground_truth: (optional) ideal answer for some metrics
def prepare_ragas_data(qa_pairs: List[Dict]) -> Dataset:
    """
    Convert QA results to RAGAS format
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []  # Optional but needed for context_recall and answer_correctness
    }
    for pair in qa_pairs:
        data["question"].append(pair["question"])
        data["answer"].append(pair["answer"])
        data["contexts"].append(pair["contexts"])
        data["ground_truth"].append(pair.get("ground_truth", ""))
    return Dataset.from_dict(data)
# Create evaluation dataset
# In practice, you'd have ground truth answers from human annotators

def ragas_eval(question,answer,context,ground_truth):
    qa_data = [
        {
            "question":question,
            "answer": answer,
            "contexts": [doc.page_content for doc in context],
            "ground_truth": ground_truth
                            },
 # list of Q&A prepared by human to evaluate the system 
    ]
    eval_dataset = prepare_ragas_data(qa_data)
    # Run RAGAS evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        # context_recall requires ground truth contexts, not just answers
    ]
    llm=ChatMistralAI(model="mistral-small-latest",temperature=.3)
    embeddings=OpenAIEmbeddings() 
    results = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    print("=== RAGAS Results ===")
    print(results)