
""" Production-grade rag evaluation through open-source RAGS framework code snippet used from https://atalupadhyay.wordpress.com/2026/01/30/rag-evaluation-from-bleu-scores-to-production-ready-metrics/"""

from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)
from rrf_reranker import build_ranked_context
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from typing import List,Dict
from datasets import Dataset
import pandas as pd
import json
from pathlib import Path
import pandas as pd
from langchain_openai import ChatOpenAI
BASE_DIR = Path(__file__).resolve().parent.parent

csv_path = BASE_DIR / "HR_Manual_Golden_QA_Dataset.csv"
# Prepare data in RAGAS format
# RAGAS expects a Dataset with specific columns:
# - question: the query
# - answer: the generated answer
# - contexts: list of retrieved chunks
# - ground_truth: (optional) ideal answer for some metrics
import time
from openai import RateLimitError

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

def ragas_eval():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "HR_Manual_Golden_QA_Dataset.csv"
    df=pd.read_csv(file_path)
    qa_data=[]
    for _, row in df.iterrows():

        question = row["question"]
        ground_truth = row["answer"]

        # Your retrieval pipeline
        response, chunks = build_ranked_context(question)

        sample = {
            "question": question,
            "answer": response,
            "contexts": [doc['document'].page_content for doc in chunks],
            "ground_truth": ground_truth,
        }

        qa_data.append(sample)
    eval_dataset = prepare_ragas_data(qa_data)
    # Run RAGAS evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
    answer_similarity,
    answer_correctness 
    ]
    llm=ChatOpenAI(model="gpt-4o")
    embeddings=OpenAIEmbeddings() 
    results = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    print("=== RAGAS Results ===")
    print(results)

ragas_eval()