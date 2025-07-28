import numpy as np
from sklearn.metrics import ndcg_score

def compute_mrr(ranked_list, relevant_docs):
    for i, doc in enumerate(ranked_list):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0

# Example: ranked_list = ["doc1", "doc3"], relevant_docs = ["doc3"]
print("MRR@10:", compute_mrr(ranked_list, relevant_docs))
