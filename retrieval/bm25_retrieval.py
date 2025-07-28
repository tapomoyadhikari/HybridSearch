from pyserini.search import SimpleSearcher
import json

searcher = SimpleSearcher('indexes/msmarco-passage')
searcher.set_bm25(k1=1.2, b=0.75)  # Tuned parameters

def retrieve(query, k=100):
    hits = searcher.search(query, k=k)
    return [{"docid": hit.docid, "score": hit.score} for hit in hits]

# Example usage
results = retrieve("What is machine learning?")
with open('bm25_results.json', 'w') as f:
    json.dump(results, f)
