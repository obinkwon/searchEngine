import json
import faiss
import numpy as np

from embedding import create_embedding

index = faiss.read_index("index/faiss.index")

with open("index/metadata.json", "r", encoding="utf-8") as f:
    filenames = json.load(f)


def search(query, top_k=3):
    query_vector = create_embedding([query])
    
    query_vector = np.array(query_vector).astype("float32") # FAISS가 사용할 수 있는 형태로 데이터를 변환

    distances, indices = index.search(query_vector, top_k)

    results = []

    for idx in indices[0]:
        results.append(filenames[idx])

    return results
