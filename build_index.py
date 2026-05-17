import os
import json
import faiss
import numpy as np
import pandas as pd

from embedding import create_embedding

DATA_DIR = "data"
INDEX_DIR = "index"

documents = []
filenames = []
metadata = []

for file in os.listdir(DATA_DIR):
    # csv 파일만 처리
    if not file.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, file)

    # csv 읽기
    df = pd.read_csv(path)

    print(f"loading: {file}")

    # 각 상품 처리
    for _, row in df.iterrows():

        # 임베딩용 텍스트 생성
        text = f"""
        {row['product_name']}
        {row['category']}
        {row['description']}
        {row['shopping_mall']}
        {row['price']}
        """

        documents.append(text)

        # 검색 결과용 메타데이터 저장
        metadata.append(
            {
                "id": int(row["id"]),
                "product_name": row["product_name"],
                "shopping_mall": row["shopping_mall"],
                "category": row["category"],
                "description": row["description"],
                "price": int(row["price"]),
                "source_file": file,
            }
        )

# 임베딩 생성
embeddings = create_embedding(documents)

# numpy 변환
embeddings = np.array(embeddings).astype("float32")

# 벡터 차원
dimension = embeddings.shape[1]

# FAISS 인덱스 생성
index = faiss.IndexFlatL2(dimension)

# 벡터 추가
index.add(embeddings)

# 저장 폴더 생성
os.makedirs(INDEX_DIR, exist_ok=True)

# 인덱스 저장
faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

# 메타데이터 저장
with open(f"{INDEX_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("FAISS index 생성 완료")
print(f"총 상품 수: {len(documents)}")
