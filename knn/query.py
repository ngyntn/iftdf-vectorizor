

import joblib
import numpy as np
from elasticsearch import Elasticsearch
from preprocess.eng_processor import clean_text


es = Elasticsearch(['http://localhost:9200'])

# Load models để transform text query
vectorizer = joblib.load('model/tfidf_model.pkl')
svd = joblib.load('model/svd_model.pkl')

TARGET_DIMS = 1000


def transform_query_to_vector(query_text):
    cleaned_query = clean_text(query_text)
    query_tfidf = vectorizer.transform([cleaned_query])
    query_reduced = svd.transform(query_tfidf)

    current_dims = query_reduced.shape[1]
    if current_dims < TARGET_DIMS:
        padding = np.zeros((query_reduced.shape[0], TARGET_DIMS - current_dims))
        query_reduced = np.hstack((query_reduced, padding))

    return query_reduced[0].tolist()


def knn_text_search(query_text, top_k=5, index_name="articles"):
    # Transform query thành vector
    query_vector = transform_query_to_vector(query_text)
    print(f"Query vector shape: {len(query_vector)} dims")

    query = {
        "field": "vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 100
    }

    response = es.search(
        index=index_name,
        knn=query,
        _source=["id", "title"],
        size = top_k
    )

    # In kết quả
    results = []
    print(f"\n Results for top_k {top_k} :{query_text}")
    for hit in response["hits"]["hits"]:
        article_id = hit["_source"].get("id", "")
        title = hit["_source"]["title"]
        score = hit["_score"]
        print(f"- {title} (score={score:.4f})")

        results.append({
            "id": article_id,
            "title": title,
            "score": round(score, 4)
        })

    return results


if __name__ == "__main__":
    search_query = ""
    print(f"Searching for: '{search_query}'")

    recs = knn_text_search(search_query, top_k=5)