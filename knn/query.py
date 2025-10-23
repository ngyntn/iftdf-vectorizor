

import joblib
from elasticsearch import Elasticsearch
from preprocess.eng_processor import clean_text


es = Elasticsearch(['http://localhost:9200'])

# Load models để transform text query
vectorizer = joblib.load('../tfidf/tfidf_model.pkl')
svd = joblib.load('../tfidf/svd_model.pkl')


def transform_query_to_vector(query_text):
    cleaned_query = clean_text(query_text)
    query_tfidf = vectorizer.transform([cleaned_query])
    query_reduced = svd.transform(query_tfidf)
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
        _source=["article_id", "title"]
    )

    # In kết quả
    print("\n Results :", query_text)
    for hit in response["hits"]["hits"]:
        title = hit["_source"]["title"]
        score = hit["_score"]
        print(f"- {title} (score={score:.4f})")



if __name__ == "__main__":
    search_query = "Football"
    print(f"Searching for: '{search_query}'")

    recs = knn_text_search(search_query, top_k=5)