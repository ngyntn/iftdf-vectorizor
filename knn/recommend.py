import math
from datetime import datetime, timezone

import numpy as np
from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])

ES_ARTICLE_INDEX = 'articles'
ES_PROFILE_INDEX = 'user_profiles'
K_CANDIDATES = 200
NUM_CANDIDATES = 500

W_RELEVANCE = 0.6  # 60% trọng số cho sự liên quan
W_FRESHNESS = 0.4  # 40% trọng số cho sự tươi mới
FRESHNESS_HALF_LIFE_DAYS = 7

def _calculate_freshness_score(created_at_str: str) -> float:
    if not created_at_str:
        return 0.0

    created_at_dt = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
    if created_at_dt.tzinfo is None:
        created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age_in_days = (now - created_at_dt).total_seconds() / (24 * 3600)

    if age_in_days < 0:
        age_in_days = 0

    if FRESHNESS_HALF_LIFE_DAYS <= 0:
        return 1.0 if age_in_days == 0 else 0.0

    # Hệ số decay suy giảm
    decay_rate_lambda = math.log(2) / FRESHNESS_HALF_LIFE_DAYS

    # Công thức suy giảm
    freshness_score = math.exp(-decay_rate_lambda * age_in_days)
    return freshness_score

def get_recommendations_for_user(user_id: int, read_ids: list[int]):

    print(f"List read_ids: {read_ids}")

    try:
        user_profile_doc = es.get(
            index=ES_PROFILE_INDEX,
            id=user_id,
            _source=["profile_vector"]
        )
        query_vector = user_profile_doc["_source"]["profile_vector"]
        print(f"Retrieve vector_profile for user: {user_id} thành công.")
    except Exception as e:
        print(f"Not found vector_profile for user: {user_id}. {e}")
        return []

    # Bắt đầu truy vấn tập ứng viên
    knn_query = {
        "knn": {
            "field": "vector",
            "query_vector": query_vector,
            "k": K_CANDIDATES + 100, # truy vấn thừa để re-ranking và lọc lần cuối
            "num_candidates": NUM_CANDIDATES,
            "filter": {
                "bool": {
                    "must_not": [
                        {"terms": {"id": read_ids}}
                    ]
                }
            }
        },
        "_source": ["created_at", "title"]
    }

    try:
        response = es.search(
            index=ES_ARTICLE_INDEX,
            body={"knn": knn_query['knn']},
            _source=knn_query['_source'],
            size=K_CANDIDATES + 100
        )
        candidates = response['hits']['hits']
    except Exception as e:
        print(f"Occur error when recommend for user {user_id}: {e}")
        return []

    # Tái xếp hạng theo độ mới mẻ
    ranked_candidates = []
    for hit in candidates:
        article_id = int(hit['_id'])
        title = hit['_source'].get('title')
        created_at_str = hit['_source'].get('created_at')

        relevance_score = hit['_score']
        freshness_score = _calculate_freshness_score(created_at_str)

        hybrid_score = (W_RELEVANCE * relevance_score) + (W_FRESHNESS * freshness_score)

        ranked_candidates.append({
            "id": article_id,
            "hybrid_score": hybrid_score,
            "relevance": relevance_score,
            "freshness": freshness_score,
            "title": title,
            "created_at": created_at_str
        })

    ranked_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    ranked_candidates = ranked_candidates[:K_CANDIDATES] # loại bỏ những phần tử xếp hạng cuối danh sách

    print(f"Re-ranking for {user_id}: ")
    for c in ranked_candidates:
        print(f"Article: {c['id']}, title: {c['title'][:30]}, created: {c['created_at']}, relevance: {c['relevance']}, freshness: {c['freshness']}, total: {c['hybrid_score']}")

    final_sorted_ids = [c['id'] for c in ranked_candidates]

    print(f"Total articles: {len(final_sorted_ids)}")
    return final_sorted_ids



if __name__ == "__main__":

    read_ids = list(range(500, 900))
    recommendations = get_recommendations_for_user(user_id=8, read_ids=read_ids)