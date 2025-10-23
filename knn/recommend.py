from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])


def get_recommendations_for_user(user_id: int, top_k: int = 5):
    ES_PROFILE_INDEX = 'user_profiles'
    ES_ARTICLE_INDEX = 'articles'

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

    knn_query = {
        "field": "vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 100
    }

    try:
        response = es.search(
            index=ES_ARTICLE_INDEX,
            knn=knn_query,
            _source=["title", "id"],
            size = top_k
        )

        results = []
        print(f"\n Result {user_id}: ")
        for hit in response["hits"]["hits"]:
            title = hit["_source"]["title"]
            score = hit["_score"]
            print(f"- {title} (score={score:.4f})")
            results.append(hit["_source"])

        return results

    except Exception as e:
        print(f"Occur error when recommend for user {user_id}: {e}")
        return []


if __name__ == "__main__":
    recommendations = get_recommendations_for_user(user_id=8, top_k=5)