import os
import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
from elasticsearch import Elasticsearch
from datetime import datetime

from config.db import get_connection, close_connection
from preprocess.eng_processor import clean_text

ES_URL = os.getenv("ES_URL")
TARGET_DIMS = 1000
ES_ARTICLE_INDEX = 'articles'

index_lock = threading.Lock()
es = Elasticsearch([ES_URL])

def query_articles_from_db(full=True):
    connection = get_connection()
    cursor = connection.cursor()
    print(f"[{datetime.now()}] Start query {"full articles" if full else "articles is not indexed"}")

    base_query = "SELECT id, title, content, created_at FROM articles WHERE moderation_status='public'"
    query = f"{base_query} AND is_indexed=false" if not full else base_query

    cursor.execute(query)
    rows = cursor.fetchall()
    articles = [{"id": row[0], "title": row[1], "content": row[1] + " " + row[2], "created_at": row[3]} for row in rows]

    print(f"[{datetime.now()}] Queried {len(articles)} articles from DB.")
    return articles

def update_indexed_flag(success_ids):
    if not success_ids:
        print("No article ids to update flag.")
        return
    connection = get_connection()
    cursor = connection.cursor()
    placeholders = ','.join(['%s'] * len(success_ids))
    query = f"UPDATE articles SET is_indexed = TRUE WHERE id IN ({placeholders})"
    cursor.execute(query, success_ids)

    connection.commit()
    updated_count = cursor.rowcount
    cursor.close()
    print(f"[{datetime.now()}] Updated is_indexed=true for {len(success_ids)} articles in DB.")


# Vectorization và Reduction (Tăng max_features cho vocab tiếng Anh phong phú)
def vectorize_articles(articles, full=True):

    if not articles:
        print(f"[{datetime.now()}] No articles to vectorize")
        return articles

    cleaned_texts = [clean_text(article['content']) for article in articles]
    print(f"[{datetime.now()}] Cleaned text from {len(cleaned_texts)} articles")

    if full:
        # Full: Fit mới và transform
        print(f"[{datetime.now()}] Fitting new TF-IDF and SVD models (full mode)")

        # TF-IDF với giới hạn features cao hơn
        vectorizer = TfidfVectorizer(max_features=8000, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        vocab_size = tfidf_matrix.shape[1]

        # Giảm chiều (giữ 1000 dims cho ES)
        n_components = min(TARGET_DIMS, max(50, vocab_size))
        svd = TruncatedSVD(n_components=n_components)
        reduced_vectors = svd.fit_transform(tfidf_matrix)

        # Lưu models
        joblib.dump(vectorizer, 'model/tfidf_model.pkl')  # Đường dẫn dựa trên CWD (vị trí file thực thi)
        joblib.dump(svd, 'model/svd_model.pkl')
        print(f"[{datetime.now()}] Saved models. Vocab size: {len(vectorizer.get_feature_names_out())}")

    else:
        # Incremental: Load và transform
        print(f"[{datetime.now()}] Loading TF-IDF and SVD models (incremental mode)")

        # Tải models
        vectorizer = joblib.load('model/tfidf_model.pkl')
        svd = joblib.load('model/svd_model.pkl')

        # Chỉ transform không fit
        tfidf_matrix = vectorizer.transform(cleaned_texts)  # Chỉ transform
        reduced_vectors = svd.transform(tfidf_matrix)

    # Đệm vector cho đủ chiều
    current_dims = reduced_vectors.shape[1]
    if current_dims < TARGET_DIMS:
        print(f"[{datetime.now()}] Padding vectors from {current_dims} to {TARGET_DIMS} dims.")
        # Tạo mảng zero để đệm
        padding = np.zeros((reduced_vectors.shape[0], TARGET_DIMS - current_dims))
        # Ghép mảng kết quả SVD với mảng zero
        reduced_vectors = np.hstack((reduced_vectors, padding))


    # Gán vector cho articles
    for i, article in enumerate(articles):
        article['vector'] = reduced_vectors[i].tolist()
        article['dims'] = reduced_vectors.shape[1]

    print(
        f"[{datetime.now()}] Vectorized {len(articles)} articles reduced to {reduced_vectors.shape[1]} dims")
    return articles



# Index vào Elasticsearch
def index_to_es(articles, full=True):
    if not articles:
        print(f"[{datetime.now()}] No articles to index")

    dims = articles[0]['dims']

    # Tạo index nếu chưa có
    if not es.indices.exists(index=ES_ARTICLE_INDEX):
        es.indices.create(
            index=ES_ARTICLE_INDEX,
            body={
                "mappings": {
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "text"},
                        "created_at": {"type": "date"},
                        "vector": {"type": "dense_vector", "dims": dims}
                    }
                }
            }
        )
        print(f"[{datetime.now()}] ES index {ES_ARTICLE_INDEX} created.")

    if full:
        # Full: Xóa tất cả cũ
        print(f"[{datetime.now()}] Full mode: Deleting old data in ES.")
        es.delete_by_query(index=ES_ARTICLE_INDEX, body={"query": {"match_all": {}}})

    success_ids = []
    for article in articles:
        try:
            doc = {
                "id": article['id'],
                "title": article['title'],
                "created_at": article['created_at'],
                "vector": article['vector']
            }
            es.index(index=ES_ARTICLE_INDEX, id=article['id'], body=doc)
            success_ids.append(article['id'])
            print(f"[{datetime.now()}] Indexed article {article['id']} to ES.")
        except Exception as e:
            print(f"[{datetime.now()}] Error indexing {article['id']}: {e}")

    if success_ids:
        print(f"[{datetime.now()}] Successfully indexed {len(success_ids)} articles.")

    return success_ids



def index_articles_full_job():
    if not index_lock.acquire(blocking=False):
        print("[Full job] Skipped because another index job is running.")
        return

    try:
        print(f"[{datetime.now()}] Running full articles job.")
        get_connection()

        try:
            articles = query_articles_from_db(full=True)
            articles = vectorize_articles(articles, full=True)
            success_ids = index_to_es(articles, full=True)
            update_indexed_flag(success_ids)

            print(f"[{datetime.now()}] Full job completed: {len(success_ids)} articles processed.")
        except Exception as e:
            print(f"[{datetime.now()}] Error in full job: {e}.")

        finally:
            close_connection()
    finally:
        index_lock.release()

def index_articles_incremental_job():
    if not index_lock.acquire(blocking=False):
        print("[Incremental job] Skipped because another index job is running.")
        return

    try:
        print(f"[{datetime.now()}] Running incremental articles job.")
        get_connection()

        try:
            articles = query_articles_from_db(full=False)
            articles = vectorize_articles(articles, full=False)
            success_ids = index_to_es(articles, full=False)
            update_indexed_flag(success_ids)

            print(f"[{datetime.now()}] Incremental job completed: {len(success_ids)} articles processed.")
        except Exception as e:
            print(f"[{datetime.now()}] Error in incremental job: {e}")
        finally:
            close_connection()
    finally:
        index_lock.release()