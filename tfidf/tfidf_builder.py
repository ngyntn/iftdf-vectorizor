import os
import threading

import numpy as np
from celery import group
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
from elasticsearch import Elasticsearch
from datetime import datetime

from config.db import get_connection, close_connection
from worker import clean_text_batch_task, index_batch_task, process_incremental_batch_task

ES_URL = os.getenv("ES_URL")
ES_ARTICLE_INDEX = 'articles'
TARGET_DIMS = 1000

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


def chunk_list(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]



def update_db_flag_worker(ids):
    if not ids: return
    try:
        conn = get_connection()
        cursor = conn.cursor()
        placeholders = ','.join(['%s'] * len(ids))
        query = f"UPDATE articles SET is_indexed = TRUE WHERE id IN ({placeholders})"
        cursor.execute(query, ids)
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"{[datetime.now()]} Error is_indexed updating DB in worker: {e}")


def index_articles_full_job():
    try:
        get_connection()
        if not index_lock.acquire(blocking=False):
            print("[Full job] Skipped because another index job is running.")
            return

        try:
            print(f"[{datetime.now()}] Running full articles job.")
            articles = query_articles_from_db(full=True)
            if not articles:
                print(f"{[datetime.now()]} No articles found")
                return

            es.delete_by_query(index=ES_ARTICLE_INDEX, body={"query": {"match_all": {}}})
            print(f"[{datetime.now()}] Cleared old data in ES.")

            BATCH_SIZE = 200
            raw_contents = [a['content'] for a in articles]
            batches = chunk_list(raw_contents, BATCH_SIZE)
            print(f"[{datetime.now()}] Dispatching {len(batches)} cleaning batches to Celery...")

            # Bắn task vào hàng đợi và đợi kết quả
            job_group = group(clean_text_batch_task.s(batch) for batch in batches)
            result_group = job_group.apply_async()
            results_matrix = result_group.get() # [[text1, text2, ...],...,[..,text199, text200]]

            cleaned_texts = [text for batch in results_matrix for text in batch]
            print(f"[{datetime.now()}] Cleaning finished. Total: {len(cleaned_texts)} texts.")

            # vectorize
            vectorizer = TfidfVectorizer(max_features=8000, min_df=2)
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

            vocab_size = tfidf_matrix.shape[1]
            n_components = min(TARGET_DIMS, max(50, vocab_size))
            svd = TruncatedSVD(n_components=n_components)
            reduced_vectors = svd.fit_transform(tfidf_matrix)

            # Lưu Models
            joblib.dump(vectorizer, 'model/tfidf_model.pkl')
            joblib.dump(svd, 'model/svd_model.pkl')
            print(f"[{datetime.now()}] Models saved.")

            # Padding Vectors
            current_dims = reduced_vectors.shape[1]
            if current_dims < TARGET_DIMS:
                padding = np.zeros((reduced_vectors.shape[0], TARGET_DIMS - current_dims))
                reduced_vectors = np.hstack((reduced_vectors, padding))

            for i, article in enumerate(articles):
                article['vector'] = reduced_vectors[i].tolist()
                # Xóa content để giảm tải khi gửi qua Redis (chỉ cần vector và metadata)
                del article['content']

            article_batches = chunk_list(articles, BATCH_SIZE)
            print(f"[{datetime.now()}] Dispatching {len(article_batches)} indexing batches to Celery...")

            index_group = group(index_batch_task.s(batch) for batch in article_batches)
            print(f"[{datetime.now()}] Stamp 1")
            result_group = index_group.apply_async()
            print(f"[{datetime.now()}] Stamp 2")
            results_matrix = result_group.get()
            print(f"[{datetime.now()}] Stamp 3")

            success_ids = [uid for batch in results_matrix for uid in batch]

            if success_ids:
                update_db_flag_worker(success_ids)

            print(f"[{datetime.now()}] Finish indexing {len(articles)} artice_vector into ES")

            print(f"[{datetime.now()}] Full Job Successfully.")

        except Exception as e:
            print(f"{[datetime.now()]} Error in Full Job: {e}")
        finally:
            index_lock.release()

    finally:
        close_connection()

def index_articles_incremental_job():

    try:
        get_connection()
        if not index_lock.acquire(blocking=False):
            print("[Incremental job] Skipped because another index job is running.")
            return

        try:
            print(f"[{datetime.now()}] Running incremental articles job.")
            articles = query_articles_from_db(full=False)
            if not articles:
                print(f"{[datetime.now()]} No articles found")
                return

            BATCH_SIZE = 100
            batches = chunk_list(articles, BATCH_SIZE)
            print(f"[{datetime.now()}] Dispatching {len(batches)} incremental batches to Celery...")

            job_group = group(process_incremental_batch_task.s(batch) for batch in batches)
            print(f"[{datetime.now()}] Stamp 1")
            result_group = job_group.apply_async()
            print(f"[{datetime.now()}] Stamp 2")
            results_matrix = result_group.get()
            print(f"[{datetime.now()}] Stamp 3")

            success_ids = [uid for batch in results_matrix for uid in batch]

            if success_ids:
                update_db_flag_worker(success_ids)
            print(f"[{datetime.now()}] Finish indexing {len(articles)} artice_vector into ES")

            print(f"[{datetime.now()}] Incremental Job Successfully.")

        except Exception as e:
            print(f"{[datetime.now()]} Error in Incremental Job: {e}")
        finally:
            index_lock.release()
    finally:
        close_connection()