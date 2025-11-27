
import os
from datetime import datetime

import joblib
import numpy as np
from celery import Celery
from elasticsearch import Elasticsearch, helpers
from config.db import get_connection, close_connection


from preprocess.eng_processor import clean_text

REDIS_URL = os.getenv("REDIS_URL")
ES_URL = os.getenv("ES_URL")
ES_ARTICLE_INDEX = 'articles'
TARGET_DIMS = 1000


celery = Celery('indexer', broker=REDIS_URL, backend=REDIS_URL)

celery.conf.update(
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_serializer='json',
    accept_content=['json'],
    broker_connection_retry_on_startup=True,

    # redis_socket_keepalive=True,
    # redis_socket_connect_timeout=30,
    # redis_socket_timeout=None,
)

loaded_vectorizer = None
loaded_svd = None

es = Elasticsearch([ES_URL])

def load_models_worker():
    global loaded_vectorizer, loaded_svd
    if loaded_vectorizer is None:
        try:
            loaded_vectorizer = joblib.load('model/tfidf_model.pkl')
            loaded_svd = joblib.load('model/svd_model.pkl')
            print(f"{[datetime.now()]} Worker loaded models successfully")
        except Exception as e:
            print(f"{[datetime.now()]} Worker failed to load models: {e}")


# task này xử lý raw text -> clean text theo từng batch
@celery.task(name="clean_text_batch")
def clean_text_batch_task(raw_contents):
    print(f"{[datetime.now()]} Cleaning a batch of {len(raw_contents)} contents")
    return [clean_text(content) for content in raw_contents]


# task này xử lý bulk index vector into ES theo từng batch
@celery.task(name="index_batch")
def index_batch_task(articles_with_vectors):
    actions = []
    success_ids = []

    for art in articles_with_vectors:
        action = {
            "_index": ES_ARTICLE_INDEX,
            "_id": art['id'],
            "_source": {
                "id": art['id'],
                "title": art['title'],
                "created_at": art['created_at'],
                "vector": art['vector']
            }
        }

        actions.append(action)
        success_ids.append(art['id'])

    if actions:
        # bulk index
        helpers.bulk(es, actions)
        print(f"{[datetime.now()]} Indexed a batch of {len(success_ids)} articles into ES")


    return success_ids

# task này xử lý incremental job theo từng batch
@celery.task(name="process_incremental_batch")
def process_incremental_batch_task(articles_raws):

    load_models_worker()
    if not loaded_vectorizer:
        return 0

    cleaned_texts = [clean_text(a['content']) for a in articles_raws]

    tfidf_matrix = loaded_vectorizer.transform(cleaned_texts)
    reduced_vectors = loaded_svd.transform(tfidf_matrix)

    # Padding nếu thiếu chiều
    current_dims = reduced_vectors.shape[1]
    if current_dims < TARGET_DIMS:
        padding = np.zeros((reduced_vectors.shape[0], TARGET_DIMS - current_dims))
        reduced_vectors = np.hstack((reduced_vectors, padding))

    # Index to ES
    actions = []
    success_ids = []

    for i, art in enumerate(articles_raws):
        vector = reduced_vectors[i].tolist()

        action = {
            "_index": ES_ARTICLE_INDEX,
            "_id": art['id'],
            "_source": {
                "id": art['id'],
                "title": art['title'],
                "created_at": art['created_at'],
                "vector": vector
            }
        }

        actions.append(action)
        success_ids.append(art['id'])

    if actions:
        helpers.bulk(es, actions)
        print(f"{[datetime.now()]} Indexed a batch of {len(success_ids)} articles into ES")


    return success_ids

