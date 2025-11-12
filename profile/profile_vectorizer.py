import math
import time

import numpy as np
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from elasticsearch import Elasticsearch

from config.db import get_connection, close_connection


BASE_SCORES = {
    'click': 1,
    'read': 3,
    'like': 5,
    'comment': 7,
    'bookmark': 8
}

TARGET_DIMS = 1000 # thay đổi theo tfidf_builder
ES_ARTICLE_INDEX = 'articles'
ES_PROFILE_INDEX = 'user_profiles'

JOB_INTERVAL_MINUTES = 1440  # Chạy mỗi 1 ngay
ROLLING_WINDOW_DAYS = 90  # Chỉ xét action trong 3 tháng gần nhất

es = Elasticsearch(['http://localhost:9200'])

def get_final_score(action: str, created_at: datetime) -> float:

    base_score = BASE_SCORES.get(action, 0)
    if base_score == 0:
        return 0.0

    # Tính hệ số suy giảm theo tuần
    days_passed = (datetime.now() - created_at).days
    weeks_passed = math.floor(days_passed / 7)
    decay_factor = 0.9 ** weeks_passed

    return base_score * decay_factor


def get_active_users(db_connection, minutes_since_last_run: int) -> list[int]:
    # Lấy danh sách các user_id có tương tác mới kể từ lần chạy job trước.
    time_threshold = datetime.now() - timedelta(minutes=minutes_since_last_run)

    query = """
    SELECT DISTINCT user_id 
    FROM user_article_interactions
    WHERE created_at >= %s
    """

    active_users = []
    try:
        print(f"Retrieve user active with created_at > {time_threshold}")
        cursor = db_connection.cursor()
        cursor.execute(query, (time_threshold,))
        active_users = [item[0] for item in cursor.fetchall()]
        cursor.close()
        print(f"Found {len(active_users)} user active to update.")
    except Exception as e:
        print(f"Occur error when query user: {e}")

    return active_users


def update_profile_for_user(user_id: int, db_connection):
    # Tính toán và cập nhật vector hồ sơ cho user
    print(f"Processing user profile: {user_id}")

    # Lấy các tương tác trong 90 ngày
    window_start_date = datetime.now() - timedelta(days=ROLLING_WINDOW_DAYS)
    query = """
    SELECT article_id, action, created_at 
    FROM user_article_interactions
    WHERE user_id = %s AND created_at >= %s
    """

    cursor = db_connection.cursor(dictionary=True)
    cursor.execute(query, (user_id, window_start_date))
    interactions = cursor.fetchall()

    print(f"Interactions of user {user_id} ", interactions)
    cursor.close()

    if not interactions:
        print(f"No interactions with in 90 day of {user_id}. Pass.")
        return

    # Lấy ID bài viết và các vector tương ứng từ Elasticsearch
    article_ids = list(set([row['article_id'] for row in interactions]))

    try:
        response = es.mget(
            index=ES_ARTICLE_INDEX,
            body={'ids': article_ids},
            _source=['vector']
        )

        # Tạo map {article_id: vector} để tra cứu nhanh
        vectors_map = {}
        for doc in response['docs']:
            if doc['found'] and '_source' in doc and 'vector' in doc['_source']:
                vectors_map[int(doc['_id'])] = np.array(doc['_source']['vector'])

    except Exception as e:
        print(f"Occur error when retrieve articles from ES {user_id}: {e}")
        return

    # Tính toán vector hồ sơ Pu (Weighted Average)
    total_weighted_vector = np.zeros(TARGET_DIMS)
    total_score = 0.0

    for interaction in interactions:
        article_id = interaction['article_id']
        vector_Vi = vectors_map.get(article_id)

        # Chỉ xử lý nếu bài viết này có vector
        if vector_Vi is not None:
            final_score = get_final_score(
                interaction['action'],
                interaction['created_at']
            )

            print(f"User {user_id} has final score: {final_score} with articles {article_id}")

            # Chỉ tính các điểm có giá trị (tránh số quá nhỏ)
            if final_score > 0.01:
                print(f"Calculating vector_profile += vector_article({article_id}) * final_score_interact({final_score})")
                total_weighted_vector += (vector_Vi * final_score)
                total_score += final_score

    # Lưu vector hồ sơ Pu vào Elasticsearch
    if total_score > 0:
        profile_vector_Pu = (total_weighted_vector / total_score).tolist()

        doc_body = {
            'user_id': user_id,
            'profile_vector': profile_vector_Pu,
            'last_updated': datetime.now()
        }

        es.index(index=ES_PROFILE_INDEX, id=user_id, body=doc_body)
        print(f"Index ES successfully for vector_profile: {user_id}")
    else:
        print(f"Cannot calculate proper score for user: {user_id}. Pass.")



def ensure_profile_index_exists():
    if not es.indices.exists(index=ES_PROFILE_INDEX):
            es.indices.create(
                index=ES_PROFILE_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "user_id": {"type": "integer"},
                            "profile_vector": {"type": "dense_vector", "dims": TARGET_DIMS},
                            "last_updated": {"type": "date"}
                        }
                    }
            })
            print(f"Created ES index '{ES_PROFILE_INDEX}'")


def profile_update_job():
    ensure_profile_index_exists()

    print(f"[{datetime.now()}] Running profile update job")
    connection = None
    try:
        connection = get_connection()

        # Lấy danh sách user cần cập nhật
        active_users = get_active_users(connection, minutes_since_last_run=JOB_INTERVAL_MINUTES)

        if not active_users:
            print("No user active. End job.")
            return

        # Lặp và cập nhật cho từng user
        for user_id in active_users:
            try:
                update_profile_for_user(user_id, connection)
            except Exception as e:
                print(f"Occur error when udpated user: {user_id}: {e}")

        print(f"[{datetime.now()}] Job update vector user profile is complete.")

    except Exception as e:
        print(f"[{datetime.now()}] Error in profile_update_job: {e}")
    finally:
        if connection:
            close_connection()