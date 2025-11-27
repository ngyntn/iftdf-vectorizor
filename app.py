
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, Blueprint, request, jsonify

from knn.query import knn_text_search
from knn.recommend import get_recommendations_for_user
from profile.profile_vectorizer import profile_update_job
from tfidf.tfidf_builder import index_articles_full_job, index_articles_incremental_job

app = Flask(__name__)


scheduler = BackgroundScheduler()

scheduler.add_job(index_articles_full_job, 'interval', seconds=86400, id='full_index_job', max_instances=1, coalesce=True) # 1 day
scheduler.add_job(index_articles_incremental_job, 'interval', seconds=3600, id='incremental_index_job', max_instances=1, coalesce=True) # 1 hour
scheduler.add_job(profile_update_job,'interval', seconds=60, id='profile_job', max_instances=1, coalesce=True) # 1 hour

scheduler.start()


search_bp = Blueprint("search", __name__)


def paginate_results(results, page: int, size: int):
    start = (page - 1) * size
    end = start + size
    return results[start:end]


@app.route("/")
def home():
    return "Hello Flask"

@app.route("/articles/search/knn", methods=["GET"])
def search_knn():

    key = request.args.get("key", "").strip()
    page = int(request.args.get("page", 1))
    size = int(request.args.get("size", 10))
    top_k = page * size + 20

    if not key:
        return jsonify({
            "status": "error",
            "error": {
                "code": "1111",
                "message": "Missing 'key' parameter"
            }
        }), 400

    try:
        results = knn_text_search(key, top_k=top_k)
        paginated = paginate_results(results, page, size)

        return jsonify({
            "status": "success",
            "data": {
                "page": page,
                "size": size,
                "results": paginated
            }
        })
    except Exception as e:
        print(f"Occur error when search with key {key}: {str(e)}")
        return jsonify({
            "status": "error",
            "error": {
                "code": "1112",
                "message": e.args[0] if e.args else str(e).split('\n')[0] # compact message
            }
        }), 500

@app.route("/articles/recommend", methods=["POST"])
def recommend ():

    data = request.get_json()
    user_id = int(data.get("user_id"))
    read_ids = data.get("read_ids", [])

    if not user_id:
        print("Missing 'user' parameter")
        return jsonify({
            "status": "error",
            "error": {
                "code": "1113",
                "message": "Missing 'user' paramenter"
            }
        }), 400

    try:
        results = get_recommendations_for_user(user_id, read_ids)

        return jsonify({
            "status": "success",
            "data": {
                "results": results
            }
        })
    except Exception as e:
        print(f"Occur error when recommend articles for user {user_id}: {str(e)}")
        return jsonify({
            "status": "error",
            "error": {
                "code": "1112",
                "message": e.args[0] if e.args else str(e).split('\n')[0] # compact message
            }
        }), 500



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
