
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

from tfidf.tfidf_builder import query_articles_from_db, vectorize_articles, index_to_es, index_articles_job

app = Flask(__name__)

# Khởi tạo scheduler
scheduler = BackgroundScheduler()

# Mode: chạy mỗi 1 phút
scheduler.add_job(index_articles_job, 'interval', seconds=10, id='index_job')

scheduler.start()

@app.route("/")
def home():
    return "Hello Flask"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
