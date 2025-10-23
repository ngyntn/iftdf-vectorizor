from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from tfidf.tfidf_builder import query_articles_from_db, vectorize_articles, index_to_es

app = Flask(__name__)

def index_articles_job():
    print(f"[{datetime.now()}] Running vectorized article job")
    articles = query_articles_from_db()
    articles = vectorize_articles(articles)
    index_to_es(articles)
    print(f"[{datetime.now()}] Vectorized article job successfully.\n")

# Khởi tạo scheduler
# scheduler = BackgroundScheduler()


# MODE: chạy mỗi 1 phút
# scheduler.add_job(index_articles_job, 'interval', minutes=1, id='index_job')

# MODE: chạy lúc 2h sáng mỗi ngày
# scheduler.add_job(index_articles_job, 'cron', hour=2, minute=0, id='index_job')

# scheduler.start()
#
# @app.route("/")
# def home():
#     return "Hello Flask"

if __name__ == "__main__":
    # app.run(debug=True)
    index_articles_job()

